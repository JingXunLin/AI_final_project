# mini_metro_edge_env.py
import os
import sys

# 假设你的项目结构是 envs/mini_metro_edge_env.py 和 src/config.py 等
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src')) # 更通用的路径添加

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
import logging

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 游戏特定导入 ---
try:
    from config import (
        num_metros, num_paths, num_stations_max, station_spawning_interval_step,
        framerate, screen_color, screen_width, screen_height,
        station_grid_size, station_capacity,
        # passenger_max_wait_time_ms # 需要确保Station类能访问到这个或其等价值
    )
    NUM_HORIZONTAL_GRIDS, NUM_VERTICAL_GRIDS = station_grid_size
    STATION_MAX_CAPACITY = station_capacity
    from geometry.type import ShapeType
    from mediator import Mediator, MeditatorState
    from visuals.background import draw_waves
    from entity.station import Station # 假设可以导入Station类用于类型提示
except ImportError as e:
    logger.critical(f"无法导入必要的模块或配置: {e}. 请确保 PYTHONPATH 和配置文件正确。")
    raise

# --- Stable Baselines3 ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    logger.warning("stable_baselines3 未安装或无法导入。训练和评估脚本部分将无法运行。")
    PPO = None


class MiniMetroEdgeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': framerate if 'framerate' in globals() and framerate else 30}

    def __init__(self, gamespeed: int = 1, visuals: bool = False, is_eval_env: bool = False):
        super().__init__()
        logger.info("Initializing MiniMetroEdgeEnv...")

        if not pygame.font.get_init(): pygame.font.init()
        if visuals and not pygame.get_init(): pygame.init()

        self.mediator = Mediator(gamespeed=gamespeed, gen_stations_first=False)

        self.max_station_slots = num_stations_max
        self.max_paths = num_paths
        try:
            self.shape_types_enum: List[ShapeType] = list(ShapeType)
        except TypeError:
             logger.warning("ShapeType might not be a standard Enum or list(). Using fallback.")
             self.shape_types_enum = [s for s in ShapeType] if hasattr(ShapeType, '__iter__') else []
        self.num_shape_types = len(self.shape_types_enum)
        if self.num_shape_types == 0: logger.critical("num_shape_types is 0!")

        self.visuals = visuals
        self.is_eval_env = is_eval_env
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        if self.visuals:
            if not pygame.display.get_init(): pygame.display.init()
            try:
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Mini Metro RL (Edge Env)")
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                logger.warning(f"Could not initialize Pygame display: {e}. Visuals disabled.")
                self.visuals = False

        # --- 动作空间 ---
        # 0: NO_OP
        # 1 to N_PAIRS: ADD_EDGE(sA, sB)
        # N_PAIRS+1 to 2*N_PAIRS: DELETE_EDGE(sA, sB)
        # sA != sB. 我们将为每个有序对 (sA, sB) 创建动作。
        self.num_station_pairs = self.max_station_slots * (self.max_station_slots - 1)
        self.action_space = spaces.Discrete(1 + 2 * self.num_station_pairs)
        
        # 用于将扁平化的 pair_idx 映射回 (sA, sB)
        self._action_to_edge_map: Dict[int, Tuple[int, int]] = {}
        current_idx = 0
        for sA in range(self.max_station_slots):
            for sB in range(self.max_station_slots):
                if sA == sB:
                    continue
                self._action_to_edge_map[current_idx] = (sA, sB)
                current_idx += 1
        
        logger.debug(f"Action space size: {self.action_space.n}")
        logger.debug(f"  NO_OP: 0")
        logger.debug(f"  ADD_EDGE actions: 1 to {self.num_station_pairs}")
        logger.debug(f"  DELETE_EDGE actions: {self.num_station_pairs + 1} to {self.num_station_pairs * 2}")


        # --- 状态空间 (观测空间) ---
        # 1. 站点特征
        #    exists(1), shape_one_hot(num_shapes), pass_count_norm(4), timeout_pressure(1), dest_profile(num_shapes)
        self.station_feature_size = 1 + self.num_shape_types + 4 + 1 + self.num_shape_types
        obs_size_stations = self.max_station_slots * self.station_feature_size

        # 2. 邻接矩阵 (max_station_slots x max_station_slots)
        obs_size_adj_matrix = self.max_station_slots * self.max_station_slots
        
        # 3. 全局特征
        #    num_active_paths_norm(1), avg_path_len_norm(1), score_norm(1), total_pass_waiting_norm(1)
        obs_size_global = 4
        
        self.total_obs_size = obs_size_stations + obs_size_adj_matrix + obs_size_global
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_obs_size,), dtype=np.float32)
        logger.info(f"Initialized Observation Space Shape: ({self.total_obs_size},)")
        logger.debug(f"  Station features per station: {self.station_feature_size}")

        self.last_score = 0
        self.current_rl_step = 0
        self.game_ticks_per_rl_step = 20 # 可以调整，增加让每个RL动作后的游戏演化更多
        # self.max_rl_steps_per_episode 已移除，不再基于步数截断

    def _get_station_by_slot_idx(self, slot_idx: int) -> Optional[Station]:
        """安全地获取指定槽位的站点对象，如果存在的话。"""
        if 0 <= slot_idx < len(self.mediator.stations):
            return self.mediator.stations[slot_idx] # type: ignore
        return None

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.total_obs_size, dtype=np.float32)
        obs_idx_ptr = 0

        # --- 1. 站点特征 ---
        for s_slot_idx in range(self.max_station_slots):
            station_obj = self._get_station_by_slot_idx(s_slot_idx)
            station_exists = station_obj is not None
            
            obs[obs_idx_ptr] = 1.0 if station_exists else 0.0; obs_idx_ptr += 1 # Exists

            if station_exists:
                # Shape Type (One-Hot)
                shape_one_hot = np.zeros(self.num_shape_types, dtype=np.float32)
                if self.num_shape_types > 0:
                    try: 
                        s_type_idx = self.shape_types_enum.index(station_obj.shape.type)
                        shape_one_hot[s_type_idx] = 1.0
                    except ValueError: logger.warning(f"S{s_slot_idx} 形状 {station_obj.shape.type} 未知.")
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = shape_one_hot
                obs_idx_ptr += self.num_shape_types
                
                # Passenger Count (Normalized)
                _station_max_cap = STATION_MAX_CAPACITY if 'STATION_MAX_CAPACITY' in globals() and STATION_MAX_CAPACITY > 0 else 1.0
                passenger_cnt = [0] * 4
                for p in station_obj.passengers:
                    passenger_cnt[int(p.destination_shape.type.value)-1] += 1
                obs[obs_idx_ptr: obs_idx_ptr+4] = np.asarray(passenger_cnt, dtype=np.float32) / _station_max_cap; obs_idx_ptr += 4
                
                # Timeout Pressure (Normalized)
                acc_penalty_ms = getattr(station_obj, 'time_inactive_penalty_acc_ms', 0.0)
                threshold_ms = getattr(station_obj, 'time_inactive_penalty_threshold_ms', 1.0) 
                if threshold_ms <= 0: threshold_ms = 1.0 
                obs[obs_idx_ptr] = acc_penalty_ms / threshold_ms; obs_idx_ptr += 1
                
                # Passenger Destination Profile
                pass_dest_profile = np.zeros(self.num_shape_types, dtype=np.float32)
                if len(station_obj.passengers) > 0 and self.num_shape_types > 0:
                    for passenger in station_obj.passengers:
                        try:
                            dest_type_idx = self.shape_types_enum.index(passenger.destination_shape.type)
                            pass_dest_profile[dest_type_idx] += 1.0
                        except ValueError: logger.warning(f"乘客目标形状 {passenger.destination_shape.type} 未知.")
                    pass_dest_profile /= (len(station_obj.passengers) + 1e-6) 
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = pass_dest_profile
                obs_idx_ptr += self.num_shape_types
            else: # 站点槽位为空
                obs[obs_idx_ptr : obs_idx_ptr + (self.station_feature_size - 1)] = 0.0 # -1 because 'exists' is already set
                obs_idx_ptr += (self.station_feature_size - 1)

        # --- 2. 邻接矩阵 ---
        # adj[sA_slot_idx, sB_slot_idx] = 1.0 if sA -> sB is a segment on any path
        adj_matrix = np.zeros((self.max_station_slots, self.max_station_slots), dtype=np.float32)
        # 我们需要一个从 Station 对象到其观测槽位索引的映射
        station_obj_to_slot_idx_map: Dict[int, int] = {
            id(self.mediator.stations[i]): i
            for i in range(len(self.mediator.stations)) if i < self.max_station_slots
        }
        for path_idx, path in enumerate(self.mediator.paths):
            for i in range(len(path.stations) - 1):
                s_obj_A = path.stations[i]
                s_obj_B = path.stations[i+1]
                s_slot_A = station_obj_to_slot_idx_map.get(id(s_obj_A), -1)
                s_slot_B = station_obj_to_slot_idx_map.get(id(s_obj_B), -1)
                if s_slot_A != -1 and s_slot_B != -1:
                    adj_matrix[s_slot_A, s_slot_B] = path_idx
            if path.is_looped and len(path.stations) > 1: # 连接最后一个到第一个
                s_obj_Last = path.stations[-1]
                s_obj_First = path.stations[0]
                s_slot_Last = station_obj_to_slot_idx_map.get(id(s_obj_Last), -1)
                s_slot_First = station_obj_to_slot_idx_map.get(id(s_obj_First), -1)
                if s_slot_Last != -1 and s_slot_First != -1:
                    adj_matrix[s_slot_Last, s_slot_First] = path_idx
        
        obs[obs_idx_ptr : obs_idx_ptr + adj_matrix.size] = adj_matrix.flatten()
        obs_idx_ptr += adj_matrix.size

        # --- 3. 全局特征 ---
        obs[obs_idx_ptr] = len(self.mediator.paths) / (self.max_paths + 1e-6); obs_idx_ptr += 1 # num_active_paths_norm
        
        total_path_len = sum(len(p.stations) for p in self.mediator.paths)
        avg_path_len = total_path_len / (len(self.mediator.paths) + 1e-6) if self.mediator.paths else 0
        obs[obs_idx_ptr] = avg_path_len / (self.max_station_slots + 1e-6); obs_idx_ptr += 1 # avg_path_len_norm
        
        obs[obs_idx_ptr] = self.mediator.score / (self.mediator.steps + 1e-5) if self.mediator.steps > 0 else 0.0; obs_idx_ptr += 1 # score_norm
        
        total_pass_waiting = sum(len(s.passengers) for s in self.mediator.stations)
        max_possible_pass_system = self.max_station_slots * (STATION_MAX_CAPACITY if 'STATION_MAX_CAPACITY' in globals() else 10) + 1e-6
        obs[obs_idx_ptr] = total_pass_waiting / max_possible_pass_system if max_possible_pass_system > 0 else 0.0; obs_idx_ptr += 1 # total_pass_waiting_overall_norm
        
        if obs_idx_ptr != self.total_obs_size:
            logger.critical(f"观测向量大小不匹配! 预期 {self.total_obs_size}, 得到 {obs_idx_ptr}.")
            # 进行修正以防止崩溃，但这表示 _get_obs 或 __init__ 中的计算有误
            if obs_idx_ptr < self.total_obs_size: obs[obs_idx_ptr:] = 0.0
            else: obs = obs[:self.total_obs_size]
            logger.warning(f"已修正观测向量长度为 {len(obs)}")

        return np.clip(obs, -1.0, 1.0)

    def _get_edge_from_action(self, action_value: int) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """ 将扁平化的动作值转换为 (mode, sA_slot_idx, sB_slot_idx) """
        if action_value == 0:
            return "NO_OP", None, None
        
        # 动作从1开始计数有效边操作
        effective_action_idx = action_value - 1
        
        if effective_action_idx < self.num_station_pairs: # ADD_EDGE
            mode = "ADD_EDGE"
            pair_flat_idx = effective_action_idx
        elif effective_action_idx < 2 * self.num_station_pairs: # DELETE_EDGE
            mode = "DELETE_EDGE"
            pair_flat_idx = effective_action_idx - self.num_station_pairs
        else: # 无效动作值 (理论上不应发生，因为action_space.sample()在此范围内)
            logger.error(f"无效的内部动作索引: {effective_action_idx}")
            return "INVALID", None, None

        sA_slot_idx, sB_slot_idx = self._action_to_edge_map.get(pair_flat_idx, (None, None)) #type: ignore
        if sA_slot_idx is None: # 映射表中未找到 (理论上不应发生)
             logger.error(f"无法从 pair_flat_idx {pair_flat_idx} 映射到站点对。")
             return "INVALID", None, None
        return mode, sA_slot_idx, sB_slot_idx


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is None and self.is_eval_env: seed = 42
        if seed is not None:
            self.mediator.seed = seed; random.seed(seed); np.random.seed(seed)
            logger.info(f"环境已使用种子 {seed} 重置。")
        self.mediator.reset_progress()
        self.last_score = 0; self.current_rl_step = 0
        
        initial_dt_ms = 1000 // (self.metadata['render_fps'] or 30)
        for _ in range(20):
            if len(self.mediator.stations) >= min(3, self.max_station_slots): break
            self.mediator.increment_time(initial_dt_ms)
        if len(self.mediator.stations) > 0: self.mediator.increment_time(1)
        
        observation = self._get_obs()
        info = self._get_info()
        self.last_score = self.mediator.score
        #logger.info(f"环境重置完成。初始分数: {self.last_score}")
        return observation, info

    def step(self, action: int):
        logger.debug(f"RL 步数: {self.current_rl_step}, 接收到原始动作: {action}")
        reward_for_action_logic = 0.0 # 动作执行逻辑本身的奖励/惩罚
        terminated = False
        truncated = False # 取消了基于步数的截断
        invalid_action_penalty = -0.1 # 对无效操作的惩罚

        mode, sA_slot_idx, sB_slot_idx = self._get_edge_from_action(action)
        logger.debug(f"解析动作: mode={mode}, sA_slot={sA_slot_idx}, sB_slot={sB_slot_idx}")

        action_executed_successfully = False
        if mode == "NO_OP":
            action_executed_successfully = True
        elif mode == "ADD_EDGE":
            action_executed_successfully = False # 重置标志
            if sA_slot_idx is not None and sB_slot_idx is not None:
                station_A_obj = self._get_station_by_slot_idx(sA_slot_idx)
                station_B_obj = self._get_station_by_slot_idx(sB_slot_idx)

                if station_A_obj and station_B_obj and station_A_obj != station_B_obj:
                    #logger.debug(f"尝试 ADD_EDGE 操作在 S{sA_slot_idx} ({station_A_obj.id if hasattr(station_A_obj, 'id') else 'N/A'}) 和 S{sB_slot_idx} ({station_B_obj.id if hasattr(station_B_obj, 'id') else 'N/A'}) 之间")

                    path_extended = False
                    # --- 1. 尝试扩展现有线路 ---
                    # 优先考虑 sA 作为某条线路的尾部，添加 sB
                    for path_to_potentially_extend in list(self.mediator.paths): # 使用 list() 复制以允许在迭代中修改
                        if not path_to_potentially_extend.stations: # 空路径，跳过
                            continue
                        
                        # 检查 sA 是否是该路径的尾部
                        if path_to_potentially_extend.stations[-1] == station_A_obj:
                            # 检查 sB 是否已在该路径上 (避免形成简单的内部环或重复添加)
                            if station_B_obj not in path_to_potentially_extend.stations:
                                #logger.info(f"找到可扩展的路径 (ID: {path_to_potentially_extend.id if hasattr(path_to_potentially_extend, 'id') else 'N/A'})，尾部为 S{sA_slot_idx}。尝试添加 S{sB_slot_idx}。")
                                
                                original_stations_in_path = list(path_to_potentially_extend.stations)
                                # original_is_loop = path_to_potentially_extend.is_looped # is_looped 会在 finish_path_creation 中根据情况重新判断

                                self.mediator.cancel_path(path_to_potentially_extend)
                                #logger.debug(f"  原路径已取消。开始重建并扩展...")

                                # 确保所有原始站点仍然有效
                                current_valid_original_stations = []
                                for s_orig in original_stations_in_path:
                                    found = self._get_station_by_slot_idx(self.mediator.stations.index(s_orig)) if s_orig in self.mediator.stations else None
                                    if found:
                                        current_valid_original_stations.append(found)
                                    else:
                                        # logger.warning(f"  重建时，原路径中的站点 {s_orig.id if hasattr(s_orig, 'id') else 'N/A'} 已失效。扩展中止。")
                                        # 尝试恢复被错误取消的路径 (可选的高级恢复逻辑)
                                        # self._try_recreate_path_safely(original_stations_in_path, original_is_loop)
                                        break 
                                if len(current_valid_original_stations) != len(original_stations_in_path):
                                    path_extended = False # 标记扩展失败，可能会尝试创建新路径
                                    break # 跳出外层 for 循环

                                self.mediator.start_path_on_station(current_valid_original_stations[0])
                                for s_intermediate in current_valid_original_stations[1:]:
                                    self.mediator.add_station_to_path(s_intermediate)
                                self.mediator.add_station_to_path(station_B_obj) # 添加新站点

                                path_obj_being_created_ref = self.mediator.path_being_created
                                path_count_before_finish = len(self.mediator.paths)
                                self.mediator.finish_path_creation()

                                if (path_obj_being_created_ref is not None and path_obj_being_created_ref in self.mediator.paths) or \
                                   (len(self.mediator.paths) > path_count_before_finish): # 检查是否成功
                                    action_executed_successfully = True
                                    path_extended = True
                                    reward_for_action_logic += 0.03 # 成功扩展线路的小奖励
                                    # logger.info(f"  成功扩展线路。新线路包含 S{sB_slot_idx}。现有线路: {len(self.mediator.paths)}")
                                else:
                                    logger.warning(f"  扩展线路后，Mediator未能完成新线路的创建。")
                                    # 尝试恢复被错误取消的路径 (可选)
                                    # self._try_recreate_path_safely(original_stations_in_path, original_is_loop)
                                break # 已找到并尝试扩展一个路径，不再检查其他路径
                        
                        # （可选）在这里添加逻辑：如果 sB 是某条线路的头部，尝试将 sA 添加到其前端
                        # 这会更复杂，因为需要重建整个线路 sA -> sB -> P_original_stations[1:]

                    if path_extended: # 如果已经成功扩展，则跳过后续创建新路径的逻辑
                        pass
                    # --- 2. 如果没有扩展发生，则尝试创建新线路 ---
                    else:
                        logger.debug(f"未能找到可扩展的路径，或扩展失败。尝试创建 S{sA_slot_idx} -> S{sB_slot_idx} 的新线路。")
                        if len(self.mediator.paths) < self.max_paths:
                            # 检查是否已存在 sA -> sB 的直接连接 (避免完全重复的2站点线路)
                            already_directly_connected = False
                            for p_existing in self.mediator.paths:
                                if len(p_existing.stations) == 2 and \
                                   p_existing.stations[0] == station_A_obj and \
                                   p_existing.stations[1] == station_B_obj:
                                    already_directly_connected = True; break
                            
                            if not already_directly_connected:
                                path_count_before_create = len(self.mediator.paths)
                                self.mediator.start_path_on_station(station_A_obj)
                                self.mediator.add_station_to_path(station_B_obj)
                                path_obj_being_created_ref = self.mediator.path_being_created
                                self.mediator.finish_path_creation()
                                
                                if (path_obj_being_created_ref is not None and path_obj_being_created_ref in self.mediator.paths) or \
                                   (len(self.mediator.paths) > path_count_before_create):
                                    action_executed_successfully = True
                                    reward_for_action_logic += 0.02 # 成功添加新线路的小奖励
                                    #logger.info(f"成功添加新线路 S{sA_slot_idx}-S{sB_slot_idx}。现有线路: {len(self.mediator.paths)}")
                                else:
                                    reward_for_action_logic += invalid_action_penalty 
                                    #logger.debug(f"创建新线路 S{sA_slot_idx}-S{sB_slot_idx} 失败 (Mediator未能完成)。")
                            else:
                                 reward_for_action_logic += invalid_action_penalty * 0.2 
                                 #logger.debug(f"创建新线路 S{sA_slot_idx}->S{sB_slot_idx} 失败：已存在完全相同的2站点线路。")
                        else:
                            reward_for_action_logic += invalid_action_penalty # 达到最大线路数
                            #logger.debug(f"创建新线路 S{sA_slot_idx}->S{sB_slot_idx} 失败：已达最大线路数 {self.max_paths}。")
                else:
                    reward_for_action_logic += invalid_action_penalty # 无效站点或相同站点
                    #logger.debug(f"ADD_EDGE S{sA_slot_idx}->S{sB_slot_idx} 失败：站点无效或相同。")
            else: # sA_slot_idx 或 sB_slot_idx 为 None，这是 _get_edge_from_action 内部的错误
                reward_for_action_logic += invalid_action_penalty 
                #logger.error(f"ADD_EDGE 失败：内部站点索引解析错误。sA={sA_slot_idx}, sB={sB_slot_idx}")

        elif mode == "DELETE_EDGE":
            if sA_slot_idx is not None and sB_slot_idx is not None: # mypy
                station_A_obj = self._get_station_by_slot_idx(sA_slot_idx)
                station_B_obj = self._get_station_by_slot_idx(sB_slot_idx)
                if station_A_obj and station_B_obj: # 确保站点对象存在
                    path_to_delete = None
                    # 优先删除包含 sA -> sB 片段的路径
                    for p in self.mediator.paths:
                        for i in range(len(p.stations) - 1):
                            if p.stations[i] == station_A_obj and p.stations[i+1] == station_B_obj:
                                path_to_delete = p; break
                        if path_to_delete: break
                    
                    # 如果没找到 sA -> sB，再找 sB -> sA (如果希望边是双向概念上的删除)
                    if not path_to_delete and False: # 暂时只考虑单向删除 sA->sB
                         for p in self.mediator.paths:
                            for i in range(len(p.stations) - 1):
                                if p.stations[i] == station_B_obj and p.stations[i+1] == station_A_obj:
                                    path_to_delete = p; break
                            if path_to_delete: break
                    
                    # 如果还没找到，就找第一条同时包含 sA 和 sB 的路径 (不一定相邻)
                    if not path_to_delete:
                        for p in self.mediator.paths:
                            if station_A_obj in p.stations and station_B_obj in p.stations:
                                path_to_delete = p; break
                    
                    if path_to_delete:
                        # logger.info(f"尝试删除包含 S{sA_slot_idx} 和 S{sB_slot_idx} 的线路 (路径ID: {path_to_delete.id if hasattr(path_to_delete, 'id') else 'N/A'})")
                        self.mediator.cancel_path(path_to_delete)
                        action_executed_successfully = True
                        reward_for_action_logic += 0.01 # 成功删除线路的小奖励/探索奖励
                        # logger.info(f"成功删除线路。现有线路: {len(self.mediator.paths)}")
                    else:
                        reward_for_action_logic += invalid_action_penalty # 未找到包含这两个站点的线路
                        # logger.debug(f"删除边 S{sA_slot_idx}-{sB_slot_idx} 失败：未找到相关线路。")
                else:
                    reward_for_action_logic += invalid_action_penalty # 无效站点
                    # logger.debug(f"删除边 S{sA_slot_idx}-{sB_slot_idx} 失败：站点无效。")
            else: reward_for_action_logic += invalid_action_penalty # 内部映射错误
        
        elif mode == "INVALID":
            reward_for_action_logic += invalid_action_penalty * 2 # 严重的无效动作
            logger.error(f"接收到解析后无效的动作模式！原始动作: {action}")


        # --- 游戏模拟 ---
        game_outcome_state = MeditatorState.RUNNING
        _framerate_actual = self.metadata['render_fps'] or 30
        dt_ms = 1000 // _framerate_actual
        
        for _ in range(self.game_ticks_per_rl_step):
            if self.visuals and self.screen:
                for pygame_event in pygame.event.get():
                    if pygame_event.type == pygame.QUIT: terminated = True; break
            if terminated: break
            try:
                game_outcome_state = self.mediator.increment_time(dt_ms)
            except Exception as e_sim_tick: # 捕获模拟单个tick时可能发生的错误
                logger.error(f"Mediator.increment_time 发生错误: {e_sim_tick}", exc_info=True)
                terminated = True # 认为游戏无法继续
                reward_for_action_logic -= 20.0 # 对导致游戏核心错误的步骤给予重罚
                break # 跳出游戏模拟循环

            if game_outcome_state == MeditatorState.ENDED:
                # logger.info("游戏因 MediatorState.ENDED 而终止。")
                terminated = True; break
        
        # --- 计算总奖励 ---
        reward = reward_for_action_logic
        
        score_change = self.mediator.score - self.last_score
        reward += score_change
        self.last_score = self.mediator.score

        if terminated and game_outcome_state == MeditatorState.ENDED:
            reward -= 50.0 # 加大游戏结束惩罚

        _station_max_cap_reward = STATION_MAX_CAPACITY if 'STATION_MAX_CAPACITY' in globals() and STATION_MAX_CAPACITY > 0 else 10.0
        if self.current_rl_step % 15 == 0:
            num_crowded_stations = sum(1 for s in self.mediator.stations if len(s.passengers) / _station_max_cap_reward > 0.85) # 阈值可以调整
            reward -= num_crowded_stations * 6 # 加大对每个拥挤站点的惩罚
        if self.current_rl_step % 5 == 0:
            num_total_passengers_waiting = sum(len(s.passengers) for s in self.mediator.stations)
            reward -= num_total_passengers_waiting * 0.02 # 对每个等待的乘客进行较小的持续惩罚

        self.current_rl_step += 1
        # 基于步数的 truncated 已移除

        observation = self._get_obs()
        info = self._get_info()
        logger.debug(f"步骤结束。总奖励: {reward:.3f}, 分数: {self.mediator.score}, 终止: {terminated}, RL步数: {self.current_rl_step}")
        
        if self.visuals: self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if not self.visuals and 'rgb_array' not in self.metadata['render_modes']: return None
        if self.screen is None: 
            if not pygame.display.get_init(): pygame.display.init()
            self.screen = pygame.Surface((screen_width, screen_height)) 
        self.screen.fill(screen_color)
        if 'draw_waves' in globals() and callable(draw_waves): draw_waves(self.screen, self.mediator.time_ms)
        self.mediator.render(self.screen)
        if 'human' in self.metadata['render_modes'] and self.visuals:
            pygame.display.flip()
            if self.clock: self.clock.tick(self.metadata['render_fps'])
            return None
        if 'rgb_array' in self.metadata['render_modes']:
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2) 
        return None

    def _get_info(self) -> Dict:
        return {
            "score": self.mediator.score,
            "total_game_ticks": self.mediator.steps,
            "rl_episode_steps": self.current_rl_step,
            "num_stations": len(self.mediator.stations),
            "num_completed_paths": len(self.mediator.paths),
            "passengers_waiting_total": sum(len(s.passengers) for s in self.mediator.stations),
            "mediator_internal_state": self.mediator.increment_time(0) # 获取当前状态
        }

    def close(self):
        logger.info("关闭 MiniMetroEdgeEnv 环境。")
        if self.visuals or (self.screen is not None):
            try: pygame.display.quit()
            except pygame.error as e: logger.warning(f"关闭 Pygame 显示时出错: {e}")
        self.screen = None; self.clock = None


# --- 训练与评估脚本示例 ---
def run_edge_env_training(load_model_path: Optional[str] = None):
    logger.info("启动 MiniMetroEdgeEnv 的训练...")

    # 确保必要的配置和类已定义
    try:
        _ = NUM_HORIZONTAL_GRIDS; _ = ShapeType; _ = STATION_MAX_CAPACITY; _ = Mediator
    except NameError as e:
        logger.critical(f"错误：一些必要的配置或类未定义: {e}。脚本将中止。")
        return
    if PPO is None:
        logger.critical("stable_baselines3 未安装。无法执行训练。")
        return

    log_dir = "./mini_metro_edge_rl_logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_save_name = "ppo_mini_metro_edge"

    train_env = MiniMetroEdgeEnv(visuals=False, gamespeed=20) # 训练时提高游戏速度
    
    # 检查环境 (在重大更改后运行一次是个好习惯)
    try:
        logger.info("检查环境规范性...")
        check_env(train_env)
        logger.info("环境检查通过。")
    except Exception as e_check:
        logger.error(f"环境检查失败: {e_check}", exc_info=True)
        train_env.close()
        return

    eval_env = MiniMetroEdgeEnv(visuals=False, gamespeed=1, is_eval_env=True)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=os.path.join(log_dir, 'best_model/'),
                                 log_path=os.path.join(log_dir, 'results/'),
                                 eval_freq=50000, # 根据训练速度调整
                                 deterministic=True,
                                 render=False)

    # PPO 超参数 (可能需要针对新的状态/动作空间进行调整)
    model_params = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,      # 每个 rollout 的步数
        "ent_coef": 0.05,     # 熵系数，鼓励探索
        "learning_rate": 3e-4, # 学习率
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "batch_size": 64,
        "n_epochs": 10,
        "seed": 42,
        "tensorboard_log": os.path.join(log_dir, "tensorboard/")
    }

    if load_model_path and os.path.exists(load_model_path):
        logger.info(f"从 {load_model_path} 加载预训练模型...")
        model = PPO.load(load_model_path, env=train_env, custom_objects=model_params)
        # 如果学习率等也想重置，可以 model.learning_rate = new_value
    else:
        if load_model_path: logger.warning(f"指定的模型路径 {load_model_path} 未找到。将从头开始训练。")
        model = PPO("MlpPolicy", train_env, verbose=1, **model_params)

    logger.info(f"模型策略网络结构: {model.policy}")
    logger.info(f"观测空间维度: {train_env.observation_space.shape}")
    logger.info(f"动作空间大小: {train_env.action_space.n}")

    total_timesteps_to_train = 500000 # 调整总训练步数
    logger.info(f"开始训练，总步数: {total_timesteps_to_train}...")
    try:
        model.learn(total_timesteps=total_timesteps_to_train,
                    callback=eval_callback,
                    progress_bar=True,
                    reset_num_timesteps= not (load_model_path and os.path.exists(load_model_path))) # 如果加载模型，不要重置时间步计数
        model.save(os.path.join(log_dir, model_save_name))
        logger.info(f"训练完成。最终模型已保存到 {os.path.join(log_dir, model_save_name)}.zip")
    except Exception as e_train:
        logger.error(f"训练过程中发生错误: {e_train}", exc_info=True)
    finally:
        train_env.close()
        eval_env.close()

    # --- 评估 ---
    logger.info("\n--- 开始评估训练好的模型 ---")
    try:
        eval_model_path = os.path.join(log_dir, 'best_model/best_model.zip')
        if not os.path.exists(eval_model_path):
            eval_model_path = os.path.join(log_dir, model_save_name + ".zip")

        if os.path.exists(eval_model_path) and PPO is not None:
            loaded_model = PPO.load(eval_model_path)
            logger.info(f"已加载模型进行评估: {eval_model_path}")
            eval_env_for_policy = MiniMetroEdgeEnv(visuals=False, gamespeed=1, is_eval_env=True)
            mean_reward, std_reward = evaluate_policy(loaded_model,
                                                      eval_env_for_policy,
                                                      n_eval_episodes=10, # 增加评估回合数
                                                      deterministic=True)
            logger.info(f"评估结果: 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
            eval_env_for_policy.close()
        else:
            logger.warning(f"评估模型 {eval_model_path} 未找到或PPO未导入。")
    except Exception as e_eval:
        logger.error(f"评估过程中发生错误: {e_eval}", exc_info=True)


def run_edge_env_visual_evaluation(model_path: Optional[str] = None):
    logger.info("\n--- 使用训练好的模型进行可视化演示 (EdgeEnv) ---")
    if PPO is None: logger.critical("PPO未导入，无法进行可视化。"); return

    if model_path is None:
        log_dir = "./mini_metro_edge_rl_logs/"
        default_model_path = os.path.join(log_dir, 'best_model/best_model.zip')
        if not os.path.exists(default_model_path):
            default_model_path = os.path.join(log_dir, "ppo_mini_metro_edge.zip")
        model_path = default_model_path
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件 {model_path} 未找到。无法进行可视化。")
        return

    try:
        loaded_model = PPO.load(model_path)
        logger.info(f"已加载模型进行可视化: {model_path}")

        env_visual = MiniMetroEdgeEnv(visuals=True, gamespeed=1, is_eval_env=True)
        
        for episode in range(1): # 演示3个episodes
            obs_visual, _ = env_visual.reset()
            terminated_visual, truncated_visual = False, False
            episode_reward = 0.0
            logger.info(f"\n--- 可视化演示 Episode {episode+1} ---")
            while True: # 限制演示步数
                action_visual, _ = loaded_model.predict(obs_visual, deterministic=False)
                mode, sA, sB = env_visual._get_edge_from_action(action_visual) # type: ignore
                # logger.info(f" Action: {action_visual} ({mode}, S{sA}-S{sB}), Score: {env_visual.mediator.score:.0f}, Reward_this_step: ...")
                
                obs_visual, reward_visual, terminated_visual, truncated_visual, info_visual = env_visual.step(action_visual)
                episode_reward += reward_visual
                
                if terminated_visual or truncated_visual:
                    logger.info(f"演示 Episode {episode+1} 结束。最终得分: {info_visual['score']:.0f}, RL步数: {info_visual['rl_episode_steps']}, 总奖励: {episode_reward:.2f}")
                    break
            if not (terminated_visual or truncated_visual):
                logger.info(f"演示 Episode {episode+1} 达到最大演示步数。当前得分: {info_visual['score']:.0f}")
        env_visual.close()

    except Exception as e_viz:
        logger.error(f"可视化演示过程中发生错误: {e_viz}", exc_info=True)
        if 'env_visual' in locals(): env_visual.close() # type: ignore


if __name__ == '__main__':
    # 训练新模型或继续训练
    # run_edge_env_training() 
    # run_edge_env_training(load_model_path=".\mini_metro_edge_rl_logs\best_model\best_model.zip") # 示例：继续训练
    
    # 仅进行可视化评估
    # run_edge_env_visual_evaluation()
    run_edge_env_visual_evaluation(model_path=".\mini_metro_edge_rl_logs\\best_model\\best_model.zip")