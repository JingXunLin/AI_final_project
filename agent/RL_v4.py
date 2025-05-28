# mini_metro_hybrid_env.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
import logging
import math # 用于计算距离

# --- Stable Baselines3 ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    logger.warning("stable_baselines3 未安装或无法导入。训练和评估脚本部分将无法运行。")
    PPO = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from config import (
        num_metros as CONFIG_TOTAL_METROS,
        num_paths as CONFIG_MAX_PATHS_TOTAL, # 总路径数
        num_stations_max,
        framerate, screen_color, screen_width, screen_height,
        station_grid_size, station_capacity,
    )
    NUM_HORIZONTAL_GRIDS, NUM_VERTICAL_GRIDS = station_grid_size
    STATION_MAX_CAPACITY = station_capacity
    from geometry.type import ShapeType
    from mediator import Mediator, MeditatorState, Path as GamePath # 明确导入Path
    from visuals.background import draw_waves
    from entity.station import Station
    from entity.metro import Metro
except ImportError as e:
    logger.critical(f"Import Error: {e}.")
    raise

try:
    from stable_baselines3 import PPO
    # ... (其他 SB3 导入)
except ImportError:
    logger.warning("stable_baselines3 not found.")
    PPO = None

MAX_METROS_PER_PATH_RL = 2 # RL控制的路径最多可以有多少列车 (如果RL有分配列车的动作)
# 固定环线将只使用1个列车，由系统自动分配

class MiniMetroHybridEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': framerate if 'framerate' in globals() and framerate else 30}

    def __init__(self, gamespeed: int = 1, visuals: bool = False, is_eval_env: bool = False):
        super().__init__()
        logger.info("Initializing MiniMetroHybridEnv (Fixed Grand Tour + RL Aux Paths)...")

        if not pygame.font.get_init(): pygame.font.init()
        if visuals and not pygame.get_init(): pygame.init()

        self.mediator = Mediator(gamespeed=gamespeed, gen_stations_first=False)

        self.max_station_slots = num_stations_max
        self.total_game_paths = CONFIG_MAX_PATHS_TOTAL # 游戏允许的总线路数
        if self.total_game_paths < 1:
            logger.warning("CONFIG_MAX_PATHS_TOTAL is less than 1. Setting to 1.")
            self.total_game_paths = 1
        
        # 一条路径用于固定策略的“全站点环线”
        self.num_rl_managed_paths = self.total_game_paths - 1
        if self.num_rl_managed_paths < 0: # 确保至少有一条总路径给固定策略
             logger.warning(f"Total game paths ({self.total_game_paths}) is too low for a dedicated fixed path and RL paths. RL will have 0 paths to manage.")
             self.num_rl_managed_paths = 0


        self.total_initial_metros = CONFIG_TOTAL_METROS

        try: self.shape_types_enum: List[ShapeType] = list(ShapeType)
        except TypeError: logger.warning("ShapeType non-iterable."); self.shape_types_enum = []
        self.num_shape_types = len(self.shape_types_enum)
        if self.num_shape_types == 0: logger.critical("num_shape_types is 0!")

        self.visuals = visuals; self.is_eval_env = is_eval_env
        self.screen: Optional[pygame.Surface] = None; self.clock: Optional[pygame.time.Clock] = None
        if self.visuals: # Pygame display init ...
            if not pygame.display.get_init(): pygame.display.init()
            try:
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Mini Metro RL (Hybrid Env)")
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                logger.warning(f"Pygame display init error: {e}. Visuals off."); self.visuals = False

        # --- RL 智能体的动作空间 ---
        # 0: NO_OP
        # 1 to N_ORDERED_PAIRS: ADD_EDGE(sA, sB) for RL paths
        # N_ORDERED_PAIRS+1 to N_ORDERED_PAIRS + num_rl_managed_paths: DELETE_RL_PATH(rl_path_idx)
        
        self.num_station_ordered_pairs = self.max_station_slots * (self.max_station_slots - 1)
        
        self.action_idx_no_op = 0
        self.action_idx_add_edge_start = 1
        self.action_idx_add_edge_end = self.action_idx_add_edge_start + self.num_station_ordered_pairs -1
        
        self.action_idx_delete_rl_path_start = self.action_idx_add_edge_end + 1
        # 删除动作的数量取决于 num_rl_managed_paths，但为了action space固定，我们用 total_game_paths -1
        # 如果 num_rl_managed_paths 为0，则不应该有删除动作，或者这些动作无效
        # 为了gym.space的固定性，我们用CONFIG_MAX_PATHS_TOTAL - 1作为删除动作的数量上限
        # 实际有效的删除动作会在step中根据当前RL管理的路径数来判断
        num_potential_rl_paths_to_delete = max(0, self.total_game_paths - 1)

        self.action_idx_delete_rl_path_end = self.action_idx_delete_rl_path_start + num_potential_rl_paths_to_delete -1
        
        if num_potential_rl_paths_to_delete > 0:
            self.action_space = spaces.Discrete(self.action_idx_delete_rl_path_end + 1)
        else: # 如果没有RL路径可管理，则只有NO_OP和ADD_EDGE（尽管ADD_EDGE会一直失败）
            self.action_space = spaces.Discrete(self.action_idx_add_edge_end + 1)

        self._sA_sB_pair_map: Dict[int, Tuple[int, int]] = {} 
        _pair_idx_counter = 0
        for sA in range(self.max_station_slots):
            for sB in range(self.max_station_slots):
                if sA == sB: continue
                self._sA_sB_pair_map[_pair_idx_counter] = (sA, sB); _pair_idx_counter += 1
        
        logger.debug(f"RL Action space size: {self.action_space.n}")
        logger.debug(f"  ADD_EDGE actions: {self.action_idx_add_edge_start} - {self.action_idx_add_edge_end}")
        if num_potential_rl_paths_to_delete > 0:
            logger.debug(f"  DELETE_RL_PATH actions: {self.action_idx_delete_rl_path_start} - {self.action_idx_delete_rl_path_end}")
        else:
            logger.debug(f"  No DELETE_RL_PATH actions as num_rl_managed_paths is 0.")


        # --- 状态空间 (观测空间) ---
        # 1. 站点特征 (与之前 EdgeEnv 类似)
        self.station_feature_size = 1 + self.num_shape_types + 1 + 1 + self.num_shape_types 
        obs_size_stations = self.max_station_slots * self.station_feature_size
        # 2. 邻接矩阵 (所有路径贡献)
        obs_size_adj_matrix = self.max_station_slots * self.max_station_slots
        # 3. 路径特征 (为游戏中的每条可能路径 self.total_game_paths)
        #    exists(1), is_loop(1), num_stations(1), load(1), station_sequence(max_slots), 
        #    num_metros(1), max_station_pressure(1), is_fixed_tour_path(1) <--- NEW
        self.path_feature_size = 1 + 1 + 1 + 1 + self.max_station_slots + 1 + 1 + 1
        obs_size_paths = self.total_game_paths * self.path_feature_size
        # 4. 全局特征
        obs_size_global = 5 # num_paths, avg_path_len, score, total_wait, metros_in_pool
        
        self.total_obs_size = obs_size_stations + obs_size_adj_matrix + obs_size_paths + obs_size_global
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_obs_size,), dtype=np.float32)
        #logger.info(f"Observation Space Shape: ({self.total_obs_size},)")

        self.last_score = 0; self.current_rl_step = 0
        self.game_ticks_per_rl_step = 20 
        
        self.fixed_tour_path_object: Optional[GamePath] = None
        self.last_num_stations_for_tour_update = 0 # 用于检测新站点

    def _get_station_by_slot_idx(self, slot_idx: int) -> Optional[Station]: # सेम
        if 0 <= slot_idx < len(self.mediator.stations): return self.mediator.stations[slot_idx] # type: ignore
        return None

    def _get_path_by_game_idx(self, game_path_idx: int) -> Optional[GamePath]:
        """获取mediator.paths中指定索引的路径 (如果存在)"""
        if 0 <= game_path_idx < len(self.mediator.paths): return self.mediator.paths[game_path_idx]
        return None

    def _update_fixed_tour_path(self):
        """
        固定策略：维护一条连接所有当前活动站点的环线。
        这条线路将使用 self.mediator.paths 中的一个槽位。
        如果站点列表发生变化 (新站点出现)，则重新构建此环线。
        """
        current_stations_list = [s for s in self.mediator.stations if s] # 获取当前所有有效站点对象
        if not current_stations_list: # 没有站点，则移除固定环线（如果存在）
            if self.fixed_tour_path_object and self.fixed_tour_path_object in self.mediator.paths:
                #logger.info("固定环线：没有站点了，移除现有环线。")
                self.mediator.cancel_path(self.fixed_tour_path_object)
            self.fixed_tour_path_object = None
            self.last_num_stations_for_tour_update = 0
            return

        # 检查站点数量是否变化，或者固定环线是否不存在/被意外删除
        needs_update = (len(current_stations_list) != self.last_num_stations_for_tour_update) or \
                       (self.fixed_tour_path_object is None) or \
                       (self.fixed_tour_path_object not in self.mediator.paths)

        if not needs_update:
            return # 站点未变且固定环线存在，无需更新

        # logger.info(f"固定环线：需要更新。当前站点数: {len(current_stations_list)}, 上次: {self.last_num_stations_for_tour_update}")

        # 如果已存在固定环线，先取消它
        if self.fixed_tour_path_object and self.fixed_tour_path_object in self.mediator.paths:
            logger.debug("固定环线：取消现有环线以便重建。")
            self.mediator.cancel_path(self.fixed_tour_path_object)
        self.fixed_tour_path_object = None # 清除旧引用

        if len(self.mediator.paths) >= self.total_game_paths and self.num_rl_managed_paths < self.total_game_paths -1 :
             # 如果总路径已满，并且固定环线还未占用它的名额，可能需要为固定环线腾出空间
             # （这种冲突理论上不应发生，因为RL控制的路径数是 total-1）
             # 但作为保险，如果RL路径把所有槽都占了，固定环线可能无法创建。
             # 更简单的做法是，固定环线优先使用一个槽。
             logger.warning("固定环线：游戏总路径数已满，可能无法创建/重建固定环线，除非RL路径被删除。")
             # 此处不主动删除RL路径，等待RL体删除或游戏自然腾出空间

        if len(current_stations_list) < 2: # 站点太少，无法形成有效路径
            logger.debug("固定环线：站点少于2个，无法形成环线。")
            self.last_num_stations_for_tour_update = len(current_stations_list)
            return
        
        # 确保固定环线有槽位可用 (它占用 self.total_game_paths 中的一个)
        num_rl_paths_currently = len([p for p in self.mediator.paths if p != self.fixed_tour_path_object]) # 重新计算以防万一
        if num_rl_paths_currently > self.num_rl_managed_paths:
            logger.warning(f"固定环线：当前RL路径数 ({num_rl_paths_currently}) 超出允许 ({self.num_rl_managed_paths})，可能无法创建固定环线。")
            # 这通常意味着总路径数已满，且固定环线还未创建/恢复
            # 如果固定环线是必须的，这里可以考虑强制删除一条RL线路，但目前不这样做，让RL体自己管理。

        if len(self.mediator.paths) < self.total_game_paths : # 只有在总路径数未满时才能创建新路径
            # 简单排序策略：按站点在 mediator.stations 列表中的当前顺序（大致是出现顺序）
            # 或者可以按坐标排序以获得更稳定的几何顺序
            # sorted_stations = sorted(current_stations_list, key=lambda s: (s.grid_x, s.grid_y))
            sorted_stations = current_stations_list # 使用当前列表顺序

            logger.debug(f"固定环线：开始在 {len(sorted_stations)} 个站点上构建新环线。")
            self.mediator.start_path_on_station(sorted_stations[0])
            for i in range(1, len(sorted_stations)):
                self.mediator.add_station_to_path(sorted_stations[i])
            
            # 形成环路
            if len(sorted_stations) > 1: # add_station_to_path 会自动处理
                self.mediator.add_station_to_path(sorted_stations[0]) 
            
            path_obj_ref = self.mediator.path_being_created
            self.mediator.finish_path_creation()

            if path_obj_ref and path_obj_ref in self.mediator.paths:
                self.fixed_tour_path_object = path_obj_ref
                # logger.info(f"固定环线：已成功创建/更新。ID: {path_obj_ref.id if hasattr(path_obj_ref,'id') else 'N/A'}, 站点数: {len(path_obj_ref.stations)}")
                # 确保固定环线有列车 (Mediator的finish_path_creation会自动分配一个，如果池中有)
                if not path_obj_ref.metros and self.mediator.metros:
                    logger.debug("固定环线：尝试为新环线分配列车。")
                    path_obj_ref.add_metro(self.mediator.metros.pop(0))
            else:
                logger.error("固定环线：尝试创建/更新，但未能最终形成线路。")
                self.fixed_tour_path_object = None # 确保状态一致
        else:
            logger.warning("固定环线：无法创建/更新，因为游戏总路径数已达到上限。")

        self.last_num_stations_for_tour_update = len(current_stations_list)


    def _get_obs(self) -> np.ndarray: # 基本与之前EdgeEnv版本类似，增加 is_fixed_tour_path
        obs = np.zeros(self.total_obs_size, dtype=np.float32)
        obs_idx_ptr = 0
        _smc = STATION_MAX_CAPACITY if STATION_MAX_CAPACITY > 0 else 1.0
        s_timeout_pressures = {}

        # --- 1. 站点特征 ---
        for s_slot_idx in range(self.max_station_slots):
            s_obj = self._get_station_by_slot_idx(s_slot_idx)
            obs[obs_idx_ptr] = 1.0 if s_obj else 0.0; obs_idx_ptr += 1
            if s_obj:
                s_one_hot=np.zeros(self.num_shape_types,dtype=np.float32);
                if self.num_shape_types>0:
                    try:idx=self.shape_types_enum.index(s_obj.shape.type);s_one_hot[idx]=1.0
                    except ValueError:pass
                obs[obs_idx_ptr:obs_idx_ptr+self.num_shape_types]=s_one_hot; obs_idx_ptr+=self.num_shape_types
                obs[obs_idx_ptr]=len(s_obj.passengers)/_smc; obs_idx_ptr+=1 # total_pass_count_norm (这里原为4，改为1)
                
                acc_p=getattr(s_obj,'time_inactive_penalty_acc_ms',0.0);thr_p=getattr(s_obj,'time_inactive_penalty_threshold_ms',1.0)
                pressure=acc_p/(thr_p if thr_p>0 else 1.0)
                obs[obs_idx_ptr]=pressure; obs_idx_ptr+=1; s_timeout_pressures[s_slot_idx]=pressure
                
                s_dest_prof=np.zeros(self.num_shape_types,dtype=np.float32)
                if len(s_obj.passengers)>0 and self.num_shape_types>0:
                    for p in s_obj.passengers:
                        try:idx=self.shape_types_enum.index(p.destination_shape.type);s_dest_prof[idx]+=1.0
                        except ValueError:pass
                    s_dest_prof/=(len(s_obj.passengers)+1e-6)
                obs[obs_idx_ptr:obs_idx_ptr+self.num_shape_types]=s_dest_prof; obs_idx_ptr+=self.num_shape_types
            else: obs_idx_ptr+=(self.station_feature_size-1)

        # --- 2. 邻接矩阵 ---
        adj_matrix = np.zeros((self.max_station_slots,self.max_station_slots),dtype=np.float32)
        s_obj_to_slot_map:Dict[int,int]={id(self.mediator.stations[i]):i for i in range(len(self.mediator.stations)) if i<self.max_station_slots}
        for p_obj in self.mediator.paths:
            for i in range(len(p_obj.stations)-1):
                sA_slot=s_obj_to_slot_map.get(id(p_obj.stations[i]),-1);sB_slot=s_obj_to_slot_map.get(id(p_obj.stations[i+1]),-1)
                if sA_slot!=-1 and sB_slot!=-1: adj_matrix[sA_slot,sB_slot]=1.0 # Binary connection
            if p_obj.is_looped and len(p_obj.stations)>1:
                sL_slot=s_obj_to_slot_map.get(id(p_obj.stations[-1]),-1);sF_slot=s_obj_to_slot_map.get(id(p_obj.stations[0]),-1)
                if sL_slot!=-1 and sF_slot!=-1: adj_matrix[sL_slot,sF_slot]=1.0
        obs[obs_idx_ptr:obs_idx_ptr+adj_matrix.size]=adj_matrix.flatten(); obs_idx_ptr+=adj_matrix.size

        # --- 3. 路径特征 ---
        # (为游戏中的每条可能路径 self.total_game_paths)
        for p_game_slot_idx in range(self.total_game_paths): # Iterate up to total game paths capacity
            p_obj = self._get_path_by_game_idx(p_game_slot_idx) # Tries to get path from mediator.paths
            obs[obs_idx_ptr] = 1.0 if p_obj else 0.0; obs_idx_ptr += 1 # path_exists
            if p_obj:
                obs[obs_idx_ptr]=1.0 if p_obj.is_looped else 0.0; obs_idx_ptr+=1
                obs[obs_idx_ptr]=len(p_obj.stations)/(self.max_station_slots+1e-6); obs_idx_ptr+=1
                crowd_s=sum(len(s.passengers)/_smc for s in p_obj.stations) if p_obj.stations else 0.0
                obs[obs_idx_ptr]=crowd_s/(len(p_obj.stations)+1e-6) if p_obj.stations else 0.0; obs_idx_ptr+=1
                
                p_seq_indices=np.full(self.max_station_slots,-1.0,dtype=np.float32)
                max_p_pressure=0.0
                for i,s_on_p in enumerate(p_obj.stations):
                    if i<self.max_station_slots:
                        s_obs_slot=s_obj_to_slot_map.get(id(s_on_p),-1)
                        if s_obs_slot!=-1:
                            norm_s_idx=float(s_obs_slot)/(self.max_station_slots-1) if self.max_station_slots>1 else 0.0
                            p_seq_indices[i]=norm_s_idx
                            max_p_pressure=max(max_p_pressure,s_timeout_pressures.get(s_obs_slot,0.0))
                    else: break
                obs[obs_idx_ptr:obs_idx_ptr+self.max_station_slots]=p_seq_indices; obs_idx_ptr+=self.max_station_slots
                obs[obs_idx_ptr]=len(p_obj.metros)/(CONFIG_TOTAL_METROS+1e-6); obs_idx_ptr+=1 # Norm by total metros as rough guide
                obs[obs_idx_ptr]=max_p_pressure; obs_idx_ptr+=1
                obs[obs_idx_ptr]=1.0 if p_obj == self.fixed_tour_path_object else 0.0; obs_idx_ptr+=1 # is_fixed_tour_path
            else: obs_idx_ptr+=(self.path_feature_size-1)

        # --- 4. 全局特征 ---
        obs[obs_idx_ptr]=len(self.mediator.paths)/(self.total_game_paths+1e-6); obs_idx_ptr+=1
        avg_l=sum(len(p.stations) for p in self.mediator.paths)/(len(self.mediator.paths)+1e-6) if self.mediator.paths else 0
        obs[obs_idx_ptr]=avg_l/(self.max_station_slots+1e-6); obs_idx_ptr+=1
        obs[obs_idx_ptr]=self.mediator.score/(self.mediator.steps+1e-5) if self.mediator.steps>0 else 0.0; obs_idx_ptr+=1
        tot_w=sum(len(s.passengers) for s in self.mediator.stations)
        max_sys_p=self.max_station_slots*_smc+1e-6
        obs[obs_idx_ptr]=tot_w/max_sys_p if max_sys_p>0 else 0.0; obs_idx_ptr+=1
        obs[obs_idx_ptr]=len(self.mediator.metros)/(self.total_initial_metros+1e-6); obs_idx_ptr+=1
        
        if obs_idx_ptr != self.total_obs_size: logger.critical(f"OBS MISMATCH! Exp {self.total_obs_size}, Got {obs_idx_ptr}.")
        return np.clip(obs, -1.0, 1.0)

    def _interpret_rl_action(self, action_flat: int) -> Tuple[str, Optional[int], Optional[int]]:
        if action_flat == self.action_idx_no_op: return "NO_OP", None, None
        
        if self.action_idx_add_edge_start <= action_flat <= self.action_idx_add_edge_end:
            pair_flat_idx = action_flat - self.action_idx_add_edge_start
            sA, sB = self._sA_sB_pair_map[pair_flat_idx]
            return "ADD_EDGE", sA, sB
        
        # Check if DELETE_RL_PATH actions are even possible based on num_rl_managed_paths
        if self.num_rl_managed_paths > 0 and \
           self.action_idx_delete_rl_path_start <= action_flat <= self.action_idx_delete_rl_path_end:
            # The action_flat gives an index relative to the max possible RL paths.
            # We need to map this to an actual RL-controlled path.
            rl_path_target_idx = action_flat - self.action_idx_delete_rl_path_start
            return "DELETE_RL_PATH", rl_path_target_idx, None # arg1 is the target index for RL paths
            
        logger.warning(f"Unknown or out-of-bounds flat RL action index: {action_flat}")
        return "INVALID_ACTION", None, None

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
        # 1. 更新固定策略的环线 (在RL动作之前，这样RL可以看到最新的固定环线状态)
        self._update_fixed_tour_path()

        logger.debug(f"Step {self.current_rl_step}, Raw RL Action: {action}")
        reward_action_logic = 0.0
        terminated = False; truncated = False
        invalid_action_penalty = -0.1

        action_type, arg1, arg2 = self._interpret_rl_action(action)
        logger.debug(f"Interpreted RL Action: {action_type}, Arg1: {arg1}, Arg2: {arg2}")
        
        # RL 控制的路径列表 (不包括固定环线)
        rl_controlled_paths = [p for p in self.mediator.paths if p != self.fixed_tour_path_object]

        if action_type == "NO_OP" or self.current_rl_step % 10 != 0: 
            pass # No direct reward/penalty for NO_OP
        
        elif action_type == "ADD_EDGE":
            sA_idx, sB_idx = arg1, arg2
            sA_obj = self._get_station_by_slot_idx(sA_idx) #type:ignore
            sB_obj = self._get_station_by_slot_idx(sB_idx) #type:ignore

            if sA_obj and sB_obj and sA_obj != sB_obj:
                path_to_extend_at_A_tail = None
                # 尝试扩展RL控制的路径
                for p_rl in rl_controlled_paths: # 只迭代RL控制的路径
                    if p_rl.stations and p_rl.stations[-1] == sA_obj and sB_obj not in p_rl.stations:
                        path_to_extend_at_A_tail = p_rl; break
                
                if path_to_extend_at_A_tail:
                    # ... (扩展逻辑与之前类似，但确保操作的是 rl_controlled_paths 中的对象)
                    # ... (并确保重建后它仍然被认为是RL路径，而不是意外变成fixed_tour_path_object)
                    original_stations = list(path_to_extend_at_A_tail.stations)
                    self.mediator.cancel_path(path_to_extend_at_A_tail)
                    
                    current_valid_stations_rebuild = [s for s_orig in original_stations if (s:=self._get_station_by_slot_idx(self.mediator.stations.index(s_orig)) if s_orig in self.mediator.stations else None)]
                    if len(current_valid_stations_rebuild) == len(original_stations):
                        self.mediator.start_path_on_station(current_valid_stations_rebuild[0])
                        for s_inter in current_valid_stations_rebuild[1:]: self.mediator.add_station_to_path(s_inter)
                        self.mediator.add_station_to_path(sB_obj)
                        
                        path_obj_ref = self.mediator.path_being_created; pc_before = len(self.mediator.paths)
                        self.mediator.finish_path_creation()
                        if (path_obj_ref and path_obj_ref in self.mediator.paths and path_obj_ref != self.fixed_tour_path_object) or \
                           (len(self.mediator.paths) > pc_before and self.mediator.paths[-1] != self.fixed_tour_path_object) : # 确保新路径不是固定环线
                            reward_action_logic += 0.03; #logger.info(f"  RL: Extended path to S{sB_idx}.")
                        else: reward_action_logic += invalid_action_penalty; logger.warning("  RL: Failed to finalize extended path or it conflicted."); self._try_recreate_path_safely(original_stations)
                    else: reward_action_logic += invalid_action_penalty; logger.warning("  RL: Stations for extension became invalid.")
                
                else: # 尝试创建新的RL控制的双站点路径
                    num_current_rl_paths = len(rl_controlled_paths)
                    if num_current_rl_paths < self.num_rl_managed_paths: # 检查RL是否有可用槽位
                        is_direct_link_on_any_path = any(len(p.stations)==2 and ((p.stations[0]==sA_obj and p.stations[1]==sB_obj)) for p in self.mediator.paths)
                        if not is_direct_link_on_any_path:
                            pc_before = len(self.mediator.paths)
                            self.mediator.start_path_on_station(sA_obj); self.mediator.add_station_to_path(sB_obj)
                            path_obj_ref = self.mediator.path_being_created; self.mediator.finish_path_creation()
                            if (path_obj_ref and path_obj_ref in self.mediator.paths and path_obj_ref != self.fixed_tour_path_object) or \
                               (len(self.mediator.paths) > pc_before and self.mediator.paths[-1] != self.fixed_tour_path_object):
                                reward_action_logic += 0.02; #logger.info(f"  RL: Created new 2-station path S{sA_idx}-S{sB_idx}.")
                            else: reward_action_logic += invalid_action_penalty; logger.debug("  RL: Failed to finalize new 2-station path or conflict.")
                        else: reward_action_logic += invalid_action_penalty*0.5; logger.debug("  RL: Direct link already exists.")
                    else: reward_action_logic += invalid_action_penalty; logger.debug("  RL: Max RL paths reached.")
            else: reward_action_logic += invalid_action_penalty; logger.debug("  RL ADD_EDGE: Invalid stations.")

        elif action_type == "DELETE_RL_PATH":
            rl_path_idx_to_delete = arg1 # 这是相对于 rl_controlled_paths 列表的索引
            if self.num_rl_managed_paths > 0 and rl_path_idx_to_delete is not None and 0 <= rl_path_idx_to_delete < len(rl_controlled_paths):
                path_to_delete = rl_controlled_paths[rl_path_idx_to_delete]
                #logger.info(f"  RL: Attempting to delete RL path (internal index {rl_path_idx_to_delete}, obj ID {path_to_delete.id if hasattr(path_to_delete, 'id') else 'N/A'}).")
                self.mediator.cancel_path(path_to_delete)
                reward_action_logic += 0.01 # 小奖励鼓励有效删除
            else:
                reward_action_logic += invalid_action_penalty
                logger.debug(f"  RL DELETE_RL_PATH: Invalid rl_path_idx {rl_path_idx_to_delete} or no RL paths to delete.")
        
        elif action_type == "INVALID_ACTION":
            reward_action_logic += invalid_action_penalty * 2
            logger.error(f"  RL: Invalid action type processed. Original flat action: {action}")

        # --- Game Simulation & Main Reward ---
        # ... (与之前 EdgeEnv 版本相同) ...
        game_outcome_state = MeditatorState.RUNNING
        _framerate_actual = self.metadata['render_fps'] or 30
        dt_ms = 1000 // _framerate_actual
        for _ in range(self.game_ticks_per_rl_step):
            if self.visuals and self.screen:
                for pygame_event in pygame.event.get():
                    if pygame_event.type == pygame.QUIT: terminated = True; break
            if terminated: break
            try: game_outcome_state = self.mediator.increment_time(dt_ms)
            except Exception as e_sim_tick: 
                logger.error(f"Mediator.increment_time error: {e_sim_tick}", exc_info=True)
                terminated = True; reward_action_logic -= 20.0; break 
            if game_outcome_state == MeditatorState.ENDED: terminated = True; break
        
        reward = reward_action_logic
        score_change = self.mediator.score - self.last_score
        reward += score_change; self.last_score = self.mediator.score

        if terminated and game_outcome_state == MeditatorState.ENDED: reward -= 50.0 # Increased penalty
        _smc_reward = STATION_MAX_CAPACITY if STATION_MAX_CAPACITY > 0 else 10.0
        num_crowded = sum(1 for s in self.mediator.stations if len(s.passengers) / _smc_reward > 0.85)
        reward -= num_crowded * 2.0 # Increased penalty for crowded stations
        
        num_total_wait = sum(len(s.passengers) for s in self.mediator.stations)
        # 更强的等待惩罚，但可能与拥挤惩罚部分重叠
        # reward -= (num_total_wait ** 1.1) * 0.005 # 非线性惩罚
        reward -= num_total_wait * 0.015


        self.current_rl_step += 1
        observation = self._get_obs(); info = self._get_info()
        logger.debug(f"Step done. R={reward:.3f}, S={self.mediator.score}, Term={terminated}, RLSteps={self.current_rl_step}")
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
        logger.info("关闭 MiniMetroHybridEnv 环境。")
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

    train_env = MiniMetroHybridEnv(visuals=False, gamespeed=20) # 训练时提高游戏速度
    
    # 检查环境 (在重大更改后运行一次是个好习惯)
    try:
        logger.info("检查环境规范性...")
        check_env(train_env)
        logger.info("环境检查通过。")
    except Exception as e_check:
        logger.error(f"环境检查失败: {e_check}", exc_info=True)
        train_env.close()
        return

    eval_env = MiniMetroHybridEnv(visuals=False, gamespeed=1, is_eval_env=True)
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
        "n_steps": 4096,      # 每个 rollout 的步数
        "ent_coef": 0.08,     # 熵系数，鼓励探索
        "learning_rate": 3e-4, # 学习率
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "batch_size": 128,
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

    total_timesteps_to_train = 300000 # 调整总训练步数
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
            eval_env_for_policy = MiniMetroHybridEnv(visuals=False, gamespeed=1, is_eval_env=True)
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

        env_visual = MiniMetroHybridEnv(visuals=True, gamespeed=1, is_eval_env=True)
        
        for episode in range(1): # 演示3个episodes
            obs_visual, _ = env_visual.reset()
            terminated_visual, truncated_visual = False, False
            episode_reward = 0.0
            logger.info(f"\n--- 可视化演示 Episode {episode+1} ---")
            while True: # 限制演示步数
                action_visual, _ = loaded_model.predict(obs_visual, deterministic=False)
                #action_type, arg1, arg2 = self._interpret_rl_action(action)
                mode, sA, sB = env_visual._interpret_rl_action(action_visual) # type: ignore
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
    run_edge_env_visual_evaluation(model_path=".\mini_metro_edge_rl_logs\\ppo_mini_metro_edge.zip")