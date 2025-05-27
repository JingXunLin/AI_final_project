# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
<<<<<<< HEAD
from typing import List, Dict, Tuple, Optional, Any # Added Any
from enum import Enum, auto
=======
from typing import List, Dict, Tuple, Optional # Added for type hinting
from stable_baselines3.common.vec_env import SubprocVecEnv
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20

# 游戏特定导入 (根据你的项目结构调整路径)
from config import (
    num_metros, num_paths, num_stations_max, station_spawning_interval_step,
    framerate, screen_color, screen_width, screen_height,
    station_grid_size, station_capacity, 
)
# 以下常量至关重要，假设在 config.py 中定义:
NUM_HORIZONTAL_GRIDS, NUM_VERTICAL_GRIDS = station_grid_size
<<<<<<< HEAD
STATION_MAX_CAPACITY = station_capacity # 或从 Station/Metro 类获取 (如果是静态成员)
=======
STATION_MAX_CAPACITY= station_capacity# 或从 Station/Metro 类获取 (如果是静态成员)
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
from geometry.type import ShapeType # 假设 ShapeType 是一个枚举
from mediator import Mediator, MeditatorState # 游戏逻辑核心
from visuals.background import draw_waves # 如果环境直接用于渲染背景波浪

# Stable Baselines3 用于训练和评估
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback #, StopTrainingOnRewardThreshold

<<<<<<< HEAD
class HighLevelAction(Enum):
    NO_OP = auto()
    BUILD_NEW_PATH_SMART = auto()
    EXTEND_EXISTING_PATH_SMART = auto()
    DELETE_LEAST_USEFUL_PATH = auto()
=======
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20

class MiniMetroSimpleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': framerate or 30}

    def __init__(self, gamespeed: int = 1, visuals: bool = False, is_eval_env: bool = False):
        super().__init__()

<<<<<<< HEAD
        if not pygame.font.get_init():
            pygame.font.init()
=======
        # 确保 Pygame 字体已初始化
        if not pygame.font.get_init():
            pygame.font.init()
        # 如果需要视觉效果，确保 Pygame 已初始化
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
        if visuals and not pygame.get_init():
             pygame.init()

        self.mediator = Mediator(gamespeed=gamespeed, gen_stations_first=False)
        
<<<<<<< HEAD
        self.max_station_slots = num_stations_max # Max observable station slots
        self.max_paths = num_paths # Max allowed paths in game, also used as max observable paths
        self.shape_types_enum: List[ShapeType] = list(ShapeType)
        self.num_shape_types = len(self.shape_types_enum)
=======
        self.max_station_slots = num_stations_max
        self.max_paths = num_paths 
        self.shape_types_enum: List[ShapeType] = list(ShapeType) # ShapeType 枚举的成员列表
        self.num_shape_types = len(self.shape_types_enum)
        self.prev_num_stations = 0  # 用來追蹤上一個 step 的站點數
        self.new_station_flag = False  # 是否出現新站點
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20

        self.visuals = visuals
        self.is_eval_env = is_eval_env
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        if self.visuals:
<<<<<<< HEAD
            if not pygame.display.get_init():
                 pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Mini Metro RL (Hierarchical Env)")
            self.clock = pygame.time.Clock()

        # --- Action Space (High-Level) ---
        self.high_level_actions_enum_list = list(HighLevelAction)
        self.action_space = spaces.Discrete(len(self.high_level_actions_enum_list))
        self.high_level_action_counts = [0] * len(self.high_level_actions_enum_list)

        # --- State Space (Observation) ---
        # 1. Station Features (Simplified)
        # exists(1), shape_norm(1), pass_count_norm(1), overcrowded_flag(1), connections_norm(1), dest_profile_norm(num_shape_types)
        self.station_feature_size_simple = 1 + 1 + 1 + 1 + 1 + self.num_shape_types
        obs_size_stations_simple = self.max_station_slots * self.station_feature_size_simple

        # 2. Path Features
        # path_exists(1), length_norm(1), is_loop(1), crowding_heuristic_norm(1), shapes_covered_multi_hot(num_shape_types)
        self.path_feature_size = 1 + 1 + 1 + 1 + self.num_shape_types
        obs_size_paths = self.max_paths * self.path_feature_size # Assuming max_observable_paths = self.max_paths
        
        # 3. Global Features
        # num_existing_paths_norm(1), score_norm(1), total_pass_waiting_norm(1), frac_stations_overcrowded_norm(1)
        obs_size_global_simple = 4
        
        self.total_obs_size = obs_size_stations_simple + obs_size_paths + obs_size_global_simple
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_obs_size,), dtype=np.float32)

        self.last_score = 0
        self.current_rl_step = 0
        self.game_ticks_per_rl_step = 15
        self.max_rl_steps_per_episode = 20000

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.total_obs_size, dtype=np.float32)
        obs_idx_ptr = 0

        # --- 1. Simplified Station Features ---
        for s_slot_idx in range(self.max_station_slots):
            station_exists_in_slot = (s_slot_idx < len(self.mediator.stations))
            
            obs[obs_idx_ptr] = 1.0 if station_exists_in_slot else 0.0 # Exists
            obs_idx_ptr += 1

            if station_exists_in_slot:
                station = self.mediator.stations[s_slot_idx]
                
                try: # Shape Type (Normalized ID)
                    s_type_idx = self.shape_types_enum.index(station.shape.type)
                    obs[obs_idx_ptr] = float(s_type_idx) / (self.num_shape_types -1 + 1e-6) # ensure not 0/0 if num_shape_types is 1
                except ValueError:
                    obs[obs_idx_ptr] = 0.0 
                obs_idx_ptr += 1

                pass_count_norm = len(station.passengers) / (STATION_MAX_CAPACITY + 1e-6) # Passenger Count (Normalized)
                obs[obs_idx_ptr] = pass_count_norm
                obs_idx_ptr += 1

                obs[obs_idx_ptr] = 1.0 if pass_count_norm > 0.8 else 0.0 # Is Overcrowded Flag
                obs_idx_ptr += 1
                
                num_conns = 0 # Num Connections (Normalized)
                for path_obj in self.mediator.paths:
                    if station in path_obj.stations: 
                        num_conns +=1
                obs[obs_idx_ptr] = num_conns / 6.0 # Normalize by an estimated max connections (e.g. 6)
                obs_idx_ptr += 1

                pass_dest_profile = np.zeros(self.num_shape_types, dtype=np.float32) # Passenger Destination Profile (Normalized per station)
                if len(station.passengers) > 0:
                    for passenger in station.passengers:
                        try:
                            dest_type_idx = self.shape_types_enum.index(passenger.destination_shape.type)
                            pass_dest_profile[dest_type_idx] += 1.0
                        except ValueError: pass
                    pass_dest_profile /= (len(station.passengers) + 1e-6) 
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = pass_dest_profile
                obs_idx_ptr += self.num_shape_types
            else:
                obs[obs_idx_ptr : obs_idx_ptr + (self.station_feature_size_simple - 1)] = 0.0
                obs_idx_ptr += (self.station_feature_size_simple - 1)

        # --- 2. Path Features ---
        for p_slot_idx in range(self.max_paths):
            path_exists_in_slot = (p_slot_idx < len(self.mediator.paths))
            
            obs[obs_idx_ptr] = 1.0 if path_exists_in_slot else 0.0 # path_exists
            obs_idx_ptr += 1

            if path_exists_in_slot:
                path = self.mediator.paths[p_slot_idx]
                
                obs[obs_idx_ptr] = len(path.stations) / (self.max_station_slots + 1e-6) # path_length_norm
                obs_idx_ptr += 1
                obs[obs_idx_ptr] = 1.0 if path.is_looped else 0.0 # is_loop
                obs_idx_ptr += 1
                
                path_station_crowd_sum = 0 # path_crowding_heuristic_norm
                if path.stations:
                    for s_on_path in path.stations:
                        path_station_crowd_sum += len(s_on_path.passengers) / (STATION_MAX_CAPACITY + 1e-6)
                    obs[obs_idx_ptr] = path_station_crowd_sum / (len(path.stations) + 1e-6)
                else:
                    obs[obs_idx_ptr] = 0.0
                obs_idx_ptr += 1

                shapes_on_path_mhot = np.zeros(self.num_shape_types, dtype=np.float32) # path_station_shapes_covered_multi_hot
                if path.stations:
                    for s_on_path in path.stations:
                        try:
                            s_type_idx = self.shape_types_enum.index(s_on_path.shape.type)
                            shapes_on_path_mhot[s_type_idx] = 1.0
                        except ValueError: pass
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = shapes_on_path_mhot
                obs_idx_ptr += self.num_shape_types
            else:
                obs[obs_idx_ptr : obs_idx_ptr + (self.path_feature_size - 1)] = 0.0
                obs_idx_ptr += (self.path_feature_size - 1)

        # --- 3. Global Features ---
        obs[obs_idx_ptr] = len(self.mediator.paths) / (self.max_paths + 1e-6) # num_existing_paths_norm
        obs_idx_ptr += 1
        current_score_norm = self.mediator.score / (self.mediator.steps + 1e-5) if self.mediator.steps > 0 else 0.0 # score_norm
        obs[obs_idx_ptr] = current_score_norm
        obs_idx_ptr += 1

        total_pass_waiting = sum(len(s.passengers) for s in self.mediator.stations)
        max_possible_pass = self.max_station_slots * STATION_MAX_CAPACITY + 1e-6
        obs[obs_idx_ptr] = total_pass_waiting / max_possible_pass if max_possible_pass > 0 else 0.0 # total_pass_waiting_norm
        obs_idx_ptr += 1

        num_overcrowded_stations = sum(1 for s in self.mediator.stations if len(s.passengers) / (STATION_MAX_CAPACITY + 1e-6) > 0.8)
        obs[obs_idx_ptr] = num_overcrowded_stations / (len(self.mediator.stations) + 1e-6) if len(self.mediator.stations) > 0 else 0.0 # frac_stations_overcrowded_norm
        obs_idx_ptr += 1
        
        if obs_idx_ptr != self.total_obs_size:
             print(f"CRITICAL WARNING: Observation size mismatch! Expected {self.total_obs_size}, got {obs_idx_ptr}. This will crash SB3.")
             # This can happen if feature sizes are miscalculated or pointer logic is flawed.
             # For now, to prevent crash during testing, pad or truncate, but this indicates a bug.
             if obs_idx_ptr < self.total_obs_size:
                 obs[obs_idx_ptr:] = 0.0 # Pad with zeros
             else: # obs_idx_ptr > self.total_obs_size
                 obs = obs[:self.total_obs_size] # Truncate
             print(f"Corrected obs length to {len(obs)}")


        return np.clip(obs, -1.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
=======
            if not pygame.display.get_init(): # 确保显示模块已初始化
                 pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Mini Metro RL (Simple Env)")
            self.clock = pygame.time.Clock()

        # --- 动作空间 ---
        # 0: NO_OP
        # 1: DELETE_ALL_COMPLETED_PATHS
        # 2: FINISH_PATH_IN_PROGRESS
        # 3: CLEAR_PATH_IN_PROGRESS
        # 4 to 3+max_station_slots: ADD_STATION_SLOT_idx_TO_PATH_IN_PROGRESS
        self.num_action_types_exclusive_of_add_station = 4
        self.action_space = spaces.Discrete(self.num_action_types_exclusive_of_add_station + self.max_station_slots)

        # --- 状态空间 (观测空间) ---
        # 1. 站点特征 (每个槽位)
        #    exists(1), shape_one_hot(num_shapes), passenger_dest_profile(num_shapes),
        #    is_in_current_path(1), is_start_current(1), is_last_current(1)
        self.station_feature_size = 1 + self.num_shape_types + self.num_shape_types + 1 + 1 + 1
        obs_size_stations = self.max_station_slots * self.station_feature_size

        # 2. 连接矩阵 (站点槽位之间的邻接矩阵)
        obs_size_connectivity = self.max_station_slots * self.max_station_slots

        # 3. 全局 / 当前构建线路信息
        #    num_existing_paths_norm(1), len_current_path_norm(1), score_norm(1)
        obs_size_global = 3
        
        self.total_obs_size = obs_size_stations + obs_size_connectivity + obs_size_global + 1  # +1 for new_station_flag

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_obs_size,), dtype=np.float32)

        # 环境内部状态，用于智能体构建线路
        self.current_path_slot_indices: List[int] = [] # 存储观测空间中的站点槽位索引
        self.current_path_potential_loop: bool = False # 当前构建的线路是否可能形成环路

        self.last_score = 0
        self.current_rl_step = 0 # 当前 RL episode 中的步数
        
        self.game_ticks_per_rl_step = 20 # 每个 RL 动作后，游戏模拟多少个 tick
        self.max_rl_steps_per_episode = 15000 # RL episode 的最大长度，用于 truncated

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.total_obs_size, dtype=np.float32)
        obs_idx_ptr = 0 # 当前观测向量的写入位置

        # --- 1. 站点特征 ---
        # 创建一个从活动站点对象 ID 到其在观测中槽位索引的映射，用于快速查找
        # 注意：这里的槽位索引是 self.mediator.stations 列表中的索引
        active_station_id_to_slot_map: Dict[int, int] = {
            id(self.mediator.stations[i]): i 
            for i in range(len(self.mediator.stations))
            if i < self.max_station_slots # 确保不超过观测空间定义的最大站点数
        }

        for s_slot_idx in range(self.max_station_slots):
            # 判断此槽位是否有真实存在的站点
            station_exists = (s_slot_idx < len(self.mediator.stations))
            
            # 记录当前站点特征的起始写入位置，方便填充0（如果站点不存在）
            # start_station_features_ptr = obs_idx_ptr 

            # 特征: exists (是否存在)
            obs[obs_idx_ptr] = 1.0 if station_exists else 0.0
            obs_idx_ptr += 1

            if station_exists:
                station = self.mediator.stations[s_slot_idx] # 获取真实的站点对象
                
                # 特征: shape_one_hot (站点形状的独热编码)
                shape_one_hot = np.zeros(self.num_shape_types, dtype=np.float32)
                try:
                    s_type_idx = self.shape_types_enum.index(station.shape.type)
                    shape_one_hot[s_type_idx] = 1.0
                except ValueError: pass # 如果形状不在预定义的列表中，则跳过 (理论上不应发生)
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = shape_one_hot
                obs_idx_ptr += self.num_shape_types

                # 特征: passenger_dest_profile (站点中等待前往各目标形状的乘客数量，已标准化)
                pass_dest_profile = np.zeros(self.num_shape_types, dtype=np.float32)
                for passenger in station.passengers:
                    try:
                        # 获取乘客目标形状在形状枚举列表中的索引
                        dest_type_idx = self.shape_types_enum.index(passenger.destination_shape.type)
                        pass_dest_profile[dest_type_idx] += 1.0
                    except ValueError: pass
                # 标准化：除以站点最大容量
                obs[obs_idx_ptr : obs_idx_ptr + self.num_shape_types] = pass_dest_profile / (STATION_MAX_CAPACITY + 1e-6)
                obs_idx_ptr += self.num_shape_types

                # 特征: is_in_current_path_being_built (此站点是否在当前构建的线路中)
                obs[obs_idx_ptr] = 1.0 if s_slot_idx in self.current_path_slot_indices else 0.0
                obs_idx_ptr += 1
                # 特征: is_start_of_current_path (是否为当前构建线路的起点)
                obs[obs_idx_ptr] = 1.0 if self.current_path_slot_indices and self.current_path_slot_indices[0] == s_slot_idx else 0.0
                obs_idx_ptr += 1
                # 特征: is_last_in_current_path (是否为当前构建线路的最新加入站点)
                obs[obs_idx_ptr] = 1.0 if self.current_path_slot_indices and self.current_path_slot_indices[-1] == s_slot_idx else 0.0
                obs_idx_ptr += 1
            else: # 如果此槽位没有站点，用0填充剩余特征
                # 'exists' 特征已设为0.0
                # 需要填充的剩余特征数量 = self.station_feature_size - 1 (减去 'exists' 特征)
                remaining_features_count = self.station_feature_size - 1
                obs[obs_idx_ptr : obs_idx_ptr + remaining_features_count] = 0.0
                obs_idx_ptr += remaining_features_count
        
        # --- 2. 连接矩阵 ---
        # adj[i,j] = 1.0 表示观测槽位 i 中的站点在某条 *已完成* 线路中直接连接到观测槽位 j 中的站点
        adj_matrix = np.zeros((self.max_station_slots, self.max_station_slots), dtype=np.float32)
        for path in self.mediator.paths: # 遍历所有已完成的线路
            if len(path.stations) >= 2: # 至少需要两个站点才能构成连接
                for k_path_idx in range(len(path.stations) - 1): # 遍历线路中的站点对
                    station_A_obj = path.stations[k_path_idx]
                    station_B_obj = path.stations[k_path_idx+1]
                    
                    # 将站点对象映射到它们在观测空间的槽位索引
                    obs_slot_A = active_station_id_to_slot_map.get(id(station_A_obj), -1)
                    obs_slot_B = active_station_id_to_slot_map.get(id(station_B_obj), -1)

                    if obs_slot_A != -1 and obs_slot_B != -1: # 确保两个站点都在当前观测槽位中
                        adj_matrix[obs_slot_A, obs_slot_B] = 1.0 # 标记连接
                
                if path.is_looped and len(path.stations) >=2: # 如果是环线，最后一个站点连接到第一个
                    station_Last_obj = path.stations[-1]
                    station_First_obj = path.stations[0]
                    obs_slot_Last = active_station_id_to_slot_map.get(id(station_Last_obj), -1)
                    obs_slot_First = active_station_id_to_slot_map.get(id(station_First_obj), -1)
                    if obs_slot_Last != -1 and obs_slot_First != -1:
                         adj_matrix[obs_slot_Last, obs_slot_First] = 1.0
        
        obs[obs_idx_ptr : obs_idx_ptr + adj_matrix.size] = adj_matrix.flatten() # 展平连接矩阵并加入观测
        obs_idx_ptr += adj_matrix.size

        # --- 3. 全局 / 当前构建线路信息 ---
        # 特征: num_existing_paths_norm (已完成线路数量，标准化)
        obs[obs_idx_ptr] = len(self.mediator.paths) / self.max_paths if self.max_paths > 0 else 0.0
        obs_idx_ptr += 1
        # 特征: len_current_path_norm (当前构建线路的长度，标准化)
        obs[obs_idx_ptr] = len(self.current_path_slot_indices) / self.max_station_slots if self.max_station_slots > 0 else 0.0
        obs_idx_ptr += 1
        # 特征: score_norm (当前分数，简单标准化)
        obs[obs_idx_ptr] = self.mediator.score / (self.mediator.steps + 1e-5) # 防止除以0
        obs_idx_ptr += 1
        # 特徵: 是否有新站點產生 (0 or 1)
        obs[obs_idx_ptr] = 1.0 if self.new_station_flag else 0.0
        obs_idx_ptr += 1

        
        return np.clip(obs, -1.0, 1.0) # 确保观测值在 [-1, 1] 范围内

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed) # 处理gymnasium内部的随机种子
        
        # 如果是评估环境且未提供种子，使用固定种子以保证评估的可复现性
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
        if seed is None and self.is_eval_env:
            seed = 42 
        
        if seed is not None:
<<<<<<< HEAD
            self.mediator.seed = seed
            random.seed(seed)
            np.random.seed(seed)

        self.mediator.reset_progress()
        self.last_score = 0
        self.current_rl_step = 0
        self.high_level_action_counts = [0] * len(self.high_level_actions_enum_list)

        initial_dt_ms = 1000 // (self.metadata['render_fps'] or 30)
        for _ in range(10): 
            if len(self.mediator.stations) >= min(3, self.max_station_slots):
                break
            self.mediator.increment_time(initial_dt_ms)
        
        if len(self.mediator.stations) > 0:
            self.mediator.increment_time(1)

        observation = self._get_obs()
        info = self._get_info()
        self.last_score = self.mediator.score
        return observation, info

    def step(self, action: int, truncated_on=True):
        reward = 0.0
        terminated = False
        truncated = False
        action_execution_penalty = -0.1 

        chosen_high_level_action = self.high_level_actions_enum_list[action]
        self.high_level_action_counts[action] += 1

        action_executed_successfully = False
        if chosen_high_level_action == HighLevelAction.NO_OP or self.current_rl_step % 5 != 0:
            action_executed_successfully = True # NO_OP is always successful
        elif chosen_high_level_action == HighLevelAction.BUILD_NEW_PATH_SMART:
            success, _ = self._execute_build_new_path_heuristic()
            if success:
                #reward += 0.5 
                action_executed_successfully = True
        elif chosen_high_level_action == HighLevelAction.EXTEND_EXISTING_PATH_SMART:
            success, _ = self._execute_extend_existing_path_heuristic()
            if success:
                #reward += 0.3
                action_executed_successfully = True
            
        elif chosen_high_level_action == HighLevelAction.DELETE_LEAST_USEFUL_PATH:
            success, _ = self._execute_delete_least_useful_path_heuristic()
            if success:
                #reward += 0.1 
                action_executed_successfully = True
        
        game_outcome_state = MeditatorState.RUNNING
        dt_ms = 1000 // (self.metadata['render_fps'] or 30)

=======
            self.mediator.seed = seed # 设置游戏逻辑核心的种子
            random.seed(seed)         # 设置Python内置random的种子
            np.random.seed(seed)      # 设置NumPy的种子

        self.mediator.reset_progress() # 重置游戏逻辑状态
        self.last_score = 0
        self.current_rl_step = 0
        self.current_path_slot_indices = []
        self.current_path_potential_loop = False

        # 游戏开始时，通常会有几个初始站点。让游戏模拟运行几步以生成它们。
        initial_dt_ms = 1000 // (framerate or 30) 
        for _ in range(10): # 模拟一小段时间
            # Mini Metro 通常以3个站点开始，或者达到最大站点数的一小部分
            if len(self.mediator.stations) >= min(3, self.max_station_slots):
                break
            self.mediator.increment_time(initial_dt_ms)
            self.prev_num_stations = len(self.mediator.stations)
            self.new_station_flag = True  # 開局會被當作「新站點」

        
        # 如果有站点了，再运行一小步，确保乘客生成逻辑有机会运行
        if len(self.mediator.stations) > 0:
            self.mediator.increment_time(1)


        observation = self._get_obs()
        info = self._get_info()
        self.last_score = self.mediator.score # 重置后，根据初始状态更新分数
        return observation, info

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False
        invalid_action_penalty = -0.05

        # --- 動作處理 ---
        if action == 0:  # NO_OP
            pass
        elif action == 1:  # DELETE_ALL_COMPLETED_PATHS
            paths_to_cancel = list(self.mediator.paths)
            for path in paths_to_cancel:
                self.mediator.cancel_path(path)
        elif action == 2:  # FINISH_PATH_IN_PROGRESS
            if len(self.current_path_slot_indices) >= 2 and len(self.mediator.paths) < self.max_paths:
                actual_stations_in_path = []
                valid_path = True
                for idx in self.current_path_slot_indices:
                    if idx < len(self.mediator.stations):
                        actual_stations_in_path.append(self.mediator.stations[idx])
                    else:
                        valid_path = False
                        break
                if valid_path and len(actual_stations_in_path) >= 2:
                    self.mediator.start_path_on_station(actual_stations_in_path[0])
                    for station in actual_stations_in_path[1:]:
                        self.mediator.add_station_to_path(station)
                    self.mediator.finish_path_creation()
                    reward += 1.0  # 成功建立路線獎勵
                else:
                    reward += invalid_action_penalty * 2
                self.current_path_slot_indices = []
                self.current_path_potential_loop = False
            else:
                reward += invalid_action_penalty
                self.current_path_slot_indices = []
                self.current_path_potential_loop = False
        elif action == 3:  # CLEAR_PATH_IN_PROGRESS
            self.current_path_slot_indices = []
            self.current_path_potential_loop = False
        else:  # ADD_STATION_SLOT_idx_TO_PATH_IN_PROGRESS
            slot_idx = action - self.num_action_types_exclusive_of_add_station
            if 0 <= slot_idx < len(self.mediator.stations):
                if not self.current_path_slot_indices:
                    self.current_path_slot_indices.append(slot_idx)
                elif slot_idx != self.current_path_slot_indices[-1]:
                    self.current_path_slot_indices.append(slot_idx)
                    self.current_path_potential_loop = (
                        slot_idx == self.current_path_slot_indices[0]
                    )
                else:
                    reward += invalid_action_penalty * 0.5
            else:
                reward += invalid_action_penalty

        # --- 模擬遊戲邏輯 ---
        game_outcome_state = MeditatorState.RUNNING
        dt_ms = 1000 // (framerate or 30)
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
        for _ in range(self.game_ticks_per_rl_step):
            if self.visuals and self.screen:
                for pygame_event in pygame.event.get():
                    if pygame_event.type == pygame.QUIT:
<<<<<<< HEAD
                        terminated = True; break
            if terminated: break
            
            game_outcome_state = self.mediator.increment_time(dt_ms)
            if game_outcome_state == MeditatorState.ENDED:
                # if not self.is_eval_env: # Print only during training
                #     print(f"HL Action Counts: {self.high_level_action_counts}, Total: {sum(self.high_level_action_counts)}")
                self.high_level_action_counts = [0] * len(self.high_level_actions_enum_list)
                terminated = True; break
        
        reward += (self.mediator.score - self.last_score)
        self.last_score = self.mediator.score

        if terminated and game_outcome_state == MeditatorState.ENDED:
            reward -= 50.0 

        num_crowded_stations = sum(1 for s in self.mediator.stations if len(s.passengers) / (STATION_MAX_CAPACITY + 1e-6) > 0.8)
        reward -= num_crowded_stations * 0.01

        self.current_rl_step += 1
        if truncated_on and self.current_rl_step >= self.max_rl_steps_per_episode:
            truncated = True
            if not terminated:
                reward -= 0.5
                if not self.is_eval_env:
                    print(f"Truncated - HL Action Counts: {self.high_level_action_counts}, Total: {sum(self.high_level_action_counts)}")
                self.high_level_action_counts = [0] * len(self.high_level_actions_enum_list)
=======
                        terminated = True
                        break
            if terminated:
                break
            game_outcome_state = self.mediator.increment_time(dt_ms)
            if game_outcome_state == MeditatorState.ENDED:
                terminated = True
                break

        # --- 計算獎勵 ---
        score_gain = self.mediator.score - self.last_score
        reward += score_gain * 1.0  # 根據分數增長給予正獎勵
        self.last_score = self.mediator.score



        # 擁擠站點懲罰
        num_critical = sum(
            1 for s in self.mediator.stations if len(s.passengers) > 0.9 * STATION_MAX_CAPACITY
        )
        reward -= num_critical * 0.5

        # 爆站懲罰
        num_overload = sum(
            1 for s in self.mediator.stations if len(s.passengers) >= STATION_MAX_CAPACITY
        )
        reward -= num_overload * 2.0

        # 終止或截斷處理
        self.current_rl_step += 1
        if terminated and game_outcome_state == MeditatorState.ENDED:
            reward -= 10.0
        elif self.current_rl_step >= self.max_rl_steps_per_episode:
            truncated = True
            if not terminated:
                reward += 1.0
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20

        observation = self._get_obs()
        info = self._get_info()

        if self.visuals:
            self.render()
<<<<<<< HEAD
            
        return observation, reward, terminated, truncated, info

    def _execute_build_new_path_heuristic(self) -> Tuple[bool, Dict]:
        if len(self.mediator.paths) >= self.max_paths:
            return False, {"reason": "Max paths reached"}

        candidate_starts = []
        # Prefer stations that are observable
        observable_stations = [s for i, s in enumerate(self.mediator.stations) if i < self.max_station_slots]

        for station in observable_stations:
            score = len(station.passengers) 
            # TODO: Add more sophisticated scoring (e.g., underserved shapes, low connectivity)
            if score > 0:
                 candidate_starts.append({'station_obj': station, 'score': score})
        
        if not candidate_starts:
            return False, {"reason": "No suitable start stations found"}
        
        candidate_starts.sort(key=lambda x: x['score'], reverse=True)
        
        for start_candidate in candidate_starts[:min(3, len(candidate_starts))]:
            start_station_obj = start_candidate['station_obj']
            
            candidate_ends = []
            for end_station_obj in observable_stations:
                if end_station_obj != start_station_obj:
                    # TODO: Add scoring for end stations (e.g., passenger destination match, proximity)
                    # For now, just any different station
                    candidate_ends.append({'station_obj': end_station_obj})

            if not candidate_ends:
                continue

            # TODO: Select best end station based on score
            end_station_obj_to_connect = random.choice(candidate_ends)['station_obj'] # Simple random choice for now

            self.mediator.start_path_on_station(start_station_obj)
            self.mediator.add_station_to_path(end_station_obj_to_connect)
            # TODO: Consider heuristically adding a second segment if beneficial (e.g., to form a 3-station line)
            
            path_creation_successful = self.mediator.finish_path_creation() # Assume this returns bool

            if path_creation_successful:
                return True, {"start_station": start_station_obj.id, "end_station": end_station_obj_to_connect.id}
            
                
        return False, {"reason": "Could not form a valid path with available candidates"}

    def _execute_extend_existing_path_heuristic(self) -> Tuple[bool, Dict]:
        if not self.mediator.paths:
            return False, {"reason": "No paths to extend"}
        
        observable_stations = [s for i, s in enumerate(self.mediator.stations) if i < self.max_station_slots]

        candidate_paths = []
        for i, path in enumerate(self.mediator.paths):
            # TODO: Score paths for extension (e.g., paths with crowded stations, shorter paths)
            score = -len(path.stations) # Prefer extending shorter paths
            candidate_paths.append({'path_obj': path, 'path_idx_in_mediator': i, 'score': score})
        
        candidate_paths.sort(key=lambda x: x['score'], reverse=True) # Higher score is better candidate

        for path_candidate in candidate_paths[:min(3, len(candidate_paths))]:
            path_to_extend = path_candidate['path_obj']
            if not path_to_extend.stations: continue # Should not happen for valid paths

            last_station_in_path = path_to_extend.stations[-1]
            
            possible_extensions = []
            for station_obj in observable_stations:
                if station_obj not in path_to_extend.stations:
                    # TODO: Score extension candidates (proximity, passenger needs)
                    possible_extensions.append({'station_obj': station_obj})
            
            if not possible_extensions:
                continue

            station_to_add = random.choice(possible_extensions)['station_obj'] # Simple random choice

            # This is a simplified extension: delete old, create new.
            # A more sophisticated Mediator might allow direct extension.
            original_stations_data = [(s.id, s.shape.type, s.position) for s in path_to_extend.stations] # Or just station objects if they persist
            original_stations_objects = list(path_to_extend.stations)
            
            self.mediator.cancel_path(path_to_extend)

            # Rebuild with the new station
            if not original_stations_objects: continue

            # Ensure start_station_obj is still valid/findable if stations list in mediator changes
            # For simplicity, assume original_stations_objects[0] is still findable or use its properties to re-find.
            # This part is tricky if station objects are not stable references after path deletion/creation.
            # We assume station objects themselves are persistent and can be reused.
            
            # Find the station object in the current self.mediator.stations list
            current_start_station_obj = None
            for s in self.mediator.stations: # Search for the first station of the old path
                if s == original_stations_objects[0]: # Comparing by object reference
                    current_start_station_obj = s
                    break
            if not current_start_station_obj:
                # Fallback if original object reference is somehow lost/changed (e.g., station got deleted and re-added)
                # This part would need robust handling based on station IDs if they exist and are stable.
                # For now, if start station is gone, we can't rebuild.
                # Try to re-add the original path (best effort)
                # self._try_recreate_path(original_stations_objects) # You'd need this helper
                return False, {"reason": "Original start station for extension not found"}


            self.mediator.start_path_on_station(current_start_station_obj)
            for s_obj in original_stations_objects[1:]:
                current_s_obj = None # Find s_obj in current mediator.stations
                for s_curr in self.mediator.stations:
                    if s_curr == s_obj:
                        current_s_obj = s_curr
                        break
                if not current_s_obj: return False, {"reason": "Intermediate station for extension not found"}
                self.mediator.add_station_to_path(current_s_obj)
            
            current_station_to_add = None # Find station_to_add in current mediator.stations
            for s_curr in self.mediator.stations:
                if s_curr == station_to_add:
                    current_station_to_add = s_curr
                    break
            if not current_station_to_add: return False, {"reason": "Station to add for extension not found"}

            self.mediator.add_station_to_path(current_station_to_add)
            
            rebuild_successful = self.mediator.finish_path_creation()

            if rebuild_successful:
                return True, {"extended_path_original_idx": path_candidate['path_idx_in_mediator'], "added_station": station_to_add.id}
            else:
                # self.mediator.clear_path_creation()
                # TODO: Attempt to restore the original path if rebuild fails.
                # This is complex. For now, accept failure.
                # self._try_recreate_path(original_stations_objects)
                return False, {"reason": "Failed to rebuild path after extension attempt"}
                
        return False, {"reason": "No suitable extension found"}

    def _execute_delete_least_useful_path_heuristic(self) -> Tuple[bool, Dict]:
        if not self.mediator.paths:
            return False, {"reason": "No paths to delete"}

        # scored_paths = []
        for i, path in enumerate(self.mediator.paths):
            self.mediator.cancel_path(path)
        
        # scored_paths.sort(key=lambda x: x['score']) # Lower score = less useful
        
        # if not scored_paths: # Should not happen if self.mediator.paths is not empty
        #      return False, {"reason": "Path scoring failed"}

        return True, {}


    def render(self):
        if not self.visuals and 'rgb_array' not in self.metadata['render_modes']:
            return None
        
        if self.screen is None: 
            if not pygame.display.get_init(): pygame.display.init()
            self.screen = pygame.Surface((screen_width, screen_height)) 
        
        self.screen.fill(screen_color)
        if 'draw_waves' in globals() and callable(draw_waves): 
             draw_waves(self.screen, self.mediator.time_ms)
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
            # "path_in_progress_len": 0, # No longer agent-controlled path in progress
            "passengers_waiting_total": sum(len(s.passengers) for s in self.mediator.stations),
            "high_level_action_distribution": self.high_level_action_counts, # For debugging
        }

    def close(self):
        if self.visuals or (self.screen is not None):
            pygame.display.quit()
        self.screen = None
        self.clock = None

# --- 训练与评估脚本示例 --- (Largely unchanged, but uses the new env)
def run_training(load=False):
    print("启动 Mini Metro 环境 (Hierarchical Actions) 的训练与评估...")

    train_env = MiniMetroSimpleEnv(visuals=False, gamespeed=10) 
    # check_env(train_env) # Good to run this once after major changes

    log_dir = "./mini_metro_hierarchical_rl_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + 'best_model/', exist_ok=True)
    os.makedirs(log_dir + 'results/', exist_ok=True)
    os.makedirs(log_dir + "tensorboard/", exist_ok=True)

    model_save_name = "ppo_mini_metro_hierarchical"

    eval_env = MiniMetroSimpleEnv(visuals=False, gamespeed=1, is_eval_env=True) 
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=log_dir + 'best_model/',
                                 log_path=log_dir + 'results/', 
                                 eval_freq=20000, # Adjusted eval_freq
                                 deterministic=True, 
                                 render=False)

    # Hyperparameters might need tuning for the new state/action space
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir + "tensorboard/",
                gamma=0.99,      
                gae_lambda=0.95,
                n_steps=2048, # Might need adjustment
                ent_coef=0.1, # Might need adjustment  
                learning_rate=3e-4, 
                vf_coef=0.5,        
                max_grad_norm=0.5,  
                batch_size=128, # Might need adjustment    
                n_epochs=20,        
                seed=42,
                device='cpu' # or 'cuda' if available
               )
    if load:
        try:
            final_model_path = log_dir + model_save_name + ".zip"
            model = PPO.load(final_model_path, env=train_env)
            print(f"已加载模型: {final_model_path}")
        except FileNotFoundError:
             print(f"模型文件未找到: {final_model_path}. 将从头开始训练.")

=======

        return observation, reward, terminated, truncated, info


    def render(self):
        # 如果没有开启视觉效果，并且渲染模式中不包含 'rgb_array'，则不执行渲染
        if not self.visuals and 'rgb_array' not in self.metadata['render_modes']:
            return None
        
        # 如果屏幕对象未初始化 (例如，仅为 'rgb_array' 模式且未进行 'human' 模式渲染)
        if self.screen is None: 
            if not pygame.display.get_init(): pygame.display.init() # 确保显示模块已初始化
            # 创建一个离屏表面用于渲染，这样即使没有窗口也能得到图像数据
            self.screen = pygame.Surface((screen_width, screen_height)) 
        
        # 通用渲染流程
        self.screen.fill(screen_color) # 填充背景色
        # 检查 draw_waves 是否可用并且可调用，然后渲染背景波浪
        if 'draw_waves' in globals() and callable(draw_waves): 
             draw_waves(self.screen, self.mediator.time_ms)
        self.mediator.render(self.screen) # Mediator 负责绘制所有游戏实体

        # 如果是 'human' 渲染模式并且开启了视觉效果，则更新到屏幕上显示
        if 'human' in self.metadata['render_modes'] and self.visuals:
            pygame.display.flip() # 更新整个屏幕
            if self.clock: self.clock.tick(self.metadata['render_fps']) # 控制帧率
            return None # 'human' 模式下 render 通常返回 None
        
        # 如果是 'rgb_array' 渲染模式，返回屏幕图像的NumPy数组
        if 'rgb_array' in self.metadata['render_modes']:
            # 将 Pygame Surface 转换为 NumPy 数组，并调整维度顺序 (W,H,C) -> (H,W,C)
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2) 
        
        return None # 其他情况或默认不返回

    def _get_info(self) -> Dict: # 返回包含辅助信息的字典
        return {
            "score": self.mediator.score,
            "total_game_ticks": self.mediator.steps,    # Mediator 内部的游戏 tick 数
            "rl_episode_steps": self.current_rl_step, # 当前 RL episode 的步数
            "num_stations": len(self.mediator.stations),
            "num_completed_paths": len(self.mediator.paths),
            "path_in_progress_len": len(self.current_path_slot_indices),
            "passengers_waiting_total": sum(len(s.passengers) for s in self.mediator.stations),
        }

    def close(self): # 清理环境资源
        if self.visuals or (self.screen is not None): # 如果屏幕曾被初始化
            pygame.display.quit() # 关闭Pygame显示模块
        # pygame.font.quit() # 通常不需要单独关闭字体，pygame.quit()会处理
        # pygame.quit() # 如果应用中其他部分也使用Pygame，全局关闭可能不合适
        self.screen = None
        self.clock = None

# --- 训练与评估脚本示例 ---
def run_training(log_dir, model_save_name, load=False):
    print("启动精简版 Mini Metro 环境的训练与评估...")

    # 创建环境实例
    # 训练时通常关闭视觉效果并加速游戏以提高效率


    def make_env():
        return MiniMetroSimpleEnv(visuals=False, gamespeed=20)

    num_envs = 12  # 或 8、16 視電腦核心數
    train_env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # (可选) 检查环境是否符合 Stable Baselines3 的规范
    # print("检查环境规范性...")
    # check_env(train_env) 
    # print("环境检查通过。")

    # 设置评估环境和回调函数
    # 使用与训练环境分开的评估环境是一个好习惯，通常 gamespeed=1
    eval_env = MiniMetroSimpleEnv(visuals=False, gamespeed=1, is_eval_env=True) 
    # EvalCallback 会定期评估模型，并保存表现最好的模型
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=log_dir + 'best_model/',
                                 log_path=log_dir + 'results/', 
                                 eval_freq=10000, # 每10000个 agent 步数评估一次
                                 deterministic=True, 
                                 render=False)

    # 定义强化学习 Agent (使用PPO算法)
    # 超参数可能需要根据实际情况调整，这里提供了一些常用值作为起点
    model = PPO("MlpPolicy",        # 使用多层感知机策略网络
                train_env,          # 训练环境
                verbose=0,          # 打印训练过程信息
                tensorboard_log=log_dir + "tensorboard/", # TensorBoard 日志路径
                gamma=0.99,         # 折扣因子
                gae_lambda=0.95,    # GAE lambda 参数
                n_steps=1024,       # 每个 rollout/update 收集的步数
                ent_coef=0.005,     # 熵正则化系数，鼓励探索
                learning_rate=2.5e-4, # 学习率
                vf_coef=0.5,        # 值函数损失系数
                max_grad_norm=0.5,  # 梯度裁剪范数
                batch_size=64,      # 每个 PPO epoch 的 minibatch 大小
                n_epochs=10,        # 每个 PPO update 的 epoch 数
                seed=42,            # 随机种子，保证训练可复现
                device='cpu'
               )
    if load == True:
        final_model_path = log_dir + model_save_name + ".zip"
        model = PPO.load(final_model_path, env=train_env) # 评估时可以不传入env，policy会使用新env
        print(f"已加载模型: {final_model_path}")
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20

    print(f"模型策略网络结构: {model.policy}")
    print(f"观测空间维度: {train_env.observation_space.shape}")
    print(f"动作空间大小: {train_env.action_space.n}")

<<<<<<< HEAD
    total_timesteps_to_train = 200000 # Adjusted total timesteps
    print(f"开始训练，总步数: {total_timesteps_to_train}...")
    try:
        model.learn(total_timesteps=total_timesteps_to_train, 
                    callback=eval_callback,
                    progress_bar=True)
        model.save(log_dir + model_save_name)
        print(f"训练完成。最终模型已保存到 {log_dir + model_save_name}.zip")
=======
    # 训练模型
    total_timesteps_to_train = 300000 # 总训练步数 (可根据需要调整)
    print(f"开始训练，总步数: {total_timesteps_to_train}...")
    try:
        model.learn(total_timesteps=total_timesteps_to_train, 
                    callback=eval_callback, # 在训练过程中加入评估回调
                    progress_bar=True)      # 显示训练进度条
        model.save(log_dir + model_save_name) # 保存最终模型
        print(f"训练完成。最终模型已保存到 {log_dir + model_save_name}")
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
<<<<<<< HEAD
    finally:
        train_env.close()
        eval_env.close()

    print("\n--- 开始评估训练好的模型 ---")
    try:
        eval_model_path = log_dir + 'best_model/best_model.zip'
        if not os.path.exists(eval_model_path):
            eval_model_path = log_dir + model_save_name + ".zip"

        if os.path.exists(eval_model_path):
            loaded_model = PPO.load(eval_model_path)
            print(f"已加载模型进行评估: {eval_model_path}")

            eval_env_for_policy = MiniMetroSimpleEnv(visuals=False, gamespeed=1, is_eval_env=True)
            mean_reward, std_reward = evaluate_policy(loaded_model, 
                                                      eval_env_for_policy, 
                                                      n_eval_episodes=20,
                                                      deterministic=True)
            print(f"评估结果: 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
            eval_env_for_policy.close()
        else:
            print(f"评估模型未找到: {eval_model_path}")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")


def evaluation_visual(): # Renamed to avoid conflict if you have another evaluation function
    log_dir = "./mini_metro_hierarchical_rl_logs/"
    model_save_name = "ppo_mini_metro_hierarchical"
    print("\n--- 使用训练好的模型进行可视化演示 ---")
    try:
        visual_model_path = log_dir + 'best_model/best_model.zip'
        if not os.path.exists(visual_model_path):
            visual_model_path = log_dir + model_save_name + ".zip"
        
        if not os.path.exists(visual_model_path):
            print(f"可视化演示模型未找到: {visual_model_path}")
            return

        loaded_model_visual = PPO.load(visual_model_path)
        print(f"已加载模型进行可视化: {visual_model_path}")

        env_visual = MiniMetroSimpleEnv(visuals=True, gamespeed=1, is_eval_env=True) # Slower gamespeed for viz
        
        for episode in range(1): # Demo 1 episodes
            obs_visual, _ = env_visual.reset()
            terminated_visual = False
            truncated_visual = False
            episode_reward = 0
            print(f"\n--- 可视化演示 Episode {episode+1} ---")
            for step_num in range(env_visual.max_rl_steps_per_episode): # Limit steps for demo
                action_visual, _ = loaded_model_visual.predict(obs_visual, deterministic=True)
                obs_visual, reward_visual, terminated_visual, truncated_visual, info_visual = env_visual.step(action_visual, truncated_on=False)
                episode_reward += reward_visual
                # env_visual.render() # Called in step if visuals=True
                
                chosen_action_name = env_visual.high_level_actions_enum_list[action_visual].name
                # print(f"Step: {step_num}, Action: {chosen_action_name}, Reward: {reward_visual:.2f}, Score: {info_visual['score']}")

                if terminated_visual or truncated_visual:
                    print(f"可视化演示 Episode {episode+1} 结束。最终得分: {info_visual['score']}, "
                          f"RL步数: {info_visual['rl_episode_steps']}, 总奖励: {episode_reward:.2f}")
                    print(f"最终 HL Action 分布: {info_visual['high_level_action_distribution']}")
                    break
            if not (terminated_visual or truncated_visual): # If loop finished due to step_num limit
                 print(f"可视化演示 Episode {episode+1} 达到最大演示步数。最终得分: {info_visual['score']}, 总奖励: {episode_reward:.2f}")

        env_visual.close()
    except Exception as e:
        print(f"可视化演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
=======
    finally: # 确保环境被关闭
        train_env.close()
        eval_env.close()



def evaluation(log_dir, model_save_name):
    
    # --- 使用训练好的模型进行可视化演示 ---
    print("\n--- 使用训练好的模型进行可视化演示 ---")
    try:
        # best_model_path = log_dir + 'best_model/best_model.zip'
        # loaded_model_visual = PPO.load(best_model_path)
        final_model_path = log_dir + model_save_name
        loaded_model_visual = PPO.load(final_model_path)

        env_visual = MiniMetroSimpleEnv(visuals=True, gamespeed=1, is_eval_env=True)
        obs_visual, _ = env_visual.reset()
        for episode in range(3): # 演示5个episodes
            terminated_visual = False
            truncated_visual = False
            episode_reward = 0
            current_step = 0
            while not (terminated_visual or truncated_visual):
                action_visual, _ = loaded_model_visual.predict(obs_visual, deterministic=True)
                obs_visual, reward_visual, terminated_visual, truncated_visual, info_visual = env_visual.step(action_visual)
                episode_reward += reward_visual
                # env_visual.render() # render() 会在 step() 中被调用（如果 visuals=True）
                if (terminated_visual or truncated_visual):
                    print(f"可视化演示 Episode {episode+1} 结束。最终得分: {info_visual['score']}, "
                          f"RL步数: {info_visual['rl_episode_steps']}, 总奖励: {episode_reward:.2f}")
                    obs_visual, _ = env_visual.reset() # 重置环境以开始下一个episode
                    break
        env_visual.close()
    except Exception as e:
        print(f"可视化演示过程中发生错误: {e}")


if __name__ == '__main__':
    # 确保必要的配置变量已定义 (通常在 config.py 中)
    # 这是一个示例检查，实际应用中你可能需要更完善的配置管理
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
    essential_configs_defined = True
    try:
        _ = NUM_HORIZONTAL_GRIDS
        _ = ShapeType
        _ = STATION_MAX_CAPACITY
<<<<<<< HEAD
        _ = Mediator # Check if Mediator class is available
    except NameError as e:
        essential_configs_defined = False
        print(f"错误：一些必要的配置或类未定义: {e}")
        print("请确保 'config.py' 文件存在于Python路径中，并且已正确定义这些变量，且 Mediator 类已导入。")
        print("脚本将中止。")
    
    if essential_configs_defined:
        run_training(load=True)  # Set load=True to continue training or load a model
        evaluation_visual()
=======
    except NameError:
        essential_configs_defined = False
        print("错误：一些必要的配置变量 (如 NUM_HORIZONTAL_GRIDS, ShapeType, STATION_MAX_CAPACITY) 未定义。")
        print("请确保 'config.py' 文件存在于Python路径中，并且已正确定义这些变量。")
        print("脚本将中止。")
    
    if essential_configs_defined:
        # 定义日志和模型保存路径
        log_dir = "./mini_metro_simple_rl_logs"
        model_save_name = "ppo_mini_metro_simple_v2"
        TRAIN = 0
        if TRAIN:
            run_training(log_dir, model_save_name, load=False)
        else:
            evaluation(log_dir, model_save_name)
>>>>>>> 837f38057be0f995470958f2e8f897af4af8ff20
