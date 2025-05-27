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

try:
    from config import (
        num_metros as CONFIG_TOTAL_METROS, # 游戏中的总列车数
        num_paths as CONFIG_MAX_PATHS,     # 最大允许的线路数
        num_stations_max,
        framerate, screen_color, screen_width, screen_height,
        station_grid_size, station_capacity,
    )
    NUM_HORIZONTAL_GRIDS, NUM_VERTICAL_GRIDS = station_grid_size
    STATION_MAX_CAPACITY = station_capacity
    from geometry.type import ShapeType
    from mediator import Mediator, MeditatorState, Path as GamePath # 明确导入Path以用于类型提示
    from visuals.background import draw_waves
    from entity.station import Station
    from entity.metro import Metro
except ImportError as e:
    logger.critical(f"无法导入必要的模块或配置: {e}.")
    raise

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    logger.warning("stable_baselines3 未安装。")
    PPO = None

# 新增常量，用于列车管理
MAX_METROS_PER_PATH = 3  # 单条线路最多允许的列车数 (可配置)
MIN_METROS_PER_PATH = 1  # 单条线路最少需要的列车数 (通常是1)


class MiniMetroEdgeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': framerate if 'framerate' in globals() and framerate else 30}

    def __init__(self, gamespeed: int = 1, visuals: bool = False, is_eval_env: bool = False):
        super().__init__()
        logger.info("Initializing MiniMetroEdgeEnv (with Metro Management and Path Splitting)...")

        if not pygame.font.get_init(): pygame.font.init()
        if visuals and not pygame.get_init(): pygame.init()

        self.mediator = Mediator(gamespeed=gamespeed, gen_stations_first=False)

        self.max_station_slots = num_stations_max
        self.max_paths = CONFIG_MAX_PATHS # 使用配置中的最大路径数
        self.total_initial_metros = CONFIG_TOTAL_METROS # 使用配置中的总列车数

        try:
            self.shape_types_enum: List[ShapeType] = list(ShapeType)
        except TypeError:
             logger.warning("ShapeType non-iterable. Fallback."); self.shape_types_enum = []
        self.num_shape_types = len(self.shape_types_enum)
        if self.num_shape_types == 0: logger.critical("num_shape_types is 0!")

        self.visuals = visuals; self.is_eval_env = is_eval_env
        self.screen: Optional[pygame.Surface] = None; self.clock: Optional[pygame.time.Clock] = None
        if self.visuals:
            # ... (Pygame display init) ...
            if not pygame.display.get_init(): pygame.display.init()
            try:
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Mini Metro RL (Edge+Metro Env)")
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                logger.warning(f"Pygame display init error: {e}. Visuals off."); self.visuals = False


        # --- 动作空间 ---
        # 0: NO_OP
        # 1 to N_PAIRS: ADD_EDGE(sA, sB)
        # N_PAIRS+1 to 2*N_PAIRS: SPLIT_PATH_VIA_EDGE(sA, sB) (新的DELETE_EDGE行为)
        # 2*N_PAIRS+1 to 2*N_PAIRS + max_paths: ADD_METRO_TO_PATH(path_idx)
        # 2*N_PAIRS + max_paths + 1 to 2*N_PAIRS + 2*max_paths: RECALL_METRO_FROM_PATH(path_idx)
        
        self.num_station_ordered_pairs = self.max_station_slots * (self.max_station_slots - 1)
        
        self.action_idx_no_op = 0
        self.action_idx_add_edge_start = 1
        self.action_idx_add_edge_end = self.action_idx_add_edge_start + self.num_station_ordered_pairs -1
        
        self.action_idx_split_path_start = self.action_idx_add_edge_end + 1
        self.action_idx_split_path_end = self.action_idx_split_path_start + self.num_station_ordered_pairs -1
        
        self.action_idx_add_metro_start = self.action_idx_split_path_end + 1
        self.action_idx_add_metro_end = self.action_idx_add_metro_start + self.max_paths -1
        
        self.action_idx_recall_metro_start = self.action_idx_add_metro_end + 1
        self.action_idx_recall_metro_end = self.action_idx_recall_metro_start + self.max_paths -1
        
        self.action_space = spaces.Discrete(self.action_idx_recall_metro_end + 1)
        
        self._sA_sB_pair_map: Dict[int, Tuple[int, int]] = {} # Maps flat pair index to (sA, sB)
        _pair_idx_counter = 0
        for sA in range(self.max_station_slots):
            for sB in range(self.max_station_slots):
                if sA == sB: continue
                self._sA_sB_pair_map[_pair_idx_counter] = (sA, sB)
                _pair_idx_counter += 1
        
        logger.debug(f"Action space size: {self.action_space.n}")
        logger.debug(f"  ADD_EDGE actions: {self.action_idx_add_edge_start} - {self.action_idx_add_edge_end}")
        logger.debug(f"  SPLIT_PATH actions: {self.action_idx_split_path_start} - {self.action_idx_split_path_end}")
        logger.debug(f"  ADD_METRO actions: {self.action_idx_add_metro_start} - {self.action_idx_add_metro_end}")
        logger.debug(f"  RECALL_METRO actions: {self.action_idx_recall_metro_start} - {self.action_idx_recall_metro_end}")

        # --- 状态空间 (观测空间) ---
        # 1. 站点特征
        self.station_feature_size = 1 + self.num_shape_types + 1 + 1 + self.num_shape_types # exists, shape_one_hot, pass_count, timeout_pressure, dest_profile
        obs_size_stations = self.max_station_slots * self.station_feature_size
        # 2. 邻接矩阵
        obs_size_adj_matrix = self.max_station_slots * self.max_station_slots
        # 3. 路径特征 (为每条可能的路径 self.max_paths)
        #    exists(1), is_loop(1), num_stations(1), load(1), station_sequence(max_slots), num_metros(1), max_station_pressure(1)
        self.path_feature_size = 1 + 1 + 1 + 1 + self.max_station_slots + 1 + 1
        obs_size_paths = self.max_paths * self.path_feature_size
        # 4. 全局特征
        #    num_paths(1), avg_path_len(1), score(1), total_wait(1), metros_in_pool(1)
        obs_size_global = 5 
        
        self.total_obs_size = obs_size_stations + obs_size_adj_matrix + obs_size_paths + obs_size_global
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.total_obs_size,), dtype=np.float32)
        logger.info(f"Observation Space Shape: ({self.total_obs_size},)")
        logger.debug(f"  Station feats/station: {self.station_feature_size}, Path feats/path: {self.path_feature_size}")

        self.last_score = 0; self.current_rl_step = 0
        self.game_ticks_per_rl_step = 20

    def _get_station_by_slot_idx(self, slot_idx: int) -> Optional[Station]:
        if 0 <= slot_idx < len(self.mediator.stations): return self.mediator.stations[slot_idx] # type: ignore
        return None

    def _get_path_by_slot_idx(self, path_slot_idx: int) -> Optional[GamePath]: # Use GamePath type hint
        if 0 <= path_slot_idx < len(self.mediator.paths): return self.mediator.paths[path_slot_idx]
        return None

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.total_obs_size, dtype=np.float32)
        obs_idx_ptr = 0
        _station_max_cap_obs = STATION_MAX_CAPACITY if STATION_MAX_CAPACITY > 0 else 1.0

        # --- 1. 站点特征 ---
        station_timeout_pressures = {} # Store for path features: map s_slot_idx to pressure
        for s_slot_idx in range(self.max_station_slots):
            station_obj = self._get_station_by_slot_idx(s_slot_idx)
            obs[obs_idx_ptr] = 1.0 if station_obj else 0.0; obs_idx_ptr += 1 # Exists
            if station_obj:
                shape_one_hot = np.zeros(self.num_shape_types,dtype=np.float32); 
                if self.num_shape_types > 0:
                    try:idx=self.shape_types_enum.index(station_obj.shape.type);shape_one_hot[idx]=1.0
                    except ValueError:pass
                obs[obs_idx_ptr:obs_idx_ptr+self.num_shape_types]=shape_one_hot; obs_idx_ptr+=self.num_shape_types
                obs[obs_idx_ptr]=len(station_obj.passengers)/_station_max_cap_obs; obs_idx_ptr+=1
                acc_p=getattr(station_obj,'time_inactive_penalty_acc_ms',0.0); thr_p=getattr(station_obj,'time_inactive_penalty_threshold_ms',1.0)
                current_pressure = acc_p/(thr_p if thr_p>0 else 1.0)
                obs[obs_idx_ptr]=current_pressure; obs_idx_ptr+=1
                station_timeout_pressures[s_slot_idx] = current_pressure
                pass_dest_prof=np.zeros(self.num_shape_types,dtype=np.float32)
                if len(station_obj.passengers)>0 and self.num_shape_types>0:
                    for p in station_obj.passengers:
                        try:idx=self.shape_types_enum.index(p.destination_shape.type);pass_dest_prof[idx]+=1.0
                        except ValueError:pass
                    pass_dest_prof/=(len(station_obj.passengers)+1e-6)
                obs[obs_idx_ptr:obs_idx_ptr+self.num_shape_types]=pass_dest_prof; obs_idx_ptr+=self.num_shape_types
            else: obs_idx_ptr += (self.station_feature_size - 1)
        
        # --- 2. 邻接矩阵 ---
        adj_matrix = np.zeros((self.max_station_slots, self.max_station_slots), dtype=np.float32)
        s_obj_to_slot_map: Dict[int, int] = {id(self.mediator.stations[i]): i for i in range(len(self.mediator.stations)) if i < self.max_station_slots}
        for path_obj in self.mediator.paths:
            for i in range(len(path_obj.stations) - 1):
                sA_slot = s_obj_to_slot_map.get(id(path_obj.stations[i]), -1)
                sB_slot = s_obj_to_slot_map.get(id(path_obj.stations[i+1]), -1)
                if sA_slot != -1 and sB_slot != -1: adj_matrix[sA_slot, sB_slot] = 1.0
            if path_obj.is_looped and len(path_obj.stations) > 1:
                sLast_slot=s_obj_to_slot_map.get(id(path_obj.stations[-1]),-1);sFirst_slot=s_obj_to_slot_map.get(id(path_obj.stations[0]),-1)
                if sLast_slot!=-1 and sFirst_slot!=-1: adj_matrix[sLast_slot,sFirst_slot]=1.0
        obs[obs_idx_ptr:obs_idx_ptr+adj_matrix.size]=adj_matrix.flatten(); obs_idx_ptr+=adj_matrix.size

        # --- 3. 路径特征 ---
        for p_slot_idx in range(self.max_paths):
            path_obj = self._get_path_by_slot_idx(p_slot_idx)
            obs[obs_idx_ptr] = 1.0 if path_obj else 0.0; obs_idx_ptr += 1 # Exists
            if path_obj:
                obs[obs_idx_ptr]=1.0 if path_obj.is_looped else 0.0; obs_idx_ptr+=1
                obs[obs_idx_ptr]=len(path_obj.stations)/(self.max_station_slots+1e-6); obs_idx_ptr+=1
                crowd_sum=sum(len(s.passengers)/_station_max_cap_obs for s in path_obj.stations) if path_obj.stations else 0.0
                obs[obs_idx_ptr]=crowd_sum/(len(path_obj.stations)+1e-6) if path_obj.stations else 0.0; obs_idx_ptr+=1
                
                seq_indices=np.full(self.max_station_slots,-1.0,dtype=np.float32)
                max_pressure_on_path = 0.0
                for i,s_on_p in enumerate(path_obj.stations):
                    if i<self.max_station_slots:
                        s_obs_slot = s_obj_to_slot_map.get(id(s_on_p),-1)
                        if s_obs_slot != -1:
                            norm_slot_idx = float(s_obs_slot)/(self.max_station_slots-1) if self.max_station_slots>1 else 0.0
                            seq_indices[i]=norm_slot_idx
                            max_pressure_on_path = max(max_pressure_on_path, station_timeout_pressures.get(s_obs_slot, 0.0))
                    else: break
                obs[obs_idx_ptr:obs_idx_ptr+self.max_station_slots]=seq_indices; obs_idx_ptr+=self.max_station_slots
                obs[obs_idx_ptr]=len(path_obj.metros)/(MAX_METROS_PER_PATH+1e-6); obs_idx_ptr+=1 # num_metros_on_path_norm
                obs[obs_idx_ptr]=max_pressure_on_path; obs_idx_ptr+=1 # max_station_pressure_on_path_norm
            else: obs_idx_ptr += (self.path_feature_size - 1)

        # --- 4. 全局特征 ---
        obs[obs_idx_ptr]=len(self.mediator.paths)/(self.max_paths+1e-6); obs_idx_ptr+=1
        avg_len=sum(len(p.stations) for p in self.mediator.paths)/(len(self.mediator.paths)+1e-6) if self.mediator.paths else 0
        obs[obs_idx_ptr]=avg_len/(self.max_station_slots+1e-6); obs_idx_ptr+=1
        obs[obs_idx_ptr]=self.mediator.score/(self.mediator.steps+1e-5) if self.mediator.steps>0 else 0.0; obs_idx_ptr+=1
        tot_wait=sum(len(s.passengers) for s in self.mediator.stations)
        max_sys_pass=self.max_station_slots*_station_max_cap_obs+1e-6
        obs[obs_idx_ptr]=tot_wait/max_sys_pass if max_sys_pass>0 else 0.0; obs_idx_ptr+=1
        obs[obs_idx_ptr]=len(self.mediator.metros)/(self.total_initial_metros+1e-6); obs_idx_ptr+=1 # metros_in_pool_norm
        
        if obs_idx_ptr != self.total_obs_size: logger.critical(f"OBS MISMATCH! Exp {self.total_obs_size}, Got {obs_idx_ptr}.") # simplified log
        return np.clip(obs, -1.0, 1.0)

    def _interpret_action(self, action_flat: int) -> Tuple[str, Optional[int], Optional[int]]:
        """Converts flat action index to (action_type, arg1, arg2). arg1=sA/path_idx, arg2=sB."""
        if action_flat == self.action_idx_no_op: return "NO_OP", None, None
        
        if self.action_idx_add_edge_start <= action_flat <= self.action_idx_add_edge_end:
            pair_flat_idx = action_flat - self.action_idx_add_edge_start
            sA, sB = self._sA_sB_pair_map[pair_flat_idx]
            return "ADD_EDGE", sA, sB
            
        if self.action_idx_split_path_start <= action_flat <= self.action_idx_split_path_end:
            pair_flat_idx = action_flat - self.action_idx_split_path_start
            sA, sB = self._sA_sB_pair_map[pair_flat_idx]
            return "SPLIT_PATH_VIA_EDGE", sA, sB # sA, sB define the segment to break

        if self.action_idx_add_metro_start <= action_flat <= self.action_idx_add_metro_end:
            path_idx = action_flat - self.action_idx_add_metro_start
            return "ADD_METRO", path_idx, None

        if self.action_idx_recall_metro_start <= action_flat <= self.action_idx_recall_metro_end:
            path_idx = action_flat - self.action_idx_recall_metro_start
            return "RECALL_METRO", path_idx, None
            
        logger.error(f"Unknown flat action index: {action_flat}"); return "INVALID_ACTION", None, None

    def _try_recreate_path_safely(self, stations_to_recreate: List[Station], original_is_loop: bool = False) -> bool:
        """Helper to attempt to recreate a path if primary action fails. Assumes metros are handled separately."""
        if len(self.mediator.paths) >= self.max_paths: return False
        if len(stations_to_recreate) < 2: return False

        valid_current_stations = []
        for s_orig in stations_to_recreate:
            # Check if this station object still exists in the mediator's current list
            found_in_mediator = False
            for s_mediator in self.mediator.stations:
                if s_mediator == s_orig: # Compare by object reference
                    valid_current_stations.append(s_mediator)
                    found_in_mediator = True
                    break
            if not found_in_mediator:
                logger.warning(f"  _try_recreate_path_safely: Station {s_orig.id if hasattr(s_orig, 'id') else 'N/A'} no longer valid.")
                return False # Cannot recreate if a station is gone

        logger.debug(f"  Attempting to recreate path with {len(valid_current_stations)} stations.")
        self.mediator.start_path_on_station(valid_current_stations[0])
        for s_inter in valid_current_stations[1:]:
            self.mediator.add_station_to_path(s_inter)
        
        # Loop handling: Mediator's add_station_to_path handles basic loop creation if last==first.
        # If original_is_loop was true due to more complex reasons, that's harder to restore simply.
        # For now, rely on Mediator's automatic loop detection.
        
        path_obj_ref = self.mediator.path_being_created
        path_count_before = len(self.mediator.paths)
        self.mediator.finish_path_creation()
        
        if (path_obj_ref and path_obj_ref in self.mediator.paths) or (len(self.mediator.paths) > path_count_before):
            l# ogger.info(f"  Successfully recreated a path with {len(valid_current_stations)} stations.")
            # Note: This recreated path will get a new metro from pool if available.
            return True
        logger.warning(f"  Failed to finalize recreated path.")
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None): # Standard reset
        super().reset(seed=seed)
        if seed is None and self.is_eval_env: seed = 42
        if seed is not None: self.mediator.seed = seed; random.seed(seed); np.random.seed(seed)
        self.mediator.reset_progress(); self.last_score = 0; self.current_rl_step = 0
        initial_dt_ms = 1000 // (self.metadata['render_fps'] or 30)
        for _ in range(30): # More robust initial station spawning
            if len(self.mediator.stations) >= min(3, self.max_station_slots): break
            self.mediator.increment_time(initial_dt_ms)
        if not self.mediator.stations and self.max_station_slots > 0 : # Ensure at least one if possible
             self.mediator.increment_time(initial_dt_ms*5) # Try a bit longer
        if len(self.mediator.stations) > 0: self.mediator.increment_time(1) # Trigger passenger logic
        #logger.info(f"Env reset. Initial stations: {len(self.mediator.stations)}")
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        logger.debug(f"Step {self.current_rl_step}, Raw Action: {action}")
        reward_action_logic = 0.0
        terminated = False; truncated = False # Truncation removed earlier
        invalid_action_penalty = -0.1 # Consistent penalty

        action_type, arg1, arg2 = self._interpret_action(action) # arg1=sA/path_idx, arg2=sB
        logger.debug(f"Interpreted Action: {action_type}, Arg1: {arg1}, Arg2: {arg2}")
        action_executed_successfully = False

        if action_type == "NO_OP":
            action_executed_successfully = True
        
        elif action_type == "ADD_EDGE":
            sA_idx, sB_idx = arg1, arg2
            sA_obj = self._get_station_by_slot_idx(sA_idx) # type: ignore
            sB_obj = self._get_station_by_slot_idx(sB_idx) # type: ignore

            if sA_obj and sB_obj and sA_obj != sB_obj:
                path_to_extend_at_A_tail = None
                # Try to extend if sA is a tail and sB is not on that path
                for p_idx, p in enumerate(self.mediator.paths):
                    if p.stations and p.stations[-1] == sA_obj and sB_obj not in p.stations:
                        path_to_extend_at_A_tail = p
                        logger.debug(f"  ADD_EDGE: Found path (idx {p_idx}) ending at S{sA_idx} to extend with S{sB_idx}.")
                        break
                
                if path_to_extend_at_A_tail:
                    original_stations = list(path_to_extend_at_A_tail.stations)
                    self.mediator.cancel_path(path_to_extend_at_A_tail)
                    
                    current_valid_stations_rebuild = [s for s_orig in original_stations if (s:=self._get_station_by_slot_idx(self.mediator.stations.index(s_orig)) if s_orig in self.mediator.stations else None)]
                    if len(current_valid_stations_rebuild) == len(original_stations):
                        self.mediator.start_path_on_station(current_valid_stations_rebuild[0])
                        for s_inter in current_valid_stations_rebuild[1:]: self.mediator.add_station_to_path(s_inter)
                        self.mediator.add_station_to_path(sB_obj) # Add the new station
                        
                        path_obj_ref = self.mediator.path_being_created
                        pc_before = len(self.mediator.paths)
                        self.mediator.finish_path_creation()
                        if (path_obj_ref and path_obj_ref in self.mediator.paths) or len(self.mediator.paths) > pc_before:
                            action_executed_successfully = True; reward_action_logic += 0.03
                            # logger.info(f"  Successfully extended path to S{sB_idx}.")
                        else:
                            logger.warning("  ADD_EDGE: Failed to finalize extended path. Attempting to restore original."); reward_action_logic += invalid_action_penalty
                            self._try_recreate_path_safely(original_stations) # Restore original (metros handled by recreate)
                    else: logger.warning("  ADD_EDGE: Original stations for extension became invalid."); reward_action_logic += invalid_action_penalty
                
                else: # No path to extend, try creating a new 2-station path
                    if len(self.mediator.paths) < self.max_paths:
                        is_already_direct_link = any(len(p.stations)==2 and ((p.stations[0]==sA_obj and p.stations[1]==sB_obj)) for p in self.mediator.paths) # Basic check for sA->sB
                        if not is_already_direct_link:
                            pc_before = len(self.mediator.paths)
                            self.mediator.start_path_on_station(sA_obj); self.mediator.add_station_to_path(sB_obj)
                            path_obj_ref = self.mediator.path_being_created; self.mediator.finish_path_creation()
                            if (path_obj_ref and path_obj_ref in self.mediator.paths) or len(self.mediator.paths) > pc_before:
                                action_executed_successfully = True; reward_action_logic += 0.02
                                # logger.info(f"  Successfully created new 2-station path S{sA_idx}-S{sB_idx}.")
                            else: logger.debug("  ADD_EDGE: Failed to finalize new 2-station path."); reward_action_logic += invalid_action_penalty
                        else: logger.debug("  ADD_EDGE: Direct link S{sA_idx}-S{sB_idx} already exists or invalid."); reward_action_logic += invalid_action_penalty * 0.5
                    else: logger.debug("  ADD_EDGE: Max paths reached."); reward_action_logic += invalid_action_penalty
            else: logger.debug("  ADD_EDGE: Invalid stations or sA=sB."); reward_action_logic += invalid_action_penalty

        elif action_type == "SPLIT_PATH_VIA_EDGE":
            sA_idx, sB_idx = arg1, arg2
            sA_obj = self._get_station_by_slot_idx(sA_idx) #type:ignore
            sB_obj = self._get_station_by_slot_idx(sB_idx) #type:ignore

            if sA_obj and sB_obj and sA_obj != sB_obj:
                path_to_split = None
                segment_idx_in_path = -1 # Index of sA in the sA->sB segment
                
                for p_idx, p in enumerate(self.mediator.paths):
                    for i in range(len(p.stations) - 1):
                        if p.stations[i] == sA_obj and p.stations[i+1] == sB_obj:
                            path_to_split = p; segment_idx_in_path = i; break
                    if path_to_split: break
                
                if path_to_split:
                    logger.info(f"  SPLIT_PATH: Found path (ID {path_to_split.id if hasattr(path_to_split,'id') else 'N/A'}) with segment S{sA_idx}->S{sB_idx} at station index {segment_idx_in_path}.")
                    original_stations = list(path_to_split.stations)
                    # original_is_loop = path_to_split.is_looped # May not be preserved easily
                    
                    self.mediator.cancel_path(path_to_split) # Metros return to pool
                    paths_created_count = 0

                    # Create prefix path: stations up to and including sA_obj
                    prefix_stations_objs = original_stations[:segment_idx_in_path + 1]
                    if len(prefix_stations_objs) >= 2:
                        if self._try_recreate_path_safely(prefix_stations_objs): paths_created_count +=1
                        else: logger.warning("  SPLIT_PATH: Failed to recreate prefix path.")
                    
                    # Create suffix path: stations from sB_obj кризис
                    suffix_stations_objs = original_stations[segment_idx_in_path + 1:]
                    if len(suffix_stations_objs) >= 2:
                        if self._try_recreate_path_safely(suffix_stations_objs): paths_created_count +=1
                        else: logger.warning("  SPLIT_PATH: Failed to recreate suffix path.")
                    
                    if paths_created_count > 0: # If at least one part of split was successful
                        action_executed_successfully = True; reward_action_logic += 0.03
                        # logger.info(f"  SPLIT_PATH: Successfully created {paths_created_count} new path(s) from split.")
                    else: # Split resulted in no valid new paths (e.g., original was 2 stations)
                        reward_action_logic += invalid_action_penalty # Or no penalty if original was just 2 stations
                        logger.info("  SPLIT_PATH: Split did not result in any new valid paths (original might have been too short or stations disappeared).")
                        # The original path is already cancelled. This effectively deleted it.
                        action_executed_successfully = True # Action was "valid" in that it deleted the segment/path

                else: logger.debug(f"  SPLIT_PATH: No path found with direct segment S{sA_idx}->S{sB_idx}."); reward_action_logic += invalid_action_penalty
            else: logger.debug(f"  SPLIT_PATH: Invalid stations or sA=sB."); reward_action_logic += invalid_action_penalty

        elif action_type == "ADD_METRO":
            path_idx = arg1
            target_path = self._get_path_by_slot_idx(path_idx) #type:ignore
            if target_path and self.mediator.metros:
                if len(target_path.metros) < MAX_METROS_PER_PATH:
                    metro_to_add = self.mediator.metros.pop(0)
                    target_path.add_metro(metro_to_add)
                    action_executed_successfully = True; reward_action_logic += 0.02
                    logger.info(f"  ADD_METRO: Added metro to path {path_idx}. Path now has {len(target_path.metros)} metros. Pool: {len(self.mediator.metros)}")
                else: logger.debug(f"  ADD_METRO: Path {path_idx} already at max metros ({MAX_METROS_PER_PATH})."); reward_action_logic += invalid_action_penalty * 0.5
            else: logger.debug(f"  ADD_METRO: Invalid path_idx {path_idx} or no metros in pool."); reward_action_logic += invalid_action_penalty

        elif action_type == "RECALL_METRO":
            path_idx = arg1
            target_path = self._get_path_by_slot_idx(path_idx) #type:ignore
            if target_path and len(target_path.metros) > MIN_METROS_PER_PATH:
                metro_recalled = target_path.remove_metro() # Assumes Path.remove_metro() returns a Metro
                if metro_recalled:
                    self.mediator.metros.append(metro_recalled)
                    action_executed_successfully = True; reward_action_logic += 0.01 # Smaller reward/penalty
                    logger.info(f"  RECALL_METRO: Recalled metro from path {path_idx}. Path now has {len(target_path.metros)} metros. Pool: {len(self.mediator.metros)}")
                else: logger.warning(f"  RECALL_METRO: Path {path_idx} remove_metro failed."); reward_action_logic += invalid_action_penalty
            else: logger.debug(f"  RECALL_METRO: Invalid path_idx {path_idx} or path has min metros."); reward_action_logic += invalid_action_penalty
        
        elif action_type == "INVALID_ACTION":
            reward_action_logic += invalid_action_penalty * 2
            logger.error(f"  INVALID_ACTION type processed in step loop. Original flat action: {action}")

        # --- Game Simulation ---
        game_outcome_state = MeditatorState.RUNNING
        # ... (rest of game simulation, reward calculation, obs generation - largely same as your previous version) ...
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

        if terminated and game_outcome_state == MeditatorState.ENDED: reward -= 20.0 
        _smc = STATION_MAX_CAPACITY if STATION_MAX_CAPACITY > 0 else 10.0
        num_crowded = sum(1 for s in self.mediator.stations if len(s.passengers) / _smc > 0.85)
        reward -= num_crowded * 1.5 
        num_total_wait = sum(len(s.passengers) for s in self.mediator.stations)
        reward -= num_total_wait * 0.01 
        
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
                                 eval_freq=100000, # 根据训练速度调整
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
    run_edge_env_training() 
    # run_edge_env_training(load_model_path="./mini_metro_edge_rl_logs/ppo_mini_metro_edge.zip") # 示例：继续训练
    
    # 仅进行可视化评估
    # run_edge_env_visual_evaluation()
    run_edge_env_visual_evaluation(model_path="./mini_metro_edge_rl_logs/ppo_mini_metro_edge.zip")