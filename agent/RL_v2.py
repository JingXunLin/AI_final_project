# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import List, Dict, Tuple, Optional, Any # Added Any
from enum import Enum, auto

# 游戏特定导入 (根据你的项目结构调整路径)
from config import (
    num_metros, num_paths, num_stations_max, station_spawning_interval_step,
    framerate, screen_color, screen_width, screen_height,
    station_grid_size, station_capacity, 
)
# 以下常量至关重要，假设在 config.py 中定义:
NUM_HORIZONTAL_GRIDS, NUM_VERTICAL_GRIDS = station_grid_size
STATION_MAX_CAPACITY = station_capacity # 或从 Station/Metro 类获取 (如果是静态成员)
from geometry.type import ShapeType # 假设 ShapeType 是一个枚举
from mediator import Mediator, MeditatorState # 游戏逻辑核心
from visuals.background import draw_waves # 如果环境直接用于渲染背景波浪

# Stable Baselines3 用于训练和评估
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback #, StopTrainingOnRewardThreshold

class HighLevelAction(Enum):
    NO_OP = auto()
    BUILD_NEW_PATH_SMART = auto()
    EXTEND_EXISTING_PATH_SMART = auto()
    DELETE_LEAST_USEFUL_PATH = auto()

class MiniMetroSimpleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': framerate or 30}

    def __init__(self, gamespeed: int = 1, visuals: bool = False, is_eval_env: bool = False):
        super().__init__()

        if not pygame.font.get_init():
            pygame.font.init()
        if visuals and not pygame.get_init():
             pygame.init()

        self.mediator = Mediator(gamespeed=gamespeed, gen_stations_first=False)
        
        self.max_station_slots = num_stations_max # Max observable station slots
        self.max_paths = num_paths # Max allowed paths in game, also used as max observable paths
        self.shape_types_enum: List[ShapeType] = list(ShapeType)
        self.num_shape_types = len(self.shape_types_enum)

        self.visuals = visuals
        self.is_eval_env = is_eval_env
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        if self.visuals:
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
        
        if seed is None and self.is_eval_env:
            seed = 42 
        
        if seed is not None:
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

        for _ in range(self.game_ticks_per_rl_step):
            if self.visuals and self.screen:
                for pygame_event in pygame.event.get():
                    if pygame_event.type == pygame.QUIT:
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

        observation = self._get_obs()
        info = self._get_info()

        if self.visuals:
            self.render()
            
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


    print(f"模型策略网络结构: {model.policy}")
    print(f"观测空间维度: {train_env.observation_space.shape}")
    print(f"动作空间大小: {train_env.action_space.n}")

    total_timesteps_to_train = 200000 # Adjusted total timesteps
    print(f"开始训练，总步数: {total_timesteps_to_train}...")
    try:
        model.learn(total_timesteps=total_timesteps_to_train, 
                    callback=eval_callback,
                    progress_bar=True)
        model.save(log_dir + model_save_name)
        print(f"训练完成。最终模型已保存到 {log_dir + model_save_name}.zip")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
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
    essential_configs_defined = True
    try:
        _ = NUM_HORIZONTAL_GRIDS
        _ = ShapeType
        _ = STATION_MAX_CAPACITY
        _ = Mediator # Check if Mediator class is available
    except NameError as e:
        essential_configs_defined = False
        print(f"错误：一些必要的配置或类未定义: {e}")
        print("请确保 'config.py' 文件存在于Python路径中，并且已正确定义这些变量，且 Mediator 类已导入。")
        print("脚本将中止。")
    
    if essential_configs_defined:
        run_training(load=True)  # Set load=True to continue training or load a model
        evaluation_visual()