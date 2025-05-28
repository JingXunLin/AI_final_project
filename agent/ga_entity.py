# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

# start of code

import numpy as np

from typing import List, Tuple

from api import ProgressiveStationGame

from entity.station import Station
from entity.path import Path
from geometry.utils import distance

from config import (
    station_padding,
    station_grid_size,
    screen_width,
    screen_height,
    station_capacity,
    num_paths,
    metro_speed_per_ms,
    framerate
)

import math

from itertools import islice, permutations

from concurrent.futures import ThreadPoolExecutor

padding = station_padding
grid_nx, grid_ny = station_grid_size
grid_dx = int((screen_width - 2*padding) / grid_nx)
grid_dy = int((screen_height - 2*padding) / grid_ny)

mean_min_dist = math.sqrt(grid_dx * grid_dy)

PathConfigContent = Tuple[List[int], bool]

class PathConfig:
    def __init__(self, config: PathConfigContent, in_game_idx: int):
        self.stations, self.is_loop = config
        self.ingame_idx = in_game_idx

    def bind_ingame_idx(self, new_idx: int):
        self.ingame_idx = new_idx

    @property
    def config(self):
        return (self.stations, self.is_loop)
    
    def set_config(self, config: PathConfigContent):
        self.stations, self.is_loop = config


CreateOp = Tuple[int, int]
UpdateOp = List[
    Tuple[int, PathConfigContent] # path_ind, new_path_config
]

class Operation:
    def __init__(self, type: str, op, info: str=None):
        assert type in ["C", "U"]

        self.type = type
        self.info = info
        self.op = op

        # self.realize_op = None
        self.score = 0

    @property
    def basic_info(self):
        return f"do({self.type}), i({self.info}), s({self.score:.4f}):"


class Gene:
    def __init__(self):
        self.seed = int.from_bytes(os.urandom(8), 'big')
        self.rng = np.random.default_rng(self.seed)

        # C(R)UD
        self.weights = {
            # create
            ## connect factors
            "C_dist": 0,
            "C_pass_density": 0,
            "C_station_type": 0,
            "C_dist_to_com": 0,

            ## reuse factors
            "C_avr_path_length": 0,
            "C_avr_pass_density": 0,
            "C_total_station_types": 0,


            # update
            # "U_avr_path_length": 0,
            "U_wait_time_mean": 0,
            "U_wait_time_std": 0,
            "U_total_station_types": 0,
            "U_has_isolated_stations": 0,
            "U_station_cnt": 0,
            # "U_overlap": 0,
        }

        # cache global attributes in 1 period
        self.com_cache = None # Point(center of mass)
        self.iso_cache = None # list of isolated stations

        self.randomize_weights()

    def randomize_weights(self):
        for key in self.weights.keys():
            self.weights[key] = self.rng.uniform(-1, 1)

    def score_C_connect(self, s1: Station, s2: Station, stations: List[Station]):
        score = 0

        dist = distance(s1.position, s2.position) / mean_min_dist
        score += self.weights["C_dist"] * dist

        passenger_density = (len(s1.passengers) + len(s2.passengers)) / station_capacity
        score += self.weights["C_pass_density"] * passenger_density

        type = 1 if s1.shape.type != s2.shape.type else 0
        score += self.weights["C_station_type"] * type

        if self.com_cache is None:
            stations_pos = [s.position for s in stations]
            self.com_cache = sum(stations_pos) * (1 / len(stations_pos))
        dist_to_com = distance((s1.position + s2.position) * (1/2), self.com_cache) / mean_min_dist
        score += self.weights["C_dist_to_com"] * dist_to_com

        return float(score)
    
    def score_C_reuse(self, stations: List[Station]):
        score = 0

        total_path_len = sum([distance(a.position, b.position) for a, b in zip(stations[:-1], stations[1:])])
        avr_path_len = total_path_len / len(stations) / mean_min_dist
        score += self.weights["C_avr_path_length"] * avr_path_len

        total_pass_density = sum([
            len(sta.passengers) / station_capacity
            for sta in stations]
        )
        avr_pass_density = total_pass_density / len(stations)
        score += self.weights["C_avr_pass_density"] * avr_pass_density

        station_types = len(set([sta.shape.type for sta in stations]))
        score += self.weights["C_total_station_types"] * station_types

        return float(score)
    
    def calc_wait_time_statistics(self, assumed_path: List[Station], is_loop: bool):
        """
        return mean, std
        """
        dist_between_stations = [
            distance(a.position, b.position)
            for a, b in zip(assumed_path[:-1], assumed_path[1:])
        ]

        # calculate wait time mean and std, velocity = metro_speed_per_ms * 1000 (to px/s)
        if is_loop:
            dist_between_stations.append(distance(assumed_path[-1].position, assumed_path[0].position))
            wait_time_mean = sum(dist_between_stations) / (metro_speed_per_ms * 1000) / mean_min_dist
            return wait_time_mean, 0
        
        wait_times = []

        # outer stations
        whole_wait_time = sum(dist_between_stations) / (metro_speed_per_ms * 1000) / mean_min_dist
        wait_times.extend([whole_wait_time] * 2)

        # middle stations
        for sta_ind in range(1, len(assumed_path) - 2):
            left_dist = 2 * sum(dist_between_stations[:sta_ind])
            right_dist = 2 * sum(dist_between_stations[sta_ind + 1:])

            left_wait_time = left_dist / (metro_speed_per_ms * 1000) / mean_min_dist
            right_wait_time = right_dist / (metro_speed_per_ms * 1000) / mean_min_dist

            wait_times.extend([left_wait_time / 2, right_wait_time / 2])

        wait_times = np.array(wait_times)
        return float(np.mean(wait_times)), float(np.std(wait_times))


    def score_U(self, path_config: PathConfigContent, game: ProgressiveStationGame, paths: List[PathConfig]):
        stations, is_loop = path_config
        virtual_game_stations = [game.stations[sta] for sta in stations]

        score = 0

        # avr_path_length
        if "U_avr_path_length" in self.weights:
            total_path_len = sum([distance(a.position, b.position) for a, b in zip(virtual_game_stations[:-1], virtual_game_stations[1:])])
        
            if is_loop:
                total_path_len += distance(virtual_game_stations[-1].position, virtual_game_stations[0].position)
        
            avr_path_len = total_path_len / len(virtual_game_stations) / mean_min_dist
            score += self.weights["U_avr_path_length"] * avr_path_len

        if "U_wait_time_mean" in self.weights and "U_wait_time_std" in self.weights:
            wait_time_mean, wait_time_std = self.calc_wait_time_statistics(virtual_game_stations, is_loop)
            score += self.weights["U_wait_time_mean"] * wait_time_mean
            score += self.weights["U_wait_time_std"] * wait_time_std

        # total_station_types
        if "U_total_station_types" in self.weights:
            station_types = len(set([sta.shape.type for sta in virtual_game_stations]))
            score += self.weights["U_total_station_types"] * station_types

        if "U_has_isolated_stations" in self.weights:
            contains_isolated_stations = 1 if any([sta in self.iso_cache for sta in stations]) else 0
            score += self.weights["U_has_isolated_stations"] * contains_isolated_stations

        if "U_station_cnt" in self.weights:
            score += self.weights["U_station_cnt"] * len(stations)

        # if "U_overlap" in self.weights:
        #     max_overlap_cnt = 0

        #     self_stas = set(stations)
        #     for path_conf in paths:
        #         if stations == path_conf.stations:
        #             continue
        #         other_stas = set(path_conf.stations)
        #         max_overlap_cnt = max(max_overlap_cnt, len(self_stas & other_stas))
            
        #     score += self.weights['U_overlap'] * (max_overlap_cnt / len(self_stas))

        return float(score)

class Creature(Gene):
    def __init__(self, need_calc_fitness: bool = False):
        super().__init__()

        self.fitness = None
        self.need_recalc_fitness = True

        if need_calc_fitness:
            self.try_calc_fitness()

    def try_calc_fitness(self):
        if not self.need_recalc_fitness:
            return
        
        self.calc_fitness()
        self.need_recalc_fitness = False
        self.is_test = False

    def calc_fitness(self):
        # alpha-beta pruning-like
        def simulate_game(early_stopping: int):
            try:
                game = Game(self)
                
                score = game.play(early_stopping)
                # print(score)
                return score
            except:
                print("err!")
                return None
        
        pseudo_min = simulate_game(early_stopping=1000)
        while pseudo_min is None:
            pseudo_min = simulate_game(early_stopping=1000)

        with ThreadPoolExecutor(max_workers=9) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=pseudo_min), range(9)))

        scores = [s for s in scores if s is not None]
        self.fitness = min(min(scores), pseudo_min)

    def statistics(self):
        def simulate_game(early_stopping: int):
            try:
                game = Game(self)
                return game.play(early_stopping)
            except:
                print("err!")
                return None
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=3E3), range(100)))

        print(scores)

    def single(self):
        game = Game(self, self.is_test)
        return game.play(early_stopping=1E9)

class Game:
    def __init__(self, player: Creature, is_test: bool = False):
        self.player = player
        self.is_test = is_test

        self.actions: List[Operation] = []
        
        if self.is_test:
            self.gamespeed = 31.25
            self.visuals = True
        else:
            self.gamespeed = 100
            self.visuals = False
        
        in_game_break_time = (0.5/self.gamespeed) * 1000 # stop periodic yield

        self.game = ProgressiveStationGame(
            gamespeed=self.gamespeed, yield_interval_ms=in_game_break_time, visuals=self.visuals)

        self.paths: List[PathConfig] = []

    def play(self, early_stopping: int=1000) -> int:
        simulation = self.game.run()
        try:
            while True:
                if self.game.mediator.score > early_stopping:
                    return self.game.mediator.score
                
                next(simulation)
                self.act()
        except StopIteration as score:
            return score.value

    def act(self):
        if len(self.game.stations) < 2:
            return

        self.actions = []

        if len(self.game.paths) < num_paths:
            self.get_connect_actions()

        best_action: Operation = self.get_best_connect_action()
        if best_action is not None:
            if self.is_test:
                print(best_action.basic_info)
                print("   ", best_action.op)
                
            self.apply_connect_action(best_action.op)
        

        self.actions = []

        if len(self.paths) >= 1:
            self.get_update_actions()

        best_action = self.get_best_update_action()
        if best_action is not None:
            if self.is_test:
                print(best_action.basic_info)
                for act in best_action.op:
                    print("   ", self.paths[act[0]].config, "->", act[1])
            
            self.apply_update_action(best_action.op)


    def update_before_delete_path(self, old_ingame_idx: int):
        for path in self.paths:
            if path.ingame_idx > old_ingame_idx:
                path.ingame_idx -= 1


    def get_connect_actions(self):
        stations = range(len(self.game.stations))

        for s1_ind in stations:
            for s2_ind in stations:
                if s1_ind == s2_ind:
                    continue

                is_connected = any([
                    s1_ind in path.stations and s2_ind in path.stations
                    for path in self.paths
                ])
                if not is_connected:
                    # self.actions.append(('C', (s1_ind, s2_ind)))
                    self.actions.append(Operation("C", (s1_ind, s2_ind)))

    def get_best_connect_action(self) -> Operation:
        best_act_score = -float("inf")
        best_act = None
        
        self.player.com_cache = None
        for act in self.actions:
            if act.type != "C":
                continue

            s1_ind, s2_ind = act.op
            s1 = self.game.stations[s1_ind]
            s2 = self.game.stations[s2_ind]

            act.score = self.player.score_C_connect(s1, s2, self.game.stations)

            if act.score > best_act_score:
                best_act_score = act.score
                best_act = act

        # if best_act_score < 0:
        #     return None

        return best_act
    
    def apply_connect_action(self, op: CreateOp):
        s1_ind, s2_ind = op
        use_existing_path, self_path_idx, disjoint_station, insert_at = \
            self.should_use_existing_path(s1_ind, s2_ind)

        if use_existing_path:
            self.update_before_delete_path(self.paths[self_path_idx].ingame_idx)
            self.paths[self_path_idx].stations.insert(insert_at, disjoint_station)

            new_ingame_idx = self.game.recreate_path(
                self.paths[self_path_idx].ingame_idx,
                self.paths[self_path_idx].config
            )

            self.paths[self_path_idx].bind_ingame_idx(new_ingame_idx)
        else:
            new_config = ([s1_ind, s2_ind], False)
            ingame_idx = self.game.create_path(new_config)
            self.paths.append(PathConfig(new_config, ingame_idx))

    def should_use_existing_path(self, s1_ind: int, s2_ind: int):
        def try_station(target_idx, other_idx, path_stations, ingame_stations):
            if target_idx not in path_stations:
                return None

            loc = path_stations.index(target_idx)
            if loc not in (0, len(path_stations) - 1):
                return None

            insert_at = loc + 1 if loc == len(path_stations) - 1 else loc
            score = self.player.score_C_reuse(ingame_stations)
            return (score, (True, path_index, other_idx, insert_at))

        max_choice = (-float("inf"), (False, None, None, None))

        for path_index, path in enumerate(self.paths):
            path_stations = path.stations
            ingame_stations = self.game.paths[path.ingame_idx].stations

            for target, other in [(s1_ind, s2_ind), (s2_ind, s1_ind)]:
                result = try_station(target, other, path_stations, ingame_stations)
                if result and result[0] > max_choice[0]:
                    max_choice = result

        if max_choice[0] < 0:
            return False, None, None, None

        return max_choice[1]

    def get_update_actions(self):
        """
        update operations:
            X - delete sta from path
            X - replace sta in path
            - add isolated sta to path
            X swap sta in another path
            - shuffle path order
            - toggle path loop (affects various indexes)

        action format: ("U", (path_ind, updated_path_config))
        """

        all_station_set = set(range(len(self.game.stations)))
        all_connected_stations = set()

        for path in self.paths:
            all_connected_stations.update(path.stations)

        self.player.iso_cache = list(all_station_set - all_connected_stations)

        # add isolated sta to path
        for ind, path in enumerate(self.paths):
            for sta in self.player.iso_cache:
                new_config = ([*path.stations, sta], False)

                new_op: UpdateOp = [(ind, new_config)]
                self.actions.append(Operation("U", new_op, 'iso'))
        
        # shuffle path order
        for ind, path in enumerate(self.paths):
            if len(path.stations) <= 2:
                continue
            
            for new_path in self.generate_partially_reversed_lists(path.stations):
                new_path = list(new_path)

                if new_path == path.stations \
                    or new_path == path.stations[::-1]:
                    continue
                
                new_config = (new_path, path.is_loop)

                new_op: UpdateOp = [(ind, new_config)]
                self.actions.append(Operation("U", new_op, 'shuf'))

        # toggle path loop
        for ind, path in enumerate(self.paths):
            if len(path.stations) <= 3:
                continue

            new_config = (path.stations, not path.is_loop)
            
            new_op: UpdateOp = [(ind, new_config)]
            self.actions.append(Operation("U", new_op, 'togg'))

        

    def get_all_reverse_operations(self, lst, max_count=50):
        return list(islice(permutations(lst), max_count))
    
    def generate_partially_reversed_lists(self, lst):
        n = len(lst)
        results = []

        # 從長度為 2 到 n-1 的子序列進行反轉
        for length in range(2, n):
            for start in range(n - length + 1):
                end = start + length
                # 將 [start:end] 部分反轉
                new_lst = lst[:start] + lst[start:end][::-1] + lst[end:]
                results.append(new_lst)

        return results

    def get_best_update_action(self):
        best_delta = 0
        best_act = None

        for act in self.actions:
            if act.type != "U":
                continue
            
            old_total_score = 0
            for op in act.op:
                path_ind, _ = op
                old_path_config = self.paths[path_ind].config
                old_total_score += self.player.score_U(old_path_config, self.game, self.paths)
            
            new_total_score = 0
            for op in act.op:
                path_ind, new_path_config = op
                new_total_score += self.player.score_U(new_path_config, self.game, self.paths)

            act.score = new_total_score - old_total_score

            if act.score > best_delta:
                best_delta = act.score
                best_act = act

        return best_act

    def apply_update_action(self, op: UpdateOp):
        for path_ind, path_config in op:
            self.update_before_delete_path(self.paths[path_ind].ingame_idx)
            self.paths[path_ind].set_config(path_config)

            new_ingame_idx = self.game.recreate_path(
                self.paths[path_ind].ingame_idx,
                self.paths[path_ind].config
            )

            self.paths[path_ind].bind_ingame_idx(new_ingame_idx)
