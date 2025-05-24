# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

# start of code

import numpy as np

from typing import List, Tuple

from api import ProgressiveStationGame

from entity.station import Station
from geometry.utils import distance

from config import (
    station_padding,
    station_grid_size,
    screen_width,
    screen_height,
    station_capacity,
    num_paths
)

import math

from concurrent.futures import ThreadPoolExecutor

padding = station_padding
grid_nx, grid_ny = station_grid_size
grid_dx = int((screen_width - 2*padding) / grid_nx)
grid_dy = int((screen_height - 2*padding) / grid_ny)

mean_min_dist = math.sqrt(grid_dx * grid_dy)

class Gene:
    def __init__(self):
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


            # # update
            # "U_is_loop": 0,
            # "U_avr_path_length": 0,
            # "U_avr_wait_time": 0,
            # "U_avr_pass_density": 0,
            # "U_total_station_types": 0,


            # # delete
            # "D_overlap_factor": 0,
            # "D_avr_path_length": 0,
            # "D_total_station_types": 0,
            # "D_avr_wait_time": 0,
        }

        self.com_cache = None

        self.randomize_weights()

    def randomize_weights(self):
        for key in self.weights.keys():
            self.weights[key] = np.random.uniform(-1, 1)

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

        return score
    
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

        return score
    
    
    def score_connection(self, sta1: Station, sta2: Station):
        score = 0

        dist = distance(sta1.position, sta2.position) / 100.0  # normalize roughly
        score -= self.weights["distance"] * dist

        passenger_density = (len(sta1.passengers) + len(sta2.passengers)) / 20.0  # normalize
        score += self.weights["passenger_density"] * passenger_density

        diversity = 1 if sta1.shape.type != sta2.shape.type else 0
        score += self.weights["station_type"] * diversity

        return score
    
    def evaluate_redesign_action(self, act, game, paths_config):
        path_index, (new_path, is_loop) = act
        old_path = paths_config[path_index][0]

        coverage_gain = len(set(new_path) - set(old_path))
        congestion_reduction = self.estimate_congestion_reduction(path_index, new_path)
        average_distance = np.mean([
            distance(game.stations[a].position, game.stations[b].position)
            for a, b in zip(new_path[:-1], new_path[1:])
        ]) / 100.0  # normalize
        type_diversity = len(set([game.stations[i].shape.type for i in new_path]))

        score = 0
        score += self.weights["coverage_gain"] * coverage_gain
        # score += w["congestion_reduction"] * congestion_reduction
        score -= self.weights["average_distance"] * average_distance
        score += self.weights["station_type_diversity"] * type_diversity
        score += self.weights["reuse_existing_path"] * len(set(new_path) & set(old_path))
        # score += self.weights["is_loop"] * is_loop

        return score
    
    def estimate_congestion_reduction(self, path_index, new_path):
        # 用乘客數或壅塞站數估計，也可以回傳 random for now
        # return np.random.uniform(0, 5)
        return 0

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

    def calc_fitness(self):
        # alpha-beta pruning-like
        def simulate_game(early_stopping: int):
            try:
                game = Game(self)
                return game.play(early_stopping)
            except:
                return None
        
        pseudo_min = simulate_game(early_stopping=500)
        while pseudo_min is None:
            pseudo_min = simulate_game(early_stopping=500)

        with ThreadPoolExecutor(max_workers=8) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=pseudo_min), range(8)))

        scores = [s for s in scores if s is not None]
        self.fitness = min(min(scores), pseudo_min)

    def test(self):
        def simulate_game(early_stopping: int):
            try:
                game = Game(self)
                return game.play(early_stopping)
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=1E9), range(100)))

        print(scores)

class PathConfig:
    def __init__(self, config: Tuple[List[int], bool], in_game_idx: int):
        self.stations = config[0]
        self.is_loop = config[1]
        self.ingame_idx = in_game_idx

    def bind_ingame_idx(self, new_idx: int):
        self.ingame_idx = new_idx

    @property
    def config(self):
        return (self.stations, self.is_loop)

class Game:
    def __init__(self, player: Creature):
        self.player = player
        self.actions = []

        gamespeed = 31.25 # 200 # 31.25
        in_game_break_time = (0.5/gamespeed) * 1000 # stop periodic yield
        self.game = ProgressiveStationGame(gamespeed=gamespeed, yield_interval_ms=in_game_break_time, visuals=False)

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
        
        # self.get_redesign_actions()

        best_connect, connect_score = self.get_best_connect_actions()
        # best_redesign, redesign_score = self.get_best_redesign_action()
        
        if best_connect is None: # and best_redesign is None:
            return
        
        # print(best_connect)
        self.apply_connect_action(best_connect)
        # self.apply_redesign_action(best_redesign

        # if redesign_score > connect_score:
        #     self.apply_redesign_action(best_redesign)
        # else:
        #     self.apply_connect_action(best_connect)

    def get_connect_actions(self) -> Tuple[str, Tuple[int, int]]:
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
                    self.actions.append(('C', (s1_ind, s2_ind)))

    def get_best_connect_actions(self) -> Tuple[any, float]:
        best_act_score = -float("inf")
        best_act = None
        
        self.player.com_cache = None
        for act in self.actions:
            if act[0] != "C":
                continue

            s1_ind, s2_ind = act[1]
            s1 = self.game.stations[s1_ind]
            s2 = self.game.stations[s2_ind]

            score = self.player.score_C_connect(s1, s2, self.game.stations)

            if score > best_act_score:
                best_act_score = score
                best_act = act

        if best_act is None:
            return None, best_act_score

        return best_act[1], best_act_score
    
    def apply_connect_action(self, act: Tuple[int, int]):
        s1_ind, s2_ind = act
        use_existing_path, self_path_idx, disjoint_station, insert_at = \
            self.should_use_existing_path(s1_ind, s2_ind)

        if use_existing_path:
            # old_path = self.paths[self_path_idx].stations.copy()
            self.paths[self_path_idx].stations.insert(insert_at, disjoint_station)

            new_ingame_idx = self.game.recreate_path(
                self.paths[self_path_idx].ingame_idx,
                self.paths[self_path_idx].config
            )

            self.paths[self_path_idx].bind_ingame_idx(new_ingame_idx)
            # print("reused", old_path, self.paths[self_path_idx].stations)
        else:
            # print("created", s1_ind, s2_ind)
            new_path = ([s1_ind, s2_ind], False)
            ingame_idx = self.game.create_path(new_path)
            self.paths.append(PathConfig(new_path, ingame_idx))

    def should_use_existing_path(self, s1_ind: int, s2_ind: int):
        max_score = -float("inf")
        
        # use_existing_path, self_path_idx, disjoint_station, insert_at
        best_choice = (False, None, None, None)

        for ind, path in enumerate(self.paths):
            stations = path.stations

            if s1_ind in stations:
                s1_loc = stations.index(s1_ind)
                is_outer = s1_loc in [0, len(stations) - 1]
                
                if not is_outer:
                    continue
                
                insert_ind = s1_loc
                if s1_loc == len(stations) - 1:
                    insert_ind += 1
                
                reuse_score = self.player.score_C_reuse(
                    self.game.paths[self.paths[ind].ingame_idx].stations
                )
                if reuse_score > max_score:
                    max_score = reuse_score
                    best_choice = (True, ind, s2_ind, insert_ind)
            
            elif s2_ind in stations:
                s2_loc = stations.index(s2_ind)
                is_outer = s2_loc in [0, len(stations) - 1]
                
                if not is_outer:
                    continue
                
                insert_ind = s2_loc
                if s2_loc == len(stations) - 1:
                    insert_ind += 1
                
                reuse_score = self.player.score_C_reuse(
                    self.game.paths[self.paths[ind].ingame_idx].stations
                )
                if reuse_score > max_score:
                    max_score = reuse_score
                    best_choice = (True, ind, s1_ind, insert_ind)

        return best_choice
    
    def get_redesign_actions(self):
        self.redesign_acts = []

        all_station_set = set(range(len(self.game.stations)))

        for path_index, (path, is_loop) in enumerate(self.paths):
            if len(path) < 3:
                continue  # 太短不重繪
            
            disjoint_stations = list(all_station_set - set(path))

            # generate path under del op
            for sta in path:
                new_path = list(path)
                new_path.remove(sta)

                if len(new_path) == len(set(new_path)):
                    self.redesign_acts.append((path_index, (new_path, is_loop)))

            # generate path under add op
            for sta in disjoint_stations:
                new_path = list(path)
                new_path.append(sta)

                if len(new_path) == len(set(new_path)):
                    self.redesign_acts.append((path_index, (new_path, is_loop)))

            # generate path under replace op
            for sta in disjoint_stations:
                for i in range(len(path)):
                    new_path = list(path)
                    new_path[i] = sta

                    if len(new_path) == len(set(new_path)):
                        self.redesign_acts.append((path_index, (new_path, is_loop)))

            # generate path under shuffle op
            new_path = list(path)
            np.random.shuffle(new_path)

            if len(new_path) == len(set(new_path)):
                self.redesign_acts.append((path_index, (new_path, is_loop)))

            # toggle path loop
            # new_path = list(path)

            # if len(new_path) == len(set(new_path)):
            #     self.redesign_acts.append((path_index, (new_path, not is_loop)))

    def get_best_redesign_action(self):
        best_score = -float("inf")
        best_act = None
        for act in self.redesign_acts:
            score = self.player.evaluate_redesign_action(act, self.game, self.paths)
            if score > best_score:
                best_score = score
                best_act = act
        return best_act, best_score  # 補上 score 讓 act() 可以比較

    def apply_redesign_action(self, act):
        path_index, new_path_config = act
        self.paths[path_index] = new_path_config
        
        try:
            self.game.recreate_path(path_index, new_path_config)
        except:
            print(self.paths)
            print(act)



