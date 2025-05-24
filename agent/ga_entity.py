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

from concurrent.futures import ThreadPoolExecutor

padding = station_padding
grid_nx, grid_ny = station_grid_size
grid_dx = int((screen_width - 2*padding) / grid_nx)
grid_dy = int((screen_height - 2*padding) / grid_ny)

mean_min_dist = math.sqrt(grid_dx * grid_dy)

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


            # delete
            "D_overlap_factor": 0,
            "D_path_length": 0,
            # "D_total_station_types": 0,
            "D_avr_wait_time": 0,
            "D_not_isolated": 0,
            "D_has_isolated": 0,
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

        return float(score)
    
    def score_C_reuse(self, stations: List[Station], paths_cnt: int):
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
    
    def score_D(self, path_index: int, paths: List[PathConfig], in_game_paths: List[Path], has_not_connected_station: bool):
        try:
            target_path = paths[path_index]
            target_path_obj = in_game_paths[target_path.ingame_idx]
        except:
            print(len(paths), len(in_game_paths))
            print("acc:", target_path.ingame_idx)

            raise Exception
        
        score = 0

        # overlap_factor = 0
        # for other_path in paths:
        #     if other_path.stations == target_path.stations:
        #         continue

        #     overlap_factor = max(
        #         overlap_factor,
        #         len(set(target_path.stations) & set(other_path.stations)) \
        #             / ((len(target_path.stations) + len(other_path.stations)) / 2)
        #     )
        # score += self.weights["D_overlap_factor"] * overlap_factor

        total_path_len = sum([distance(a.position, b.position) for a, b in zip(target_path_obj.stations[:-1], target_path_obj.stations[1:])])
        path_len = total_path_len / mean_min_dist # / len(stations)
        score += self.weights["D_path_length"] * path_len

        # station_types = len(set([sta.shape.type for sta in target_path_obj.stations]))
        # score += self.weights["D_total_station_types"] * station_types

        # total_wait_time = 2 * total_path_len / (metro_speed_per_ms * 1000)
        # loop_wait_time = total_wait_time / (2 if target_path.is_loop else 1)
        # score += self.weights["D_avr_wait_time"] * loop_wait_time

        # score += self.weights["D_age"] * target_path_obj.age / 1000 / 30

        if has_not_connected_station:
            score += self.weights["D_has_isolated"]
        else:
            score += self.weights["D_not_isolated"]

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
            # try:
                game = Game(self)
                
                score = game.play(early_stopping)
                # print(score)
                return score
            # except:
            #     return None
        
        pseudo_min = simulate_game(early_stopping=500)
        while pseudo_min is None:
            pseudo_min = simulate_game(early_stopping=500)

        with ThreadPoolExecutor(max_workers=5) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=pseudo_min), range(5)))

        scores = [s for s in scores if s is not None]
        self.fitness = min(min(scores), pseudo_min)

    def statistics(self):
        def simulate_game(early_stopping: int):
            try:
                game = Game(self)
                return game.play(early_stopping)
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            scores = list(executor.map(lambda _: simulate_game(early_stopping=1E9), range(100)))

        print(scores)

    def single(self):
        game = Game(self, self.is_test)
        return game.play(early_stopping=1E9)

class Game:
    def __init__(self, player: Creature, is_test: bool = False):
        self.player = player
        self.is_test = is_test

        self.actions = []
        self.best_actions = []
        
        if self.is_test:
            self.gamespeed = 10
            self.visuals = True
        else:
            self.gamespeed = 200
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
        self.best_actions = []

        if len(self.game.paths) < num_paths:
            self.get_connect_actions()
        # else:
        #     self.get_delete_actions()

        self.best_actions.append(self.get_best_connect_action())
        self.best_actions.append(self.get_best_delete_action())

        # print("best:", self.best_actions)

        self.best_actions.sort(key=lambda act: act[1], reverse=True)

        do_action = self.best_actions[0][0]

        if do_action is None:
            return

        act_type, act = do_action

        if self.is_test:
            print(f"do({act_type}): {act}")

        if act_type == "C":
            self.apply_connect_action(act)
        elif act_type == "D":
            self.apply_delete_action(act)


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
                    self.actions.append(('C', (s1_ind, s2_ind)))

    def get_best_connect_action(self) -> Tuple[any, float]:
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

        return best_act, best_act_score
    
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
        def try_station(target_idx, other_idx, path_stations, ingame_stations):
            if target_idx not in path_stations:
                return None

            loc = path_stations.index(target_idx)
            if loc not in (0, len(path_stations) - 1):
                return None

            insert_at = loc + 1 if loc == len(path_stations) - 1 else loc
            score = self.player.score_C_reuse(ingame_stations, len(self.paths))
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

    def get_delete_actions(self):
        for path_index in range(len(self.paths)):
            self.actions.append(('D', path_index))

    def get_best_delete_action(self):
        best_act_score = -float("inf")
        best_act = None

        connected_stations = set()
        for path in self.paths:
            connected_stations.update(path.stations)

        has_not_connected_station = (len(self.game.stations) - len(connected_stations)) > 0
        
        for act in self.actions:
            if act[0] != "D":
                continue

            path_index = act[1]
            score = self.player.score_D(path_index, self.paths, self.game.paths, has_not_connected_station)

            if score > best_act_score:
                best_act_score = score
                best_act = act

        if best_act is None:
            return None, best_act_score

        return best_act, best_act_score
    
    def apply_delete_action(self, act: int):
        del_ingame_idx = self.paths[act].ingame_idx
        self.game.delete_path(del_ingame_idx)
        del self.paths[act]

        # update local in-game path index
        for path in self.paths:
            if path.ingame_idx >= del_ingame_idx:
                path.ingame_idx -= 1


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



