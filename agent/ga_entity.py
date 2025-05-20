# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

# start of code

from entity.path import Path

import numpy as np

from typing import List, Tuple

from api import ProgressiveStationGame

from entity.station import Station
from geometry.utils import distance

from config import num_paths

class Gene:
    def __init__(self):
        # += w * dist(A, B)

        # += w * (passA + passB)

        # diversity_score = 1 if station_a.shape != station_b.shape else 0
        # += w * diversity_score
        
        # overlap_score = get_overlap_ratio(new_line_path, existing_lines)
        # += w * overlap_score
        self.weights = {
            # connect
            "distance": np.random.uniform(0, 2),
            "passenger_density": np.random.uniform(0, 2),
            "station_type": np.random.uniform(0, 2),
            # "line_overlap": np.random.uniform(0, 1),

            # # redesign
            # "coverage_gain": np.random.uniform(-1, 1),
            # "congestion_reduction": np.random.uniform(-1, 1),
            # "average_distance": np.random.uniform(-1, 1),
            # "station_type_diversity": np.random.uniform(-1, 1),
            # "reuse_existing_path": np.random.uniform(-1, 1),

            # "choose_connect": np.random.uniform(0, 1)
        }
    
    def score_connection(self, sta1: Station, sta2: Station):
        score = 0

        dist = distance(sta1.position, sta2.position)
        score -= self.weights["distance"] * dist

        passenger_density = len(sta1.passengers) + len(sta2.passengers)
        score += self.weights["passenger_density"] * passenger_density

        diversity = 1 if sta1.shape.type != sta2.shape.type else 0
        score += self.weights["station_type"] * diversity

        # overlap = estimate_overlap_ratio(sta1, sta2, game_state.existing_lines)
        # score -= weights["existing_line_overlap"] * overlap

        return score

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
        game = Game(self)
        self.fitness = game.play()

class Game:
    def __init__(self, player: Creature):
        self.player = player
        self.acts = []

        gamespeed = 5
        in_game_break_time = (1/gamespeed) * 1000 # stop periodic yield
        self.game = ProgressiveStationGame(gamespeed=gamespeed, yield_interval_ms=in_game_break_time, visuals=True)

        self.paths_config: List[Tuple[List[int], bool]] = []

    def play(self) -> int:
        simulation = self.game.run()
        try:
            while True:
                next(simulation)
                self.act()
        except StopIteration as score:
            return score.value

    def act(self):
        if len(self.game.stations) < 2:
            return
    
        self.acts = []

        # print("getting actions...")
        self.get_connect_actions()

        # print("choosing best action...")
        best_action = self.get_best_connect_actions()

        if best_action is None:
            # print("no best!")
            return
        
        # print("best get:", best_action)
        
        use_existing_path, path_num, disjoint_station, insert_at = \
            self.should_use_existing_path(best_action[0], best_action[1])

        # print(f"act {use_existing_path}: path {path_num} @sta {disjoint_station} @ins {insert_at}")

        if use_existing_path:
            # print("use cur")

            (new_path, new_loop) = self.paths_config[path_num]
            new_path.insert(insert_at, disjoint_station)
            self.paths_config[path_num] = (new_path, new_loop)
            self.game.recreate_path(path_num, self.paths_config[path_num])

            # print("mod path", self.paths_config[path_num])
        else:
            if len(self.game.paths) >= num_paths:
                return
            
            # print("use new")
            
            new_path = ([best_action[0], best_action[1]], False)
            self.game.create_path(new_path)
            self.paths_config.append(new_path)

        # print("modified:", self.paths_config)
        # print("--------")

    def should_use_existing_path(self, sta1: int, sta2: int):
        """
        return whether, path_index, disjoint_station, insert_at
        """

        for ind, (path, loop) in enumerate(self.paths_config):
            if sta1 in path:
                is_outer = path.index(sta1) in [0, len(path) - 1]
                if is_outer:
                    return True, ind, sta2, path.index(sta1)
            elif sta2 in path:
                is_outer = path.index(sta2) in [0, len(path) - 1]
                if is_outer:
                    return True, ind, sta1, path.index(sta2)

        return False, None, None, None

    def get_connect_actions(self):
        stations = self.game.stations
        paths = self.game.paths
        for i in range(len(stations)):
            sta1 = stations[i]
            for j in range(i+1, len(stations)):
                sta2 = stations[j]

                is_connected = any([sta1 in path.stations and sta2 in path.stations for path in paths])
                if not is_connected:
                    self.acts.append((i, j))

    def get_best_connect_actions(self):
        best_act_score = -9999999
        best_act = None

        for act in self.acts:
            score = self.player.score_connection(self.game.stations[act[0]], self.game.stations[act[1]])

            if score > best_act_score:
                best_act_score = score
                best_act = act

        # if best_act_score > 0:
        #     return best_act
        # return None

        return best_act



