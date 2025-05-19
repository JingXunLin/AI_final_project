import copy
import datetime
import os
import random
import sys
import time
from typing import List, Tuple

import pygame

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")

from api.static import StaticStationGame

game = StaticStationGame(gamespeed=25, visuals=False)

def neighbors(state: Tuple[List[int], bool]) -> List[Tuple[List[int], bool]]:
    neighbors = []
    for station in range(len(game.stations)):
        if station not in state[0]:
            for i in range(len(state[0])):
                neighbor: list = copy.copy(state[0])
                neighbor.insert(i, station)
                neighbors.append((neighbor, False))
            neighbor: list = copy.copy(state[0])
            neighbor.append(station)
            neighbors.append((neighbor, False))
    
    if len(state[0]) >= 3:
        neighbors = [(x[0], y) for x in neighbors for y in [False, True]]
    return neighbors

def local_search(previous_paths: List[Tuple[List[int], bool]] = [], start: Tuple[List[int], bool] = ([0], False)) -> Tuple[List[int], bool]:
    t = datetime.datetime.now()
    best_paths, best_score = [start], game.run(*previous_paths, start)
    
    rng = random.Random(time.time()) # don't influence game randomness
    local_minima_found = False

    chosen_path: Tuple[List[int], bool]
    while not local_minima_found:
        chosen_path = rng.choice(best_paths) # randomly select a path with same score
        best_paths.remove(chosen_path)

        local_minima_found = True
        neighbors_list = neighbors(chosen_path)
        
        for neighbor in neighbors_list:
            score = game.run(*previous_paths, neighbor)
            if score >= best_score:
                if score > best_score:
                    best_paths.clear()
                    best_score = score
                best_paths.append(neighbor)
                local_minima_found = False
        print(f'Iter thru {len(neighbors_list)} w/ score = {best_score}, t = {(datetime.datetime.now() - t).seconds} sec.; ')
        t = datetime.datetime.now()

    print(f'Found local minima! Path = {chosen_path}, Score = {best_score}')

    game.open_window()
    game.screenshot(f'./agent/local_search/results/local_search_{len(previous_paths)}th_{"_".join(map(str, chosen_path[0]))}{"_looped" if chosen_path[1] else ""}.png', *previous_paths, chosen_path)
    print(f'Best Score: {game.run(*previous_paths, chosen_path)}')
    game.close_window()

    return chosen_path

best_paths = [local_search([], ([0], False))]
uncovered_stations = [x for x in range(len(game.mediator.stations)) if x not in best_paths[0][0]]
for i in range(1, 7):
    print('Uncovered stations:', uncovered_stations)
    if len(uncovered_stations) == 0:
        start_station = random.randint(0, len(game.mediator.stations) - 1)
    else:
        start_station = uncovered_stations[0]
    new_path = local_search(previous_paths=best_paths, start=([start_station], False))
    for station in new_path[0]:
        if station in uncovered_stations:
            uncovered_stations.remove(station)
    best_paths.append(new_path)

game.open_window()
game.mediator.gamespeed = 10
game.mediator.is_paused = True
game.run(*best_paths)