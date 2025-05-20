# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

# start of code

from typing import List, Tuple
from ga_entity import Creature

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# from joblib import Parallel, delayed

import math

class GA:
    def __init__(self):
        self.population_size = 50
        self.generation_limit = 100
        self.mutation_rate = 0.25

        self.population: List[Creature] = []
        self.parent_index: List[Tuple[int, int]] = []

        self.generate_initial_population()

    def generate_initial_population(self):
        # for _ in range(self.population_size):
        #     self.population.append(Creature(need_calc_fitness=True))

        def create_one():
            return Creature(need_calc_fitness=True)

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(lambda _: create_one(), range(self.population_size)))

        self.population.extend(results)

    def sort_population(self):
        self.population.sort(key=lambda c: c.fitness, reverse=True)

    def selection(self):
        self.parent_index = []

        fitnesses = [c.fitness for c in self.population]
        total_fit = sum(fitnesses)

        probabilities = [f / total_fit for f in fitnesses]

        for _ in range(self.population_size):
            parents = np.random.choice(range(len(self.population)), size=2, replace=False, p=np.array(probabilities))
            self.parent_index.append((parents[0], parents[1]))


    def crossover(self, p1: Creature, p2: Creature) -> List[Creature]:
        child1 = Creature()
        child2 = Creature()

        for w_name in p1.weights:
            child1.weights[w_name] = p1.weights[w_name]
            child2.weights[w_name] = p2.weights[w_name]

            # swap
            if np.random.random() < 0.5:
                child1.weights[w_name], child2.weights[w_name] = child2.weights[w_name], child1.weights[w_name]

        return [child1, child2]

    def mutate(self, c: Creature):
        c.weights['distance'] += np.random.uniform(-1, 1)
        c.weights['passenger_density'] += np.random.uniform(-1, 1)
        c.weights['station_type'] += np.random.uniform(-1, 1)

    def get_survivors(self):
        self.sort_population()
        self.population = self.population[:self.population_size]

    def cycle(self):
        # 1. select parents
        # 2. crossover
        # 3. mutate
        # 4. calc fitness
        # 5. merge parent and children
        # 6. get survivors
        # 7. check if cur best is goal

        print("@0", self.population[0].fitness)
        print("w:", self.population[0].weights)
        
        print('sel')
        self.selection()
        
        new_population: List[Creature] = []

        print('co')
        
        def create_children(p1_ind, p2_ind):
            childs = self.crossover(self.population[p1_ind], self.population[p2_ind])
            result = []
            for child in childs:
                if np.random.random() < self.mutation_rate:
                    self.mutate(child)
                child.try_calc_fitness()
                result.append(child)
            return result

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(create_children, p1, p2)
                for (p1, p2) in self.parent_index
            ]
            for future in as_completed(futures):
                new_population.extend(future.result())

        self.population.extend(new_population)
        self.get_survivors()


    def run(self):
        for i in range(self.generation_limit):
            print(f"iter {i}")
            self.cycle()
