# handling path
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

# start of code

from typing import List, Tuple
from ga_entity import Creature

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class GA:
    def __init__(self):
        self.population_size = 40
        self.generation_limit = 100
        self.mutation_rate = 0.25

        self.population: List[Creature] = []
        self.parent_index: List[Tuple[int, int]] = []

        self.generate_initial_population()

    def generate_initial_population(self):
        def create_one():
            return Creature(need_calc_fitness=True)

        with ThreadPoolExecutor(max_workers=self.population_size) as executor:
            results = list(tqdm(
                executor.map(lambda _: create_one(), range(self.population_size)),
                total=self.population_size,
                desc="Initial creatures"
            ))

        self.population.extend(results)
        self.sort_population()

    def sort_population(self):
        self.population.sort(key=lambda c: c.fitness, reverse=True)

    def selection(self):
        self.parent_index = []

        fitnesses = [c.fitness + 1 for c in self.population]
        total_fit = sum(fitnesses)

        probabilities = [f / total_fit for f in fitnesses]

        for _ in range(self.population_size):
            parents = np.random.choice(range(len(self.population)), size=2, replace=False, p=np.array(probabilities))
            self.parent_index.append((parents[0], parents[1]))


    def crossover(self, p1: Creature, p2: Creature) -> List[Creature]:
        children = []

        for _ in range(2):
            child = Creature()
            r = np.random.uniform(-0.1, 1.1)
            for w_name in p1.weights:
                child.weights[w_name] = p1.weights[w_name] * r + p2.weights[w_name] * (1 - r)

            children.append(child)

        return children

    def mutate(self, c: Creature):
        for key in c.weights:
            if np.random.rand() < 0.8:  # 每個基因 80% 的機率被 mutate
                delta = np.random.uniform(-0.7, 0.7)
                c.weights[key] += delta
                c.weights[key] = max(-5.0, min(5.0, c.weights[key]))


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

        print(f"Best fit: {self.population[0].fitness}, wi's: {self.population[0].weights}")
        print("Fitnesses:", [c.fitness for c in self.population])
        
        print('Selecting...')
        self.selection()
        
        new_population: List[Creature] = []

        def create_children(p1_ind, p2_ind):
            children = self.crossover(self.population[p1_ind], self.population[p2_ind])
            result = []
            for child in children:
                if np.random.random() < self.mutation_rate:
                    self.mutate(child)
                child.try_calc_fitness()
                result.append(child)
            return result

        with ThreadPoolExecutor(max_workers=self.population_size) as executor:
            futures = [
                executor.submit(create_children, p1, p2)
                for (p1, p2) in self.parent_index
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Crossover"):
                new_population.extend(future.result())

        self.population.extend(new_population)
        self.get_survivors()


    def run(self):
        for i in range(self.generation_limit):
            print(f"iter {i}")
            self.cycle()
