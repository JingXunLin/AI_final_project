from ga_entity import Creature

best = Creature()
best.w = {'distance': 0.3259300677786757, 'passenger_density': 0.22900271536281913, 'station_type': 0.937223599062018}

best.try_calc_fitness()
print(best.fitness)