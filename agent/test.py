from ga_entity import Creature

# before early-stopping: {'C_dist': -0.2974412638856899, 'C_pass_density': -0.12898951899033978, 'C_station_type': 0.05748521385220699, 'C_dist_to_com': 0.06803900165917555, 'C_avr_path_length': -0.20618790814020688, 'C_avr_pass_density': 0.08847238439556351, 'C_total_station_types': -0.3143426279447766}

best = Creature()
# best.weights = {'C_dist': -0.13481987995739636, 'C_pass_density': 0.05160201619080579, 'C_station_type': 1.2681687710016534, 'C_dist_to_com': 0.28028569029967365, 'C_avr_path_length': -0.418454582200844, 'C_avr_pass_density': 1.265816168030553, 'C_total_station_types': -0.9781890365637345}

best.weights = {'C_dist': 0.051976860332607955, 'C_pass_density': -0.5189084049016586, 'C_station_type': 0.5953634849639453, 'C_dist_to_com': 0.22080221171395326, 'C_avr_path_length': -0.4276473443258802, 'C_avr_pass_density': -0.2914965853270199, 'C_total_station_types': -0.9653894894852386}

print(best.test())