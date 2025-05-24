from ga_entity import Creature

# before early-stopping: {'C_dist': -0.2974412638856899, 'C_pass_density': -0.12898951899033978, 'C_station_type': 0.05748521385220699, 'C_dist_to_com': 0.06803900165917555, 'C_avr_path_length': -0.20618790814020688, 'C_avr_pass_density': 0.08847238439556351, 'C_total_station_types': -0.3143426279447766}

best = Creature()
best.is_test = True
# best.weights = {'C_dist': -0.13481987995739636, 'C_pass_density': 0.05160201619080579, 'C_station_type': 1.2681687710016534, 'C_dist_to_com': 0.28028569029967365, 'C_avr_path_length': -0.418454582200844, 'C_avr_pass_density': 1.265816168030553, 'C_total_station_types': -0.9781890365637345}

# best.weights = {'C_dist': 0.051976860332607955, 'C_pass_density': -0.5189084049016586, 'C_station_type': 0.5953634849639453, 'C_dist_to_com': 0.22080221171395326, 'C_avr_path_length': -0.4276473443258802, 'C_avr_pass_density': -0.2914965853270199, 'C_total_station_types': -0.9653894894852386}

# best.weights = {'C_dist': -0.4072952759547898, 'C_pass_density': -0.7595534063698407, 'C_station_type': 0.5447911403731116, 'C_dist_to_com': -0.6240538857329713, 'C_path_length': -0.5014427562330274, 'C_avr_pass_density': -0.2008619644369345, 'C_total_station_types': 
# 0.15189103608058174}

# ^ C only

# best.weights = {'C_dist': -0.20757763299134294, 'C_pass_density': -0.5147931779393012, 'C_station_type': -0.33396945689314245, 'C_dist_to_com': -0.26178008361951766, 'C_avr_path_length': -0.266668815843335, 'C_avr_pass_density': 0.11301174544413328, 'C_total_station_types': -0.5807223836579805, 'D_overlap_factor': 0.5527935043267225, 'D_avr_path_length': 0.257949242556705, 'D_avr_wait_time': -0.18200360487855716, 'D_age': 0.22705344545525263}



best.weights = {'C_dist': 0.46090081330063964, 'C_pass_density': 0.047718610215784496, 'C_station_type': 0.0895193542886507, 'C_dist_to_com': -0.5206933011020748, 'C_avr_path_length': -0.8378982333948927, 'C_avr_pass_density': -0.24628884139543905, 'C_total_station_types': -0.36343317107425416, 'D_overlap_factor': 1.0324482106166368, 'D_path_length': 0.3912317437447609, 'D_avr_wait_time': 0.367487560024351, 'D_not_isolated': -0.3609310403918822, 'D_has_isolated': -0.2560873433672511}




# ^ C & D

print(best.single())