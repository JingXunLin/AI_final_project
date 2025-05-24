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



# best.weights = {'C_dist': 0.46090081330063964, 'C_pass_density': 0.047718610215784496, 'C_station_type': 0.0895193542886507, 'C_dist_to_com': -0.5206933011020748, 'C_avr_path_length': -0.8378982333948927, 'C_avr_pass_density': -0.24628884139543905, 'C_total_station_types': -0.36343317107425416, 'D_overlap_factor': 1.0324482106166368, 'D_path_length': 0.3912317437447609, 'D_avr_wait_time': 0.367487560024351, 'D_not_isolated': -0.3609310403918822, 'D_has_isolated': -0.2560873433672511}




# ^ C & D


# best.weights = {'C_dist': -0.39462379761124716, 'C_pass_density': -1.12437883739848, 'C_station_type': 0.11963203129972885, 'C_dist_to_com': 
# 0.38659712618831077, 'C_avr_path_length': -1.0457967629113376, 'C_avr_pass_density': -0.6246968731492903, 'C_total_station_types': 0.24974117005897548, 'U_is_loop': 0.13978631953895468, 'U_avr_path_length': -0.005060420468204752, 'U_exact_avr_wait_time': -0.7945527496918182, 'U_avr_pass_density': -0.7587256279050292, 'U_total_station_types': -0.42163559393924516}

# best.weights = {'C_dist': 0.4094063059108584, 'C_pass_density': 0.38431521512574696, 'C_station_type': 0.5675695748509652, 'C_dist_to_com': 0.3080252186526402, 'C_avr_path_length': -0.2106612225070522, 'C_avr_pass_density': 0.2647392962740906, 'C_total_station_types': -0.10284783899227484, 'U_avr_path_length': -0.4932083926635415, 'U_avr_pass_density': 0.08549721748621, 'U_total_station_types': -0.6499535244299}

# best.weights = {'C_dist': -0.07387535538572529, 'C_pass_density': -0.47501166030749215, 'C_station_type': 0.005220816570619326, 'C_dist_to_com': 0.622719544144488, 'C_avr_path_length': -0.05664755679928026, 'C_avr_pass_density': 0.09773060172147169, 'C_total_station_types': -0.7681860732777703, 'U_avr_path_length': -0.563369795753694, 'U_avr_pass_density': -0.20298467843111948, 'U_total_station_types': -0.4369313669764131}

# best.weights = {'C_dist': 0.37362594898113377, 'C_pass_density': -0.21657066789271218, 'C_station_type': -0.1561613953582895, 'C_dist_to_com': 0.3287302251791112, 'C_avr_path_length': -0.4688930048511986, 'C_avr_pass_density': 0.02134109025663304, 'C_total_station_types': -0.05180175640407855, 'U_avr_path_length': -0.012314831940689473, 'U_avr_pass_density': -0.9516134505205367, 'U_total_station_types': -0.1076513914461442}

# best.weights = {'C_dist': 0.2013070603985299, 'C_pass_density': 0.878596659488104, 'C_station_type': 0.18580867058326514, 'C_dist_to_com': 0.08099804357594342, 'C_avr_path_length': -0.23321750301814093, 'C_avr_pass_density': -1.0463719010900423, 'C_total_station_types': -0.03964309727929457, 'U_avr_path_length': 1.0252541167594655, 'U_avr_pass_density': -0.5863813234812765, 'U_total_station_types': 0.997308908788942}

best.weights = {'C_dist': 0.1945482221219615, 'C_pass_density': 1.0602180873439515, 'C_station_type': -0.32786348153859507, 'C_dist_to_com': -0.30440521239235213, 'C_avr_path_length': -0.857324728449404, 'C_avr_pass_density': 0.45716001490671426, 'C_total_station_types': -0.4952653475143276, 'U_avr_path_length': 0.4546961312564005, 'U_avr_pass_density': 0.10304167685646409, 'U_total_station_types': 0.7937839465452283}

# ^ C & U

print(best.statistics())