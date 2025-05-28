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

# best.weights = {'C_dist': 0.1945482221219615, 'C_pass_density': 1.0602180873439515, 'C_station_type': -0.32786348153859507, 'C_dist_to_com': -0.30440521239235213, 'C_avr_path_length': -0.857324728449404, 'C_avr_pass_density': 0.45716001490671426, 'C_total_station_types': -0.4952653475143276, 'U_avr_path_length': 0.4546961312564005, 'U_avr_pass_density': 0.10304167685646409, 'U_total_station_types': 0.7937839465452283}

# best.weights = {'C_dist': 0.5537510416892144, 'C_pass_density': -0.4437697681694046, 'C_station_type': 0.4811784784963223, 'C_dist_to_com': 0.48570536524268676, 'C_avr_path_length': -0.4540985686798379, 'C_avr_pass_density': -0.649306715658289, 'C_total_station_types': -0.11193510570802251, 'U_avr_path_length': 0.808342720540329, 'U_wait_time_mean': -0.22257134178683977, 'U_wait_time_std': -0.04497713613819694, 'U_total_station_types': 0.7040911370976295}

# succeed in basic difficulty!!!!
best.weights = {'C_dist': -0.5956543047854342, 'C_pass_density': -0.7870721708103461, 'C_station_type': 0.49594080055672174, 'C_dist_to_com': -0.21951759115827188, 'C_avr_path_length': -0.5443971696159355, 'C_avr_pass_density': 0.8692824219994149, 'C_total_station_types': -0.7017724212240876, 'U_avr_path_length': -0.01320670949902888, 'U_wait_time_mean': 0.5353797758950448, 'U_wait_time_std': -0.36437763589671857, 'U_total_station_types': 0.08109701042055815}

# best.weights = {'C_dist': -0.3309659413375334, 'C_pass_density': -0.004086714175130493, 'C_station_type': 0.48443959908740025, 'C_dist_to_com': -0.25295080490436334, 'C_avr_path_length': -0.6478882477062834, 'C_avr_pass_density': -0.11421934786496313, 'C_total_station_types': -0.10166129451458254, 'U_avr_path_length': -0.05261890650128036, 'U_wait_time_mean': 0.34128527171025197, 'U_wait_time_std': -0.2114544295164119, 'U_total_station_types': 0.7363912310983514, 'U_has_isolated_stations': 0.771020998615015}


# ^ basic difficulty

# best.weights = {'C_dist': 0.35690396653105555, 'C_pass_density': 0.47802767817440406, 'C_station_type': -0.20082706060200278, 'C_dist_to_com': -0.5950036972741042, 'C_avr_path_length': 0.10290579703989519, 'C_avr_pass_density': -0.13582378096009623, 'C_total_station_types': -1.2727918538069805, 'U_avr_path_length': -0.11346808088287075, 'U_wait_time_mean': 0.3826470110245438, 'U_wait_time_std': -1.0162914678930566, 'U_total_station_types': 0.6250576067751736, 'U_has_isolated_stations': -0.3524152958958396, 'U_station_cnt': 0.8889959888279673}

# best.weights = {'C_dist': -0.13943054496607898, 'C_pass_density': -0.01629060832149015, 'C_station_type': 0.8397629996, 'C_dist_to_com': -0.2203841320648124, 'C_avr_path_length': 0.1020319655540931, 'C_avr_pass_density': 0.4348987732622601, 'C_total_station_types': -0.48762828457383867, 'U_avr_path_length': -0.21521146242266997, 'U_wait_time_mean': -0.14607838677018622, 'U_wait_time_std': -0.5124704171623317, 'U_total_station_types': 0.7231075256745332, 'U_has_isolated_stations': 0.5615189389198934, 'U_station_cnt': 0.26857864061372144}

# best.weights = {'C_dist': 0.0017089998468322806, 'C_pass_density': -0.20329956064339844, 'C_station_type': 0.08629082698323413, 'C_dist_to_com': 0.33372057716957687, 'C_avr_path_length': -0.41705077005121405, 'C_avr_pass_density': 0.138451971479719, 'C_total_station_types': 0.029999343003143226, 'U_avr_path_length': -0.09460038662145372, 'U_wait_time_mean': 
# 0.6578588333434983, 'U_wait_time_std': -0.21649312129670661, 'U_total_station_types': 0.7751765533213859, 'U_has_isolated_stations': 0.3952371380512823, 'U_station_cnt': 0.7901535365120796}
"""
count   100.000000
mean    519.620000
std     353.439972
min     186.000000
25%     312.750000
50%     401.000000
75%     596.250000
max    2211.000000
"""


# best.weights = {'C_dist': -0.007824075651002235, 'C_pass_density': -0.20892131091904403, 'C_station_type': 0.08699793421516208, 'C_dist_to_com': 0.27229817490924796, 'C_avr_path_length': -0.39186393308897755, 'C_avr_pass_density': 0.09634001506234625, 'C_total_station_types': 0.007793284121506379, 'U_avr_path_length': -0.06971843786038585, 'U_wait_time_mean': 0.5999725087979634, 'U_wait_time_std': -0.21131607686241807, 'U_total_station_types': 0.7112132341998362, 'U_has_isolated_stations': 0.41025204578428526, 'U_station_cnt': 0.7475450520599092}
"""
count   100.000000
mean    676.710000
std     680.988991
min     115.000000
25%     310.750000
50%     386.000000
75%     679.750000
max    3005.000000
"""



# best.weights = {'C_dist': -0.02284773323438814, 'C_pass_density': 0.41793158245884987, 'C_station_type': 0.008637922594194714, 'C_dist_to_com': 0.5280867505007889, 'C_avr_path_length': -0.5910715104593969, 'C_avr_pass_density': -0.04449321143606247, 
# 'C_total_station_types': -0.5320315866233111, 'U_wait_time_mean': -0.023725595987743958, 'U_wait_time_std': -0.14007248778498682, 'U_total_station_types': 0.09559591211834272, 'U_has_isolated_stations': 0.1565170332460652, 'U_station_cnt': 0.10848333935977247}
"""
count   100.000000
mean    516.730000
std     386.803368
min     118.000000
25%     291.750000
50%     406.000000
75%     617.750000
max    3002.000000
"""


# best.weight = {'C_dist': -0.1924514763057822, 'C_pass_density': 0.3296679260520829, 'C_station_type': 0.14271864708972326, 'C_dist_to_com': 0.1170645743371325, 'C_avr_path_length': -0.3075730555134038, 'C_avr_pass_density': 0.01666476759881308, 'C_total_station_types': -0.8120529453207163, 'U_wait_time_mean': -0.006624604895234663, 'U_wait_time_std': -0.30471429606559225, 'U_total_station_types': 0.32096355426374407, 'U_has_isolated_stations': -0.032217037469598794, 'U_station_cnt': 0.20588052874659693}
"""
count   100.000000
mean    504.430000
std     353.790078
min      88.000000
25%     313.750000
50%     414.000000
75%     620.250000
max    3003.000000
"""


# ^ difficulty 2

# ^ C & U

# print(best.statistics())

print(best.single())