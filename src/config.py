from geometry.type import ShapeType

# game
framerate = 60

# screen
screen_width = 1080
screen_height = 720
screen_color = (117, 182, 255)

# station
num_stations_max = 20
station_grid_size = (14, 10)
station_spawning_interval_step = 60 * framerate # 60 secs?
station_padding = 100
station_size = 10
station_capacity = 12
station_color = (0, 20, 50)
station_shape_type_list = [
    ShapeType.RECT,
    ShapeType.CIRCLE,
    ShapeType.TRIANGLE,
    ShapeType.CROSS,
]
station_passengers_per_row = 4
station_full_timeout = 50 # in seconds

# passenger
passenger_size = 3
passenger_color = (0, 50, 120)
passenger_spawning_start_step = 1
passenger_spawning_interval_step = 10 * framerate
passenger_display_buffer = 3 * passenger_size

# metro
num_metros = num_stations_max
metro_size = 10
metro_color = (200, 200, 200)
metro_capacity = 6
metro_speed_per_ms = 150 / 1000  # pixels / ms
metro_passengers_per_row = 3

# path
toggle_split_lane = False
num_paths = 7
path_width = 5
path_order_shift = 5

# button
button_color = (180, 180, 180)
button_size = 20

# path button
path_button_buffer = 20
path_button_dist_to_bottom = 50

# calculate path_button_start_left if split buttons evenly
path_button_start_left = (
    (screen_width - (num_paths * path_button_buffer) - (num_paths * button_size))
    / 2
)

path_button_cross_size = 20
path_button_cross_width = 5

# text
score_font_size = 50
score_display_coords = (20, 20)
