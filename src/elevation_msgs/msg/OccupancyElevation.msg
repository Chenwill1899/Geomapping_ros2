# ElevationMap
std_msgs/Header header 

# original occupancy map data for the local map (access using: i.e. occupancy.data occupancy.info)
nav_msgs/OccupancyGrid occupancy

# height info for each grid
float32[] height

# roughness info  for each grid
float32[] roughness

# cost map
float32[] cost_map 

float32[] reward_cost 
