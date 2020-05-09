"""
Some basic info of junction in Town03.

todo: a better data structure, use class or dict?
"""

import carla

import numpy as np

# default spawn location of ego vehicle
ego_start_transform = carla.Transform(carla.Location(x=53.0, y=128.0, z=3.0),
                                      carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))

# default right turn middle waypoint
# waypoint after the junciton


# junction center
# Location(x=-1.317724, y=132.692886, z=0.000000)
junction_center = carla.Location(x=-1.317724, y=132.692886, z=0.000000)



