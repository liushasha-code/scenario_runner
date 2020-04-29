"""
Test spawn actor failure in carla 098.


"""

from __future__ import print_function

import glob
import os
import sys

# using carla 095
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'  # carla egg

# using carla096
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_096/PythonAPI'

# using carla098
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_098/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import numpy as np
import math

Red = carla.Color(r=255, g=0, b=0)
Green = carla.Color(r=0, g=255, b=0)

def draw_waypoint(world, waypoint):
    """
        Draw a waypoint in current world.
        A point at certain location, and an arrow pointing local x axis.

    """
    scalar = 0.5
    yaw = np.deg2rad(waypoint.transform.rotation.yaw)
    vector = scalar*np.array([np.cos(yaw), np.sin(yaw)])
    start = waypoint.transform.location
    end = start + carla.Location(x=vector[0], y=vector[1], z=start.z)
    # plot the waypoint
    debug = world.debug
    debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=Red, life_time=9999)
    debug.draw_point(start, size=0.05, color=Green, life_time=9999)
    world.tick()

def set_spectator(world, transform, view=0):
    """
        Set spectator at a certain viewpoint.
        param waypoint: a carla waypoint class
    """
    # transform = waypoint.transform
    location = transform.location
    rotation = transform.rotation

    if view == 0:
        print("Spectator is set to behind view.")
        # behind distance - d, height - h
        d = 10
        h = 8
        angle = transform.rotation.yaw
        a = math.radians(180 + angle)
        location = carla.Location(x=d * math.cos(a), y=d * math.sin(a), z=h) + transform.location
        rotation = carla.Rotation(yaw=angle, pitch=-15)
        print("done")
    elif view == 1:
        print("Spectator is set to overhead view.")
        # h = 200
        h = 15
        location = carla.Location(0, 0, h)+transform.location
        rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(location, rotation))
    world.tick()
    # print("d")


def main():


    host = 'localhost'
    port = 2000
    client_timeout = 2.0

    client = carla.Client(host, port)

    world = client.get_world()

    settings = world.get_settings()

    # The world contains the list blueprints that we can use for adding new
    # actors into the simulation.
    blueprint_library = world.get_blueprint_library()

    # Now let's filter all the blueprints of type 'vehicle' and choose one
    # at random.
    bp = random.choice(blueprint_library.filter('vehicle'))
    bp = blueprint_library.find('vehicle.lincoln.mkz2017')

    # Now we need to give an initial transform to the vehicle. We choose a
    # random transform from the list of recommended spawn points of the map.
    transform = random.choice(world.get_map().get_spawn_points())

    # ｔest transform difference
    map = world.get_map()
    waypoint_to_draw = map.get_waypoint(transform.location)

    # transform_2 = waypoint_to_draw.transform
    # original location
    location = waypoint_to_draw.transform.location
    # modified location
    location.z = 3

    # original rotation
    # rotation = carla.Rotation(yaw=waypoint_to_draw.transform.rotation.yaw)
    # modified rotation
    rotation = carla.Rotation(yaw=waypoint_to_draw.transform.rotation.yaw)

    # set transform_2
    transform_2 = carla.Transform(location=location, rotation=rotation)


    # So let's tell the world to spawn the vehicle.
    vehicle = world.spawn_actor(bp, transform_2)


    # draw_waypoint(world, waypoint_to_draw)

    set_spectator(world, transform, view=1)

    print("done")


if __name__ == '__main__':
    main()

