"""
Some utilitiesfor visulization of carla experiment.
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

import time
import math
import numpy as np
import random

from srunner.util_development.carla_rgb_color import *




def get_azimuth(view_vector):
    """
        Get current azimuth from direction vector.
        Using only yaw and pitch angle.
    :param view_vector: vector of view direction in vehicle coord frame
    :return: azimuth in degrees, carla.Rotation
    """

    # transform matrix from World coord to ego vehicle
    # only yaw rotation is considered
    # np.cos(np.radians())
    # M1 = np.array([np.cos])

    # rotation around y axis
    x = view_vector[0]
    y = view_vector[1]
    z = view_vector[2]
    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = np.rad2deg(np.arctan2(z, np.sqrt(x*x+y*y)))

    roll = 0  # always not using roll
    azimuth = {"pitch": pitch, "yaw": yaw, "roll": roll}
    # rotation = carla.Rotation(yaw=yaw, roll=0, pitch=pitch)
    return azimuth

def get_rotation_matrix_2D(transform):
    """
        Get a 2D transform matrix of a specified transform
        from actor reference frame to map coordinate frame
    :param transform: actor transform, actually only use yaw angle
    :return: rotation matrix
    """
    yaw = np.deg2rad(transform.rotation.yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rotation_matrix_2D = np.array([[cy, -sy],
                                   [sy, cy]])
    return rotation_matrix_2D

def plot_local_coordinate_frame(world, origin_transform, axis_length_scale=3, life_time=99999, color_scheme=0):
    """
        Plot local coordinate frame.
        Using initial vehicle transform as origin
        todo: add text to identify axis, set x-axis always along longitudinal direction
    :param origin_transform: origin of local frame, in class transform
    :param axis_length_scale: length scale for axis vector
    :return: none, plot vectors of 3 axis in server world
    """

    # for test
    # origin_transform = transform
    # axis_length_scale = 3

    # longitudinal direction(x-axis)
    global x_axis_color
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # get coords in array and uplift with a height
    h = 3
    Origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, origin_transform.location.z + h])
    # elevate z coordinate
    Origin_location = carla.Location(Origin_coord[0], Origin_coord[1], Origin_coord[2])
    # x axis destination
    x_des_coord = Origin_coord + axis_length_scale * np.array([cy, sy, 0])
    x_des = carla.Location(x_des_coord[0], x_des_coord[1], x_des_coord[2])
    # y axis destination
    y_des_coord = Origin_coord + axis_length_scale * np.array([-sy, cy, 0])
    y_des = carla.Location(y_des_coord[0], y_des_coord[1], y_des_coord[2])
    # z axis destination
    z_des_coord = Origin_coord + axis_length_scale * np.array([0, 0, 1])
    z_des = carla.Location(z_des_coord[0], z_des_coord[1], z_des_coord[2])

    """
        color for each axis, carla.Color
        x-axis red:     (255, 0, 0)
        y-axis green:   (0, 255, 0)
        z-axis blue:    (0, 0, 255)
    """
    if color_scheme == 0:
        x_axis_color = carla.Color(r=255, g=0, b=0)
        y_axis_color = carla.Color(r=0, g=255, b=0)
        z_axis_color = carla.Color(r=0, g=0, b=255)
    elif color_scheme == 1:
        x_axis_color = carla.Color(r=252, g=157, b=154)
        y_axis_color = carla.Color(r=131, g=175, b=155)
        z_axis_color = carla.Color(r=96, g=143, b=159)

    # axis feature
    # thickness = 0.1f
    # arrow_size = 0.1f

    # begin, end, thickness=0.1f, arrow_size=0.1f, color=(255,0,0), life_time=-1.0f
    # draw x axis
    world.debug.draw_arrow(Origin_location, x_des, color=x_axis_color, life_time=life_time)
    # draw y axis
    world.debug.draw_arrow(Origin_location, y_des, color=y_axis_color, life_time=life_time)
    # draw z axis
    world.debug.draw_arrow(Origin_location, z_des, color=z_axis_color, life_time=life_time)

    # draw axis text next to arrow
    offset = 0.5
    x_text = carla.Location(x_des_coord[0]+offset, x_des_coord[1]+offset, x_des_coord[2]+offset)
    y_text = carla.Location(y_des_coord[0]+offset, y_des_coord[1]+offset, y_des_coord[2]+offset)
    z_text = carla.Location(z_des_coord[0]+offset, z_des_coord[1]+offset, z_des_coord[2]+offset)
    world.debug.draw_string(x_text, text='x', color=x_axis_color)
    world.debug.draw_string(y_text, text='y', color=x_axis_color)
    world.debug.draw_string(z_text, text='z', color=x_axis_color)


