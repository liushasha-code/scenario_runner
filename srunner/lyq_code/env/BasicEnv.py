"""
This is a basic class to create a carla env.
"""

from __future__ import print_function
import glob
import os
import sys

# carla 098 API is recommended
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
import random
import numpy as np
import math
import traceback

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)


class BasicEnv:
    """
    The basic class to generate a carla env for test.
    """
    def __init__(self, town='Town03', host='localhost', port=2000, client_timeout=2.0):
        # setup client
        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.debug = self.world.debug  # world debug for plot
        self.blueprint_library = self.world.get_blueprint_library()  # blueprint
        # set world status
        self.set_world()  # world settings
        self.set_weather()  # weather

        self.spectator = self.world.get_spectator()
        self.traffic_manager = self.client.get_trafficmanager()  # check carla version before using

    def set_world(self, sync_mode=False, frame_rate=50.0, no_render_mode=False):
        """
        Setup carla world settings.
        Under sync_mode(sync_mode = True), require world.tick() to run
        """
        settings = self.world.get_settings()
        # world settings parameters
        settings.fixed_delta_seconds = 1.0 / frame_rate
        settings.no_rendering_mode = no_render_mode
        settings.synchronous_mode = sync_mode
        self.world.apply_settings(settings)
        self.world.tick()  # refresh world

    def set_weather(self, weather='ClearNoon'):
        """
        Set weather for the world
        Common weather in carla:
            ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon,
            SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset,
            CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset,
            MidRainSunset, HardRainSunset.
        """
        weather_dict = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
        }
        weather_selection = None
        for key in weather_dict:
            if key == weather:
                weather_selection = weather_dict[key]
        if not weather_selection:
            print('Specified weather not found. ClearNoon is set.')
            weather_selection = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather_selection)
        self.world.tick()

    def set_spectator(self, transform, view=1, h=50):
        """
        Set spectator on a transform at a specified viewpoint.
        """
        location = transform.location
        rotation = transform.rotation
        if view == 0:
            print("Spectator is set to behind view.")
            # behind distance - d, height - h
            _d = 8
            _h = 8
            angle = transform.rotation.yaw
            a = math.radians(180 + angle)
            location = carla.Location(x=_d * math.cos(a), y=_d * math.sin(a), z=_h) + transform.location
            rotation = carla.Rotation(yaw=angle, pitch=-15)
        elif view == 1:
            print("Spectator is set to overhead view.")
            height = h
            # height = 100
            location = carla.Location(0, 0, height) + transform.location
            rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()

    def draw_waypoint(self, transform, color=[red, green]):
        """
        Draw a point determined by transform(or waypoint).
        A spot to mark location
        An arrow to x-axis direction vector

        :param transform: carla.Transform or carla.Waypoint
        :param color: color of arrow and spot
        """
        if isinstance(transform, carla.Waypoint):
            transform = transform.transform
        scalar = 1.5
        yaw = np.deg2rad(transform.rotation.yaw)
        vector = scalar * np.array([np.cos(yaw), np.sin(yaw)])
        start = transform.location
        end = start + carla.Location(x=vector[0], y=vector[1], z=0)
        # plot the waypoint
        self.debug.draw_point(start, size=0.05, color=color[0], life_time=99999)
        self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.20, color=color[1], life_time=99999)
        self.world.tick()

    @staticmethod
    def coord2loc(coords):
        """
        Transform a coords to carla location.
        Coords will be set default on the ground, z=0.
        :param coords: coordinates of np.array of list.
        :return: carla.location
        """
        location = carla.Location(x=float(coords[0]), y=float(coords[1]), z=0.0)
        return location

    @staticmethod
    def loc2coord(location):
        """
        Transform a carla location to coords.
        :param location: carla.Location
        :return: coordinate in carla world.
        """
        coords = np.array([location.x, location.y, location.z])
        return coords


def main():
    try:
        test = BasicEnv()
        print('test env is created.')
    except:
        traceback.print_exc()
    finally:
        del test


if __name__ == '__main__':
    main()
