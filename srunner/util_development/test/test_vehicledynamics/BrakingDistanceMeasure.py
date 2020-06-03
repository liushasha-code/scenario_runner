"""
    This version didn't use OOP encapsulation

    Methods and data are not encapsulated into class.

 =======================================================

    Measure vehicle lincoln mkz braking distance.

    As a previous knowledge for DRL training

"""

import glob
import os
import sys

try:
    # sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('../CARLA_096/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import numpy as np
import math
import random

from srunner.tools.scenario_helper import *
from Env096.util import set_spectator
from srunner.challenge.utils.route_manipulation import interpolate_trajectory
from agents.navigation.local_planner import LocalPlanner

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

"""
    Some public method
"""

def plot_local_coordinate_frame(world, origin_transform, axis_length_scale = 3):
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
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, origin_transform.location.z+1])
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
    x_axis_color = carla.Color(255, 0, 0)
    y_axis_color = carla.Color(0, 255, 0)
    z_axis_color = carla.Color(0, 0, 255)

    # axis feature
    # thickness = 0.1f
    # arrow_size = 0.1f

    # draw x axis
    world.debug.draw_arrow(Origin_location, x_des, color=x_axis_color)
    # draw y axis
    world.debug.draw_arrow(Origin_location, y_des, color=y_axis_color)
    # draw z axis
    world.debug.draw_arrow(Origin_location, z_des, color=z_axis_color)

    # draw axis text next to arrow
    offset = 0.5
    x_text = carla.Location(x_des_coord[0]+offset, x_des_coord[1]+offset, x_des_coord[2]+offset)
    y_text = carla.Location(y_des_coord[0]+offset, y_des_coord[1]+offset, y_des_coord[2]+offset)
    z_text = carla.Location(z_des_coord[0]+offset, z_des_coord[1]+offset, z_des_coord[2]+offset)
    world.debug.draw_string(x_text, text='x', color=x_axis_color)
    world.debug.draw_string(y_text, text='y', color=x_axis_color)
    world.debug.draw_string(z_text, text='z', color=x_axis_color)



# def set_spectator_static(world, transform):
#     """
#         Set the spectator at a static view.
#     :param world: current world
#     :param transform: desired location and view
#     :return:
#     """
#
#
#     # behind distance - d, height - h
#     d = 6.4
#     h = 2.0
#
#     ego_trans = vehicle.get_transform()
#     angle = ego_trans.rotation.yaw
#     a = math.radians(180 + angle)
#     location = carla.Location(d * math.cos(a), d * math.sin(a), h) + ego_trans.location
#     # set spector
#     spectator = world.get_spectator()
#     spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15)))
#     # print("spectator is set on ", vehicle.attributes['role_name'])



def set_spectator(vehicle):
    """
        Plot frame at a specified waypoint
    Set spectator for a specified actor(vehicle)

    :param vehicle: selected actor
    :return: None
    """
    world = vehicle.get_world()
    # parameters for spectator:
    # behind distance - d, height - h
    d = 6.4
    h = 2.0

    ego_trans = vehicle.get_transform()
    angle = ego_trans.rotation.yaw
    a = math.radians(180 + angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), h) + ego_trans.location
    # set spector
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15)))
    # print("spectator is set on ", vehicle.attributes['role_name'])


class BrakingDistanceMeasure(object):
    """
        Measure braking distance at different speed of vehicle Lincoln MKz

        This class is used to package some methods.

    """
    def __init__(self):



        self.world = None


def delete_vehicles(world):
    """
        delete all vehicles in selected world
    """
    actors = world.get_actors()
    for actor in actors:
        # if actor.type_id == vehicle.*
        id = actor.type_id
        actor_type = id.split(".")

        # destroy vehicle
        if actor_type[0] == "vehicle":
            actor.destroy()
            print("vehicles are deleted")

    # sometimes need to tick the world to update
    # world.tick()

def get_speed(vehicle):
    """
        Get current speed of vehicle, scalar quantity.
    :param vehicle: The vehicle
    :return: Scalar speed of current velocity
    """
    vel = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    return speed

def main():
    actor_list = []

    try:
        # set up world for running
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        # world = client.get_world()

        # set map
        # todo: verify if this method is stable to load map
        town = 'Town04'
        try :
            world = client.load_world(town)
        except:
            while world.get_map() != town:
                world = client.load_world(town)
        # print('done')

        # clear vehicles
        delete_vehicles(world)

        # ==================================================
        """
        # carla timestamp test
        world_snapshot = world.get_snapshot()
        timestamp_1 = world_snapshot.timestamp

        # time.sleep(1.0)

        world_snapshot = world.get_snapshot()
        timestamp_2 = world_snapshot.timestamp
        # print('d')
        """
        # ==================================================

        # world parameters
        frame_rate = 20.0
        # set worldsettings
        settings = world.get_settings()
        settings.fixed_delta_seconds = 1.0 / frame_rate
        world.apply_settings(settings)

        # set weather
        # use ClearNoon
        weather = carla.WeatherParameters.ClearNoon
        world.set_weather(weather)

        # get the origin of coordinate frame
        """
            Backup locations:
                After the traffic light:     (x=-351, y=33.5, z=0.5)
                Before the traffic light:    (x=-390, y=33.6, z=0.5)

            Other side of the highway:
                (x=338, y=14, z=0.5) Rotation(pitch=1.194088, yaw=-179.022064, roll=0.000000)
                end point: (x=-363, y=9.0, z=0.5)

        """


        # initial spawn location
        start_location = carla.Location(x=338, y=14, z=0.5)
        start_waypoint = world.get_map().get_waypoint(start_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        end_location = carla.Location(x=-361, y=9.0, z=0.5)
        end_waypoint = world.get_map().get_waypoint(end_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # setup vehicle
        blueprint_library = world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        bp.set_attribute('role_name', 'hero')

        # stop location
        stop_location = carla.Location(x=70, y=9.5, z=0.5)
        stop_waypoint = world.get_map().get_waypoint(stop_location, project_to_road=True,
                                                    lane_type=carla.LaneType.Driving)

        ego_vehicle = world.try_spawn_actor(bp, stop_waypoint.transform)
        world.tick()

        # test lane
        lane_width = start_waypoint.lane_width

        left_waypoint = start_waypoint.get_left_lane()
        right_waypoint = start_waypoint.get_right_lane()

        left_dist = start_location.distance(left_waypoint.transform.location)
        right_dist = start_location.distance(right_waypoint.transform.location)


        a = [1, 3, 5, None]
        # b = np.array(a)
        # mean = np.mean(b)
        b = [x for x in a if x is not None]



        # flag test
        a = False
        b = False
        judge = not a and not b

        print('d')

        # spawwn actor
        ego_vehicle = world.try_spawn_actor(bp, start_waypoint.transform)
        # vehicle_2 = world.try_spawn_actor(bp, end_waypoint.transform)
        world.tick() # necessary to tick to update the world

        # set initial control to keep vehicle static
        control = ego_vehicle.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        ego_vehicle.apply_control(control)

        # ttt = ego_vehicle.get_transform()
        # set_spectator(ego_vehicle)
        # set_spectator(vehicle_2)

        # generate a route directly from start waypoint and end waypoint

        # parameters of route
        hop_resolution = 1.0

        dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        route = []
        trace = grp.trace_route(location, end_location)
        # for wp_tuple in trace:
        #     route.append((wp_tuple[0].transform, wp_tuple[1]))

        #
        local_planner = LocalPlanner(ego_vehicle)
        # set route
        local_planner.set_global_plan(trace)

        # set an overview spectator
        static_spectator_location = carla.Location(x=380, y=14, z=50)
        static_spectator_rotation = carla.Rotation(pitch=-35, yaw=-180, roll=0)
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(static_spectator_location, static_spectator_rotation))

        braking_distance_list = []
        for i in range(9):
            # target_speed = 10 + i*10
            target_speed = 100
            local_planner._target_speed = target_speed
            print('target speed is ', target_speed)

            # get initial speed
            speed = get_speed(ego_vehicle)
            max_speed = speed

            while speed <= target_speed:
                # control = local_planner.run_step()
                # control.steer = np.clip(control.steer, -0.03, 0.03)
                # if speed >= 90:
                #     control.steer = np.clip(control.steer, -0.005, 0.005)

                # without lateral control
                control.steer = 0.0
                control.throttle = 1.0
                control.brake = 0.0
                # apply control
                ego_vehicle.apply_control(control)

                # check control command
                control = ego_vehicle.get_control()
                # print('throttle: ', control.throttle)
                # print('brake: ', control.brake)
                # print('steer: ', control.steer)
                # print('\n')
                # update speed
                speed = get_speed(ego_vehicle)
                # print('speed: ', speed, 'km/h')
                # print('========================================')

                # if slides or drifting
                # if speed < max_speed-3:
                #     break

                # store max speed
                if speed >= max_speed:
                    max_speed = speed

                # record time
                # world_snapshot = world.get_snapshot()
                # timestamp = world_snapshot.timestamp

                # print('frame: ', timestamp.frame)
                # print()

                # print('========================================')

            # store max speed location
            max_speed_location = ego_vehicle.get_location()
            print('max speed reached: ', max_speed)


            # brake to stop
            while speed > 0:
                # braking
                control.throttle = 0.0
                control.brake = 1.0
                ego_vehicle.apply_control(control)

                # get speed
                speed = get_speed(ego_vehicle)
                # print('speed: ', speed, 'km/h')

            if speed == 0:
                stop_location = ego_vehicle.get_location()
                braking_distance = max_speed_location.distance(stop_location)

                # store result
                result = {'target_speed': target_speed, 'max_speed': max_speed, 'braking_distance': braking_distance}
                braking_distance_list.append(result)


                print('==================================================')
                print('Test finished.')
                print('braking distance: ', braking_distance)
                print('==================================================')

        # print(braking_distance_list)

        time.sleep(5)

    finally:

        # print('destroying actors')
        # for actor in actor_list:
        #     actor.destroy()
        print('test done.')


if __name__ == '__main__':

    main()





# Measure distance
# def measure_distance():
#
#     ego_vehicle_velocity = vehicle.get_velocity()
#
#     if velocity <=
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     return distance


#
# class MeasureBrakingDistance(object):
#     """
#         A class for measuring braking distance of different velocity.
#
#         Default vehicle model is Lincoln MKz
#     """
#
#     waypoint_list = []
#
#     def __init__(self):
#
#
#         # set up world for running
#         client = carla.Client('localhost', 2000)
#         client.set_timeout(2.0)
#
#         # world initialization
#         town = 'Town04'
#
#         self.world = self.client.load_world(config.town)
#         # apply world setting
#         settings = self.world.get_settings()
#         settings.fixed_delta_seconds = 1.0 / self.frame_rate
#         self.world.apply_settings(settings)
#
#         # Once we have a client we can retrieve the world that is currently
#         # running.
#         world = client.get_world()
#
#         self.client = None
#         self.world = None
#
#         self.ego_vehicle = None
#         self.BrakingDistance_list = []
#
#     def set_ego_vehicle(self, waypoint):
#         """
#
#         :param waypoint:
#         :return:
#         """
#
#         # spawn mkz
#         blueprint_library = self.world.get_blueprint_library()
#         bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
#         bp.set_attribute('role_name', 'hero')
#         waypoint
#
#     def load_world(self):
#
#
#         world
#






























