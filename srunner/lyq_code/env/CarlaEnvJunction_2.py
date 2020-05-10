"""
Modified from CarlaEnv_Junction.

A developing env to train RL agent for junction scenario.

log 2020.05.07
Add traffic flow module.

"""

from __future__ import print_function

import glob
import os
import sys

# using carla 095
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'  # carla egg

# using carla 098
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

import argparse
from argparse import RawTextHelpFormatter
import sys
import os
import traceback
import datetime
import time
import importlib
import math
import numpy as np
import xml.etree.ElementTree as ET

import py_trees

from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# get waypoint after a intersection
from srunner.tools.scenario_helper import generate_target_waypoint

import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.autoagents.autonomous_agent import Track
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutReader, ObjectFinder
from srunner.challenge.envs.sensor_interface import CallBack, CANBusSensor, HDMapReader
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.tools.config_parser import ActorConfiguration, ScenarioConfiguration, ActorConfigurationData
# from srunner.scenarios.master_scenario import MasterScenario
from srunner.scenarios.scenario_development.master_scenario_modified import MasterScenario  # use modified master_scenario

# scenarios from scenario_runner repo
from srunner.scenarios.background_activity import BackgroundActivity
from srunner.scenarios.trafficlight_scenario import TrafficLightScenario
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.background_activity import BackgroundActivity
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection

from srunner.scenariomanager.traffic_events import TrafficEventType

from srunner.challenge.utils.route_manipulation import interpolate_trajectory, clean_route

# ==================================================
# developing module

from srunner.lyq_code.env.env_module.TrafficFlow import TrafficFlow  # class name same as py script

# original
# from srunner.challenge.autoagents.RLagent import RLAgent
# from srunner.challenge.algorithm.dqn import DQNAlgorithm

# from srunner.challenge.algorithm.dqn_per import DQN_PERAlgorithm

# test agent
# from srunner.challenge.autoagents.RLagent_test import RLAgent

# self defined util
from srunner.util_development.util import *
from srunner.util_development.util import coords_trans
# easy test


# from srunner.challenge.autoagents.RLagent_junction import RLAgent
# from srunner.challenge.autoagents.RLagent_lon import RLAgent
from srunner.lyq_code.agent.RLagent_lon_2 import RLAgent  # newest version of agent

# from srunner.challenge.algorithm.dqn import DQNAlgorithm
# from srunner.challenge.algorithm.dqn_lon import DQNAlgorithm
from srunner.lyq_code.algorithm.dqn_lon_2 import DQNAlgorithm

number_class_translation = {

    "Scenario1": [ControlLoss],
    "Scenario2": [FollowLeadingVehicle],
    "Scenario3": [DynamicObjectCrossing],
    "Scenario4": [VehicleTurningRight, VehicleTurningLeft],
    "Scenario5": [OtherLeadingVehicle],
    "Scenario6": [ManeuverOppositeDirection],
    "Scenario7": [OppositeVehicleRunningRedLight],
    "Scenario8": [SignalizedJunctionLeftTurn],
    "Scenario9": [SignalizedJunctionRightTurn],
    "Scenario10": [NoSignalJunctionCrossing]
    # "Scenario11": [NoSignalJunctionCrossing]

}

# paras for scenario training
PENALTY_COLLISION_STATIC = 5
PENALTY_COLLISION_VEHICLE = 5
PENALTY_COLLISION_PEDESTRIAN = 5
PENALTY_TRAFFIC_LIGHT = 3
PENALTY_WRONG_WAY = 5
PENALTY_SIDEWALK_INVASION = 5
PENALTY_ROUTE_DEVIATION = 5
PENALTY_STOP = 2

# parameters for scenario
start_location = carla.Location(x=53.0, y=128.0, z=3.0)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)
# junction center
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

def convert_json_to_actor(actor_dict):
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfiguration(node)

def convert_transform_to_location(transform_vec):
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec

def convert_json_to_transform(actor_dict):
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))

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

# paras for drl training
EPISODES = 1

# default junction right turn scenario paras
scenario_para_dict = {
    'map': "Town03",
    'start_point': np.array([53.0, 128.0, 1.0]),
    'end_point': np.array([5.24, 92.28, 1.0]),
}


class ScenarioEnv(object):
    """
    Provisional code to evaluate AutonomousAgent performance
    """
    MAX_ALLOWED_RADIUS_SENSOR = 5.0
    SECONDS_GIVEN_PER_METERS = 0.25
    MAX_CONNECTION_ATTEMPTS = 5

    # route info of ego vehicle
    starting_location = carla.Location(x=53.0, y=128.0, z=0.0)
    ending_location = carla.Location(x=5.24, y=92.28, z=1.0)

    def __init__(self, args):

        host = 'localhost'
        port = 2000
        client_timeout = 30.0


        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)

        town = 'Town03'

        self.world = self.client.load_world(town)

        # todo: package into method

        sync_mode = True  # have to use sync mode
        frame_rate = 25.0
        no_render_mode = False
        settings = self.world.get_settings()
        # world settings parameters
        settings.fixed_delta_seconds = 1.0 / frame_rate
        settings.no_rendering_mode = no_render_mode
        settings.synchronous_mode = sync_mode  # set world sync mode
        self.world.apply_settings(settings)

        # check settings
        settings = self.world.get_settings()

        self.map = self.world.get_map()

        self.track = args.track

        self.debug = args.debug


        self.ego_vehicle = None
        self._system_error = False
        self.actors = []

        # Tunable parameters
        self.client_timeout = 30.0  # in seconds
        self.wait_for_world = 20.0  # in seconds

        # CARLA world and scenario handlers
        self.agent_instance = None

        # scenarios
        self.master_scenario = None
        self.background_scenario = None
        self.list_scenarios = []


        self._sensors_list = []
        self._hop_resolution = 2.0
        self.timestamp = None

        # debugging parameters
        self.route_visible = self.debug > 0


        self.config = args.config

        # For debugging
        self.route_visible = self.debug > 0

        # Try to load the world and start recording
        # If not successful stop recording and continue with next iteration


        # get some necessary API
        # self.debug = self.world.debug

        self.spectator = self.world.get_spectator()

        # set traffic manager
        self.traffic_manager = self.client.get_trafficmanager()


        #  get route
        # self.starting_point = args.starting
        # self.ending_point = args.ending

        # route calculate
        # self.route is a list of tuple(carla.Waypoint.transform, RoadOption)
        # todo: package into class
        self.gps_route, self.route = self.calculate_route()
        self.get_junction_frame()

        self.route_length = 0.0
        self.route_timeout = self.estimate_route_timeout()

        # agent algorithm
        self.agent_algorithm = None
        self.last_step_dist = self.route_length

        # trafficflow module
        self.trafficflow = TrafficFlow(self.client, self.world)

    def reach_ending_point(self):
        return False

    def get_junction_frame(self, turn_flag=1):
        """
        Get local coordinate frame using generate waypoint.
        todo: add attributes to init
        :return: local
        """
        start_waypoint = self.map.get_waypoint(self.starting_location)
        self.junction_origin = generate_target_waypoint(start_waypoint, turn_flag)  # carla.waypoint
        yaw = np.deg2rad(self.junction_origin.transform.rotation.yaw)
        [c_yaw, s_yaw] = [np.cos(yaw), np.sin(yaw)]
        self.transform_matrix = np.array([[c_yaw, s_yaw, 0],
                                         [-s_yaw, c_yaw, 0],
                                          [0, 0, 1]])

    def get_geometry_state(self):
        """
        Get some geometry state in local frame
        :return:
        """
        # ego vehicle
        # location
        ego_location = self.ego_vehicle.get_location()
        ego_location = np.array([ego_location.x, ego_location.y, ego_location.z])  # in global
        ego_location = np.matmul(self.transform_matrix, ego_location)  # in local frame
        # velocity
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_velocity = np.array([ego_velocity.x, ego_velocity.y, ego_velocity.z])  # in global
        ego_velocity = np.matmul(self.transform_matrix, ego_velocity)  # in local frame
        ego_speed = np.linalg.norm(ego_velocity)
        ego_velocity_norm = ego_velocity / ego_speed

        # npc vehicle
        npc_location = self.npc_vehicle.get_location()
        npc_location = np.array([npc_location.x, npc_location.y, npc_location.z])  # in global
        npc_location = np.matmul(self.transform_matrix, npc_location)  # in local frame

        npc_velocity = self.npc_vehicle.get_velocity()
        npc_velocity = np.array([npc_velocity.x, npc_velocity.y, npc_velocity.z])  # in global
        npc_velocity = np.matmul(self.transform_matrix, npc_velocity)  # in local frame
        npc_speed = np.linalg.norm(npc_velocity)
        npc_velocity_norm = npc_velocity / npc_speed

        # todo: normalization
        # maximum_distance =


    def cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """
        # We need enumerate here, otherwise the actors are not properly removed
        if hasattr(self, '_sensors_list'):
            for i, _ in enumerate(self._sensors_list):
                if self._sensors_list[i] is not None:
                    self._sensors_list[i].stop()
                    self._sensors_list[i].destroy()
                    self._sensors_list[i] = None
            self._sensors_list = []

        for i, _ in enumerate(self.actors):
            if self.actors[i] is not None:
                self.actors[i].destroy()
                self.actors[i] = None
        self.actors = []

        CarlaActorPool.cleanup()
        CarlaDataProvider.cleanup()

        if ego and self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'cleanup'):
            self.cleanup(True)
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

                self.world = None

    def prepare_ego_car(self, start_transform):
        """
        Spawn or update all scenario actors according to
        a certain start position.
        """

        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        if self.ego_vehicle is None:
            # TODO: the model is now hardcoded but that can change in a future.
            self.ego_vehicle = CarlaActorPool.request_new_actor('vehicle.lincoln.mkz2017', start_transform, hero=True)
            # setup sensors
            if self.agent_instance is not None:
                self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicle)
            self.ego_vehicle.set_transform(start_transform)

        # wait for the ego_car to land
        while self.ego_vehicle.get_transform().location.z > start_transform.location.z:
            self.world.tick()
            # self.world.wait_for_tick(self.wait_for_world)

    # original calculate route
    # def calculate_route(self):
    #     """
    #     This function calculate a route for giving starting_point and ending_point
    #     :return: route (includeing Waypoint.transform & RoadOption)
    #     """
    #     starting_location = carla.Location(x=float(self.starting_point.split("_")[0]),
    #                                        y=float(self.starting_point.split("_")[1]),
    #                                        z=float(self.starting_point.split("_")[2]))
    #
    #     ending_location = carla.Location(x=float(self.ending_point.split("_")[0]),
    #                                      y=float(self.ending_point.split("_")[1]),
    #                                      z=float(self.ending_point.split("_")[2]))
    #
    #     # returns list of (carla.Waypoint.transform, RoadOption) from origin to destination
    #     coarse_route = []
    #     coarse_route.append(starting_location)
    #     coarse_route.append(ending_location)
    #
    #     return interpolate_trajectory(self.world, coarse_route)

    def calculate_route(self):
        """
        Modified input format, using list to store coords now
        This function calculate a route for giving starting_point and ending_point
        :return: route (includeing Waypoint.transform & RoadOption)
        """
        # input args is a coarse route of carla.Location
        coarse_route = []
        coarse_route.append(self.starting_location)
        coarse_route.append(self.ending_location)
        # returns list of (carla.Waypoint.transform, RoadOption) from origin to destination
        gps_route, trajectory = interpolate_trajectory(self.world, coarse_route)

        # ==================================================
        # # test route
        # plot_route(self.world, trajectory)
        # # set spectator
        # map = self.world.get_map()
        # waypoint = map.get_waypoint(self.starting_location)
        # start_transform = waypoint.transform
        # set_spectator(self.world, start_transform, 1)
        # ==================================================

        return gps_route, trajectory

    def draw_waypoints(self, waypoints, turn_positions_and_labels, vertical_shift, persistency=-1.0):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        :param waypoints: list or iterable container with the waypoints to draw
        :param vertical_shift: height in meters
        :return:
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)
            self.world.debug.draw_point(wp, size=0.1, color=carla.Color(0, 255, 0), life_time=persistency)
        for start, end, conditions in turn_positions_and_labels:

            if conditions == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif conditions == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif conditions == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif conditions == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            else:  # STRAIGHT
                color = carla.Color(128, 128, 128)  # Gray

            for position in range(start, end):
                self.world.debug.draw_point(waypoints[position][0].location + carla.Location(z=vertical_shift),
                                            size=0.2, color=color, life_time=persistency)

        self.world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                                    color=carla.Color(0, 0, 255), life_time=persistency)
        self.world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                                    color=carla.Color(255, 0, 0), life_time=persistency)

    def compute_current_statistics(self):

        target_reached = False
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0

        list_traffic_events = []
        for node in self.master_scenario.scenario.test_criteria.children:
            if node.list_traffic_events:
                list_traffic_events.extend(node.list_traffic_events)

        list_collisions = []
        list_red_lights = []
        list_wrong_way = []
        list_route_dev = []
        list_sidewalk_inv = []
        list_stop_inf = []
        # analyze all traffic events
        for event in list_traffic_events:
            if event.get_type() == TrafficEventType.COLLISION_STATIC:
                score_penalty += PENALTY_COLLISION_STATIC
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                score_penalty += PENALTY_COLLISION_VEHICLE
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                score_penalty += PENALTY_COLLISION_PEDESTRIAN
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                score_penalty += PENALTY_TRAFFIC_LIGHT
                msg = event.get_message()
                if msg:
                    list_red_lights.append(event.get_message())

            elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                score_penalty += PENALTY_WRONG_WAY
                msg = event.get_message()
                if msg:
                    list_wrong_way.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                score_penalty += PENALTY_ROUTE_DEVIATION
                msg = event.get_message()
                if msg:
                    list_route_dev.append(event.get_message())

            elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                score_penalty += PENALTY_SIDEWALK_INVASION
                msg = event.get_message()
                if msg:
                    list_sidewalk_inv.append(event.get_message())

            elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                score_penalty += PENALTY_STOP
                msg = event.get_message()
                if msg:
                    list_stop_inf.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                score_route = 50.0
                target_reached = True
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                if not target_reached:
                    # if event.get_dict():
                    #     score_route = event.get_dict()['route_completed']
                    # else:
                    #     # 计算前进距离
                    #     # self.route is a list of tuple(carla.Waypoint.transform, RoadOption)

                    endpoint_dist = self.route[-1][0].location.distance(self.ego_vehicle.get_transform().location)
                    print('endpoint_dist:', endpoint_dist)
                    print('last_step_dist:', self.last_step_dist)
                    rundist = self.last_step_dist - endpoint_dist
                    print('rundist:', rundist)
                    ang = abs(self.calculate_angle())
                    print('angle:', ang)
                    if rundist > 0:
                        score_route = (rundist / self.route_length) * 2000
                    else:
                        score_route = -5
                    self.last_step_dist = endpoint_dist

            score_penalty += 2 * (ang / 90)

        # score_composed = max(score_route - score_penalty, 0.0)
        score_composed = score_route - score_penalty
        print('score_route:', score_route)
        print('score_penalty:', score_penalty)

        return score_composed, score_route, score_penalty

    def calculate_angle(self):
        # calculate angle between local waypoint direction
        map = CarlaDataProvider.get_map()
        lane_waypoint = map.get_waypoint(self.ego_vehicle.get_location())
        next_waypoint = lane_waypoint.next(2.0)[0]

        vector_wp = np.array([next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                              next_waypoint.transform.location.y - lane_waypoint.transform.location.y])

        vector_actor = np.array([math.cos(math.radians(self.ego_vehicle.get_transform().rotation.yaw)),
                                 math.sin(math.radians(self.ego_vehicle.get_transform().rotation.yaw))])

        ang = math.degrees(
            math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0)))

        return ang

    def setup_sensors(self, sensors, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param sensors: list of sensors
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in sensors:
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.scene_layout'):
                # Static sensor that gives you the entire information from the world (Just runs once)
                sensor = SceneLayoutReader(self.world)
            elif sensor_spec['type'].startswith('sensor.object_finder'):
                # This sensor returns the position of the dynamic objects in the scene.
                sensor = ObjectFinder(self.world, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.can_bus'):
                # The speedometer pseudo sensor is created directly here
                sensor = CANBusSensor(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.hd_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = HDMapReader(vehicle, sensor_spec['reading_frequency'])
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', '5000')
                    bp.set_attribute('rotation_frequency', '20')
                    bp.set_attribute('channels', '32')
                    bp.set_attribute('upper_fov', '15')
                    bp.set_attribute('lower_fov', '-30')
                    bp.set_attribute('points_per_second', '500000')
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp, sensor_transform,
                                                vehicle)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor, self.agent_instance.sensor_interface))
            self._sensors_list.append(sensor)

        # check that all sensors have initialized their data structure

        while not self.agent_instance.all_sensors_ready():
            if self.debug > 0:
                print(" waiting for one data reading from sensors...")
            self.world.tick()
            # self.world.wait_for_tick(self.wait_for_world)

    def get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    def build_master_scenario(self, route, town_name, timeout=300):
        # We have to find the target.
        # we also have to convert the route to the expected format
        master_scenario_configuration = ScenarioConfiguration()
        master_scenario_configuration.target = route[-1][0]  # Take the last point and add as target.
        master_scenario_configuration.route = convert_transform_to_location(route)
        master_scenario_configuration.town = town_name
        # TODO THIS NAME IS BIT WEIRD SINCE THE EGO VEHICLE  IS ALREADY THERE, IT IS MORE ABOUT THE TRANSFORM
        master_scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                           self.ego_vehicle.get_transform())
        master_scenario_configuration.trigger_point = self.ego_vehicle.get_transform()
        CarlaDataProvider.register_actor(self.ego_vehicle)

        # Provide an initial blackboard entry to ensure all scenarios are running correctly
        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set('master_scenario_command', 'scenarios_running')

        return MasterScenario(self.world, self.ego_vehicle, master_scenario_configuration, timeout=timeout,
                              debug_mode=self.debug > 1)

    def build_background_scenario(self, town_name, timeout=300):
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.route = None
        scenario_configuration.town = town_name

        model = 'vehicle.audi.*'
        transform = carla.Transform()
        autopilot = True
        random = True

        if town_name == 'Town01' or town_name == 'Town02':
            amount = 120
        elif town_name == 'Town03' or town_name == 'Town05':
            amount = 120
        elif town_name == 'Town04':
            amount = 300
        elif town_name == 'Town06' or town_name == 'Town07':
            amount = 150
        elif town_name == 'Town08':
            amount = 180
        elif town_name == 'Town09':
            amount = 350
        else:
            amount = 1

        actor_configuration_instance = ActorConfigurationData(model, transform, autopilot, random, amount)
        scenario_configuration.other_actors = [actor_configuration_instance]

        return BackgroundActivity(self.world, self.ego_vehicle, scenario_configuration,
                                  timeout=timeout, debug_mode=self.debug > 1)

    def build_trafficlight_scenario(self, town_name, timeout=300):
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.route = None
        scenario_configuration.town = town_name

        return TrafficLightScenario(self.world, self.ego_vehicle, scenario_configuration,
                                    timeout=timeout, debug_mode=self.debug > 1)

    def build_scenario_instances(self, scenario_definition, town_name, timeout=300):
        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        :param scenario_definition_vec: the dictionary defining the scenarios
        :param town: the town where scenarios are going to be
        :return:
        """
        scenario_instance_vec = []

        # for definition in scenario_definition_vec:
        # Get the class possibilities for this scenario number
        # possibility_vec = number_class_translation[definition['scenario_type']]

        # ScenarioClass = possibility_vec[definition['type']]

        ScenarioClass = number_class_translation[scenario_definition['scenario_type']]
        ScenarioClass = ScenarioClass[0]
        Config = scenario_definition['available_event_configurations'][0]
        # Create the other actors that are going to appear
        if 'other_actors' in Config:
            list_of_actor_conf_instances = self.get_actors_instances(Config['other_actors'])
        else:
            list_of_actor_conf_instances = []
        # Create an actor configuration for the ego-vehicle trigger position

        egoactor_trigger_position = convert_json_to_transform(Config['transform'])
        print(egoactor_trigger_position)
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.other_actors = list_of_actor_conf_instances
        scenario_configuration.town = town_name
        scenario_configuration.trigger_point = egoactor_trigger_position
        scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                    self.ego_vehicle.get_transform())
        try:
            scenario_instance = ScenarioClass(self.world, self.ego_vehicle, scenario_configuration,
                                              criteria_enable=False, timeout=timeout)
        except Exception as e:
            if self.debug > 1:
                raise e
            else:
                print("Skipping scenario due to setup error: {}".format(e))
                # continue
        # registering the used actors on the data provider so they can be updated.

        CarlaDataProvider.register_actors(scenario_instance.other_actors)

        scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def estimate_route_timeout(self):
        self.route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            self.route_length += dist
            prev_point = current_point

        return int(self.SECONDS_GIVEN_PER_METERS * self.route_length)

    def route_is_running(self):
        """
            Test if the route is still running.
        """
        if self.master_scenario is None:
            raise ValueError('You should not run a route without a master scenario')

        # The scenario status can be: INVALID, RUNNING, SUCCESS, FAILURE. Only the last two
        # indiciate that the scenario was running but terminated
        # Therefore, return true when status is INVALID or RUNNING, false otherwise
        if (self.master_scenario.scenario.scenario_tree.status == py_trees.common.Status.RUNNING or
                self.master_scenario.scenario.scenario_tree.status == py_trees.common.Status.INVALID):
            return True
        else:
            return False

    def set_spectator(self, view=0):
        """
        Set spectator at different view

        todo: different options
        """
        if view == 0:
            # print("Set overhead view on junction.")
            # height = h
            height = 100
            location = carla.Location(0, 0, height) + junction_center
            rotation = carla.Rotation(yaw=start_rotation.yaw, pitch=-90)  # rotate to forward direction
        elif view == 1:
            print("Set behind view on vehicle.")
            _d = 8  # behind distance
            _h = 8  # height
            transform = self.ego_vehicle.get_transform()
            angle = transform.rotation.yaw
            a = math.radians(180 + angle)
            location = carla.Location(x=_d * math.cos(a), y=_d * math.sin(a), z=_h) + transform.location
            rotation = carla.Rotation(yaw=angle, pitch=-15)
        elif view == 2:
            print("Set overhead view on vehicle.")
            # height = h
            height = 30
            transform = self.ego_vehicle.get_transform()
            location = carla.Location(0, 0, height) + transform.location
            rotation = self.map.get_waypoint(transform.location).transform.rotation
            rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction

        self.spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()

    def worldReset(self, episode_index):
        """
        Reset the world
        """
        # prepare the ego car to run the route.
        # It starts on the first wp of the route
        self.list_scenarios = []
        self.cleanup(True)

        # Set the actor pool so the scenarios can prepare themselves when needed
        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        # Also se the Data provider pool.
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        # # create agent
        # self.agent_instance = getattr(self.module_agent, self.module_agent.__name__)(self.config)
        # self.agent_instance.set_global_plan(self.gps_route, self.route)

        # # creat RLagent
        self.agent_instance = RLAgent(self.config, episode_index)
        self.agent_instance.set_global_plan(self.gps_route, self.route)

        # # creat Algorithm
        if self.agent_algorithm is None:
            # original
            # self.agent_algorithm = DQNAlgorithm(self.agent_instance.get_image_shape(),
            #                                     self.agent_instance.get_action_shape())

            # easy version test
            self.agent_algorithm = DQNAlgorithm(4, self.agent_instance.get_action_shape())
            # self.agent_algorithm.load_net()

            # use a flag to decide to train/run
            finetune = 0  # default is to train without previous weight
            try:
                if finetune:
                    # train without weights
                    self.agent_algorithm.load_net()
            except:
                print("Fail to load weight.")

        # set algorithm module to agent instance
        self.agent_instance.set_algorithm(self.agent_algorithm)
        self.agent_instance.algorithm.change_rate(episode_index)

        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 1.0
        self.prepare_ego_car(elevate_transform)

        # ==================================================
        # API
        # add attribute to agent
        # set ego_vehicle to agentset_lateral_controller
        self.agent_instance.set_ego_vehicle(self.ego_vehicle)
        # set world
        self.agent_instance.set_world(self.world)
        # set lateral controller
        self.agent_instance.set_lateral_controller()

        # test_ego_location = self.ego_vehicle.get_location()
        # ==================================================

        # set npc vehicle
        # self.set_npc_vehicle()


        # set spectator
        self.set_spectator()

    def run_route(self, trajectory, no_master=False):

        print('route is running', self.route_is_running())
        GameTime.restart()
        last_time = GameTime.get_time()
        steer = 0.0
        last_step_loaction_x = self.route[0][0].location.x
        last_step_loaction_y = self.route[0][0].location.y
        print('last_time:', last_time)

        # for debug RL lon
        # set ego vehicle once again
        # set ego_vehicle to agentset_lateral_controller
        self.agent_instance.set_ego_vehicle(self.ego_vehicle)
        # set world
        self.agent_instance.set_world(self.world)



        while no_master or self.route_is_running():
            self.timestamp = self.world.get_snapshot()
            # update all scenarios
            GameTime.on_carla_tick(self.timestamp)
            CarlaDataProvider.on_carla_tick()
            print("world timestamp: ", self.timestamp)

            # get geometry state for agent

            # update traffic flow
            self.trafficflow.run_step()

            # get action
            ego_action = self.agent_instance()
            # parse control action
            # throttle = ego_action
            steer = ego_action.steer

            # tick scenarios stored in list
            for scenario in self.list_scenarios:
                scenario.scenario.scenario_tree.tick_once()

            # print status
            # print("==================================================")
            # print("scenario status: ", self.master_scenario.scenario.scenario_tree.status)
            # print("==================================================")

            if self.debug > 1:
                for actor in self.world.get_actors():
                    if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                        print(actor.get_transform())

            # ego vehicle acts
            self.ego_vehicle.apply_control(ego_action)

            # set spectator on vehicle
            # if self.spectator:
            #     self.set_spectator_vehicle()



            # set ego_vehicle on static view
            # overhead of local junction
            # origin_transform =
            # set_spectator_location(self.world, origin_transform, view_mode=1)

            # show current score
            # 打分
            # 测试非训练框架时屏蔽此句
            total_score, route_score, infractions_score = self.compute_current_statistics()

            # send the reward to agent
            # 把reward传给agent保存
            # 测试非训练框架时屏蔽此句
            self.agent_instance.get_reward(total_score)

            if self.route_visible:
                turn_positions_and_labels = clean_route(trajectory)
                self.draw_waypoints(trajectory, turn_positions_and_labels,
                                    vertical_shift=1.0, persistency=50000.0)
                self.route_visible = False

            # time continues
            attempts = 0
            while attempts < self.MAX_CONNECTION_ATTEMPTS:
                try:
                    self.world.tick()
                    # self.timestamp = self.world.wait_for_tick(self.wait_for_world)
                    break
                except Exception:
                    attempts += 1
                    print('======[WARNING] The server is frozen [{}/{} attempts]!!'.format(attempts,
                                                                                           self.MAX_CONNECTION_ATTEMPTS))
                    time.sleep(2.0)
                    continue

        # learn
        # 更新Q网络
        # 测试非训练框架时屏蔽此句
        self.agent_instance.algorithm.update()
        # if self.agent_instance.algorithm.update():
        #     return

    def load_environment_and_run(self, args, world_annotations=''):


        # if not correct_sensors:
        # the sensor configuration is illegal
        #    sys.exit(-1)

        # build the master scenario based on the route and the target.

        # train for EPISODES times
        for i in range(EPISODES):
            self.worldReset(i)
            self.master_scenario = self.build_master_scenario(self.route, self.map.name, timeout=self.route_timeout)
            # self.background_scenario = self.build_background_scenario(self.map, timeout=self.route_timeout)
            # self.traffself.load_environment_and_run(argsenarios_definitions = world_annotations[self.map][0]
            self.list_scenarios = [self.master_scenario]

            # original
            # self.list_scenarios += self.build_scenario_instances(sampled_scenarios_definitions, self.map,timeout=self.route_timeout)

            # ==================================================
            # todo: add a module to add specified scenarios
            # ref on self.build_scenario_instances method
            # self.list_scenarios += self.build_scenario_instances(sampled_scenarios_definitions, self.map,timeout=self.route_timeout)

            # ==================================================

            # main loop!
            if self.run_route(self.route):
                return
            # self.world.tick()

        self.agent_algorithm.save_net()
        print("==================================================")
        print("Net is saved")
        print("==================================================")

    def run(self, args):
        """
        Run route according to provided commandline args
        """

        # tick world so we can start.
        # self.world.tick()

        # retrieve worlds annotations
        # world_annotations = parser.parse_annotations_file(args.scenarios)

        # Try to run the route
        # If something goes wrong, still take the current score, and continue
        try:
            # self.load_environment_and_run(args,world_annotations)
            self.load_environment_and_run(args)
        except Exception as e:
            if self.debug > 0:
                traceback.print_exc()
                raise
            if self._system_error or not self.agent_instance:
                print(e)
                sys.exit(-1)

        # clean up
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self.agent_instance.destroy()
        self.agent_instance = None
        self.cleanup(ego=True)

        for scenario in self.list_scenarios:
            # Reset scenario status for proper cleanup
            scenario.scenario.terminate()
            # Do not call del here! Directly enforce the actor removal
            scenario.remove_all_actors()
            scenario = None

        self.list_scenarios = []

        self.master_scenario = None
        self.background_scenario = None

    # 'start_point': np.array([53.0, 128.0, 1.0]),
    # 'end_point': np.array([5.24, 92.28, 0]),


if __name__ == '__main__':
    DESCRIPTION = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    PARSER = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    PARSER.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate")
    PARSER.add_argument("--config", type=str, help="Path to Agent's configuration file", default=" ")

    PARSER.add_argument('--track', type=int, help='track type', default=4)

    PARSER.add_argument('--debug', type=int, help='Run with debug output', default=0)

    ARGUMENTS = PARSER.parse_args()

    try:
        env = ScenarioEnv(ARGUMENTS)
        env.run(ARGUMENTS)
    except Exception as e:
        traceback.print_exc()
    finally:
        del env
