"""
Based on RLagent_lon
Action space consider only longitudinal control.

Using A* to generate a local route
Using local_planner(PID) to take over lateral control.

In developing: move some env methods into agent module,
i.e. generate route

2020.04.23
fixed state representation for junction scenario.

2020.03.26
Change state API, using different state representation.

"""
from __future__ import print_function

import glob
import os
import sys

# using carla 095
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'  # carla egg

# if using carla098
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

import numpy as np
import math
import torch
from itertools import product
from collections import namedtuple
from srunner.challenge.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.utils.route_manipulation import downsample_route
from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from PIL import Image

import heapq
from collections import deque

# import from util
from srunner.util_development.util import get_rotation_matrix_2D

# PID controller
from srunner.challenge.autoagents.agent_development.lat_control.controller_modified import PIDLateralController

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RLAgent:
    """
    Package methods of getting state and reward in RL training.
    """
    def __init__(self, world, episode_index):

        self.world = None  # need to manually set world
        self.ego_vehicle = None
        self.algorithm = None
        # current global plans to reach a destination
        self._global_plan = None
        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.action = None
        self.state = None
        self.next_state = None
        self.reward = None

        # action space:[accelerate, maintain, decelerate]
        self.action_space = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
        self.action_dim = len(self.action_space)

        # todo: calculate state dimension automaticlly
        self.state_dim = 20

        # image index???
        self.index = 0

        # todo: remove Track info
        # self.track = Track.CAMERAS
        self.track = 0

        # ==================================================

        self.episode_index = episode_index

        # ==================================================
        # waypoint buffer for navigation and state representation
        # waypoints queue
        self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # waypoint buffer
        self._buffer_size = 50
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # near waypoints is waypoint popped out from buffer
        self.near_waypoint_queue = deque(maxlen=50)

        # min distance threthold of waypoint reaching
        MIN_DISTANCE_PERCENTAGE = 1.5
        self._target_speed = 5.0  # m/s
        self._sampling_radius = self._target_speed * 1  # maximum distance vehicle move in 1 seconds
        self._min_distance = self._sampling_radius * MIN_DISTANCE_PERCENTAGE

        self.off_route = False

        # ==================================================
        # set lateral controller
        self.lateral_controller = None  # lateral controller will be set later by env script
        # self.set_lateral_controller()

        # near npc vehicle state
        self.near_npc_dict = None

    # ==================================================
    # interaction API
    # setters
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def set_ego_vehicle(self, ego_vehicle):
        """
        Get ego vehicle from env.
        """
        self.ego_vehicle = ego_vehicle

    def set_world(self, world):
        self.world = world

    # getters
    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_reward(self, reward):
        self.reward = reward
        print('reward:', self.reward)

    def get_near_npc(self, near_npc_dict):
        """
        Get dict contains near npc info from env.
        Structure of dict is contained in TrafficFlow module.
        """
        self.near_npc_dict = near_npc_dict

    # function module
    def set_lateral_controller(self, args_lateral=None):
        """
        Set controller for the ego vehicle
        Current is only lateral controller.
        """
        # set up PID controller
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': 0.001}

        # todo: check if get dict correctly
        # init controller
        if not args_lateral:
            args_lateral = args_lateral_dict

        self.lateral_controller = PIDLateralController(self.ego_vehicle, **args_lateral)

    def set_lon_controller(self):
        """
        Set a PID controller for longitudinal control.

        todo: finish this method
        """
        # self.lon_controller =
        pass

    def all_sensors_ready(self):
        return self.sensor_interface.all_sensors_ready()

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        sensors = [
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 300, 'height': 200, 'fov': 100, 'id': 'Mid'},
            # {'type':'sensor.camera.semantic_segmentation','x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 300, 'height': 200, 'fov': 100, 'id': 'Sem'}
        ]
        return sensors

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Get global plan from env.
        :param global_plan_gps: route in gps
        :param global_plan_world_coord: route in world coords
        """
        if self.track == Track.CAMERAS or self.track == Track.ALL_SENSORS:
            ds_ids = downsample_route(global_plan_world_coord, 32)

            self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                             for x in ds_ids]
            self._global_plan = [global_plan_gps[x] for x in ds_ids]

        else:  # No downsampling is performed

            self._global_plan = global_plan_gps
            self._global_plan_world_coord = global_plan_world_coord  # this is the final route to follow

        # ==================================================
        # get waypoints in queue for buffering
        self._waypoints_queue.clear()
        # for elem in self._global_plan_world_coord:
        for elem in global_plan_world_coord:
            self._waypoints_queue.append(elem)

        # print('debug global route')

    def buffer_waypoint(self):
        """
        Buffering waypoints for algorithm.

        :return: 2 nearest waypoint
        """
        # Buffering the waypoints

        # # original version
        # if not self._waypoint_buffer:
        #     for i in range(self._buffer_size):
        #         if self._waypoints_queue:
        #             self._waypoint_buffer.append(
        #                 self._waypoints_queue.popleft())
        #         else:
        #             break

        # check and buffer waypoints
        least_buffer_num = 3
        if len(self._waypoint_buffer) <= least_buffer_num:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            # if waypoint here is transform
            current_location = waypoint
            if waypoint.location.distance(self.ego_vehicle.get_location()) < self._min_distance:
                # if waypoint here is waypoint class
                # if waypoint.transform.location.distance(self.ego_vehicle.get_location()) < self._min_distance:
                max_index = i
        # supplyment more waypoints
        if max_index >= 0:
            for i in range(max_index + 1):  # max_index + 1 is the amount number of waypoints to pop out into buffer
                self.near_waypoint_queue.append(self._waypoint_buffer.popleft())

        # find 2 nearest waypoint in near_waypoint_queue
        distance_list = []
        for i, (waypoint, _) in enumerate(self.near_waypoint_queue):
            # distance = waypoint.transform.location.distance(self.ego_vehicle.get_location())
            distance = waypoint.location.distance(self.ego_vehicle.get_location())
            add_dict = {'index': i, 'waypoint': waypoint, 'distance': distance}
            distance_list.append(add_dict)

        # todo: check if distance is effective
        # distance between vehicle and waypoints should be smaller than original distance between waypoints
        # if distance_to_vehicle(self.ego_vehicle, waypoint.location) >= min_distance:
        #     pass

        # find 2 minimal distance waypoints, using heapq
        [next_waypoint, last_waypoint] = heapq.nsmallest(2, distance_list, key=lambda s: s['distance'])
        # todo: check correct order and plot
        if next_waypoint['index'] < last_waypoint['index']:
            cup = next_waypoint
            next_waypoint = last_waypoint
            last_waypoint = cup

        # extract waypoint location
        last_waypoint_location = last_waypoint['waypoint'].location
        next_waypoint_location = next_waypoint['waypoint'].location

        # plot local direction of plan
        last_waypoint_location.z = 2
        next_waypoint_location.z = 2
        debug = self.world.debug
        debug.draw_arrow(last_waypoint_location, next_waypoint_location, thickness=0.1, arrow_size=0.1,
                         color=carla.Color(0, 255, 0),
                         life_time=10)

        # draw point, tested
        # debug.draw_point(last_waypoint, size=0.15, color=carla.Color(255, 0, 0), life_time=1000)
        # debug.draw_point(last_waypoint, size=0.15, color=carla.Color(0, 255, 0), life_time=1000)
        print('local direction updated.')
        return last_waypoint_location, next_waypoint_location

    def get_navigation_state(self):
        """
        The State based on ground truth geometry and kinematics.

        Consider coord frame fixed with local lane.

        todo: add target waypoints from a coarse route as state

        :return: State vector
        """

        # set first waypoint as Origin
        # todo: fix origin selection
        Origin_transform = self._global_plan_world_coord[0][0]
        # get local transformation matrix
        # method <get_rotation_matrix_2D> is from util
        trans_matrix = get_rotation_matrix_2D(Origin_transform)

        # get ego location in local frame
        # 2D location coords
        self.ego_location = self.ego_vehicle.get_location()
        temp_vector = self.ego_location - Origin_transform.location
        ego_location_local = np.array([temp_vector.x, temp_vector.y])
        ego_location_local = np.matmul(trans_matrix, ego_location_local)

        # ego vehicle velocity
        velocity = self.ego_vehicle.get_velocity()
        velocity_2D = np.array([velocity.x, velocity.y])
        velocity_2D_local = np.matmul(trans_matrix, velocity_2D)

        # debug
        # waypoint buffer empty
        if not self._waypoint_buffer:
            print("waypoint buffer empty!")

        # offset respect to target waypoint
        # todo: local frame will be changed after intersection
        target_waypoint_location = self._waypoint_buffer[0][0].location
        temp_location = target_waypoint_location - self.ego_location  # this is a location class
        vector_ego_target_2D = np.array([temp_location.x, temp_location.y])  # with out transformation
        # transform into local frame
        lon_dist, lat_dist = np.matmul(trans_matrix, vector_ego_target_2D)
        # print("navigation state update")

        # visualization to target waypoint
        # plot_local_coordinate_frame(self.world, Origin_transform) # this method is stored in util
        # draw arrow from ego vehicle to target waypoint
        # todo: add debug as class attribute
        debug = self.world.debug
        # draw arrow at a higher location
        arrow_start = self.ego_location
        arrow_start.z = 2.0
        arrow_end = target_waypoint_location
        arrow_end.z = 2.0
        debug.draw_arrow(arrow_start, arrow_end,
                         thickness=0.1,
                         arrow_size=0.1,
                         color=carla.Color(255, 0, 0),
                         life_time=3)

        return lon_dist, lat_dist

    # @staticmethod
    def get_local_geometry_state(self, ego_transform, last_waypoint_location, next_waypoint_location):
        """
        Get contents of penalty reward.
        lateral offset, diversion angle
        :param ego_transform: transform of ego vehicle
        :param last_waypoint_location:
        :param next_waypoint_location:
        :return:
        """

        length_factor = 1

        ego_location = ego_transform.location

        A = np.array([last_waypoint_location.x, last_waypoint_location.y])
        B = np.array([next_waypoint_location.x, next_waypoint_location.y])
        E = np.array([ego_location.x, ego_location.y])

        Vector_AE = E - A
        Vector_AB = B - A

        # calculate lateral offset
        # in meters
        temp = Vector_AE.dot(Vector_AB) / Vector_AB.dot(Vector_AB)
        temp = temp * Vector_AB
        lat_offset = np.linalg.norm(Vector_AE - temp)

        # calculate diversion angle
        # in radians
        yaw = ego_transform.rotation.yaw
        ego_direction = [math.cos(math.radians(yaw)),
                         math.sin(math.radians(yaw))]
        # in radians
        diversion_angle = math.atan2(ego_direction[1], ego_direction[0]) - math.atan2(Vector_AB[1], Vector_AB[0])

        # normalization selection
        _if_normalization = False

        if _if_normalization:
            diversion_angle = diversion_angle / 0.5 / math.pi

        # check if invalid
        if diversion_angle < -0.5 * math.pi:
            print("Diversion angle < -90 degrees! Will be normalized")
            print("diversion_angle = ", math.degrees(diversion_angle))
            diversion_angle = -0.5 * math.pi
        elif diversion_angle > 0.5 * math.pi:
            print("Diversion angle > 90 degrees! Will be normalized")
            print("diversion_angle = ", math.degrees(diversion_angle))
            diversion_angle = 0.5 * math.pi

        # ==================================================
        # visualization
        debug = self.world.debug

        # ego direction in 2d
        start = np.array([ego_location.x, ego_location.y])
        end = start + ego_direction * length_factor

        # height of arrow plane
        h = 5
        arrow_start = carla.Location(x=start[0], y=start[1], z=h)
        arrow_end = carla.Location(x=end[0], y=end[1], z=h)

        # arrow_list.append([arrow_start, arrow_end])
        debug.draw_arrow(arrow_start, arrow_end,
                         thickness=0.1,
                         arrow_size=0.1,
                         color=carla.Color(255, 0, 0),
                         life_time=3)

        # draw direction vector of local route
        Vector_AB = Vector_AB / np.linalg.norm(Vector_AB)

        local_direction_end = A + Vector_AB * length_factor
        arrow_start = carla.Location(x=A[0], y=A[1], z=h)
        arrow_end = carla.Location(x=local_direction_end[0], y=local_direction_end[1], z=h)
        # arrow_list.append([arrow_start, arrow_end])
        debug.draw_arrow(arrow_start, arrow_end,
                         thickness=0.1,
                         arrow_size=0.1,
                         color=carla.Color(0, 255, 0),
                         life_time=3)

        return lat_offset, diversion_angle

    def get_state(self):
        """
        Get state for RL module.
        State consists of ego state and npc state.
        """
        # ego vehicle
        transform = self.ego_vehicle.get_transform()
        loc = [transform.location.x, transform.location.y, transform.rotation.yaw]
        velo = [self.ego_vehicle.get_velocity().x, self.ego_vehicle.get_velocity().y]
        ego_state = loc + velo

        # npc vehicle
        npc_state = {
            'left': [],
            'right': [],
            'straight': [],
        }

        for key in self.near_npc_dict:
            if self.near_npc_dict[key]:
                npc = self.near_npc_dict[key][0]

                loc = [npc.get_transform().location.x, npc.get_transform().location.y]
                rot = [npc.get_transform().rotation.yaw]
                velo = [npc.get_velocity().x, npc.get_velocity().y]

                # todo: normalization
                # norm_para = [18.0, 1.0, 30]  # meter, degrees, m/s
                # loc = loc / norm_para[0]
                # rot = rot / norm_para[1]
                # velo = velo / norm_para[2]

                state = loc + rot + velo

            else:
                # todo: this criterias needs to modified if scenario changes
                if key == 'left':
                    state = [5.76, 175.50] + [270.0] + [0.0, -5.0]
                elif key == 'straight':
                    state = [-6.26, 90.84] + [0.0] + [5.0, 0.0]
                elif key == 'right':
                    state = [-46.92, 135.03] + [90.0] + [0.0, 5.0]

            npc_state[key] = state

        print('NPC state', npc_state)
        # complete state
        state = ego_state + npc_state['left'] + npc_state['straight'] + npc_state['right']

        return state

    def buffer_waypoint2(self):
        """
        Buffering waypoints method 2.

        :return: 2 nearest waypoint
        """
        # dynamic distance threthold
        MIN_DISTANCE_PERCENTAGE = 1.
        d_t = .5
        vel = self.ego_vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        self._sampling_radius = speed * d_t  # maximum distance vehicle move in time step(in seconds)
        self._min_distance = self._sampling_radius * MIN_DISTANCE_PERCENTAGE

        # flag of if off route
        self.off_route = False

        # Buffering the waypoints

        # # original version
        # if not self._waypoint_buffer:
        #     for i in range(self._buffer_size):
        #         if self._waypoints_queue:
        #             self._waypoint_buffer.append(
        #                 self._waypoints_queue.popleft())
        #         else:
        #             break

        # check and buffer waypoints
        least_buffer_num = 10
        if len(self._waypoint_buffer) <= least_buffer_num:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            # if waypoint here is transform
            current_location = waypoint
            if waypoint.location.distance(self.ego_vehicle.get_location()) < self._min_distance:
                # if waypoint here is waypoint class
                # if waypoint.transform.location.distance(self.ego_vehicle.get_location()) < self._min_distance:
                max_index = i

        # supplyment more waypoints
        if max_index >= 0:
            for i in range(max_index + 1):  # max_index + 1 is the amount number of waypoints to pop out into buffer
                self.near_waypoint_queue.append(self._waypoint_buffer.popleft())

        # find 2 nearest waypoint in near_waypoint_queue
        distance_list = []
        for i, (waypoint, _) in enumerate(self.near_waypoint_queue):
            # distance = waypoint.transform.location.distance(self.ego_vehicle.get_location())
            distance = waypoint.location.distance(self.ego_vehicle.get_location())
            add_dict = {'index': i, 'waypoint': waypoint, 'distance': distance}
            distance_list.append(add_dict)

        # todo: check if distance is effective
        # distance between vehicle and waypoints should be smaller than original distance between waypoints
        # if distance_to_vehicle(self.ego_vehicle, waypoint.location) >= min_distance:
        #     pass

        # using heapq to find nearest distance waypoints
        if len(distance_list) < 2:
            wp1_loc = self.ego_vehicle.get_location()
            wp2_loc = distance_list[0]['waypoint'].location

        else:
            wp1, wp2 = heapq.nsmallest(2, distance_list, key=lambda s: s['distance'])
            threthold = 3.0
            if wp1['distance'] > threthold and wp2['distance'] > threthold:
                self.off_route = True
                print('offroute: ', self.off_route)

            # plot local direction of plan
            if wp1['index'] < wp2['index']:
                cup = wp1
                wp1 = wp2
                wp2 = cup

            wp1_loc = wp1['waypoint'].location
            wp2_loc = wp2['waypoint'].location

        wp1_loc.z = 1.
        wp2_loc.z = 1.
        debug = self.world.debug
        debug.draw_arrow(wp1_loc, wp2_loc, thickness=0.1, arrow_size=0.1,
                         color=carla.Color(0, 255, 0), life_time=1.)

    def __call__(self):
        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        # if using input
        # control = self.run_step(self.state)

        control = self.run_step()
        control.manual_gear_shift = False

        return control

    def run_step(self, state=None):
        """
        Execute one step of vehicle action.
        :return: control command of ego vehicle (carla.VehicleControl)
        """

        """
        previous version 
        
        # ==================================================
        # get kinematic state of ego vehicle

        # buffer waypoints and get 2 nearest waypoints
        last_waypoint_location, next_waypoint_location = self.buffer_waypoint()

        # calculate state for RL
        ego_transform = self.ego_vehicle.get_transform()
        lat_offset, diversion_angle = self.get_local_geometry_state(ego_transform,
                                                                    last_waypoint_location, next_waypoint_location)
        print("lat_offset: ", lat_offset)
        print("diversion_angle", diversion_angle)

        # get state about navigation(target waypoint)
        # vector to target waypoint in local frame
        lon_diatance, lat_diatance = self.get_navigation_state()

        print("lon_diatance: ", lon_diatance)
        print("lat_diatance: ", lat_diatance)

        # try to catch a bug
        # if diversion_angle >= 50:
        #     print("need to check")
        # ==================================================
        
        """

        self.buffer_waypoint2()

        # get lateral control action
        steering = self.lateral_controller.run_step(self._waypoint_buffer[0][0])  # carla.transform

        next_loc = self._waypoint_buffer[0][0].location
        self.world.debug.draw_point(next_loc, size=0.15, color=carla.Color(255, 0, 0), life_time=.5)

        # get longitudinal control from RL
        # state is a list consists of float value
        state = self.get_state()

        if self.state is None:
            self.state = state
        else:
            self.next_state = state
            transition = Transition(self.state, self.action, self.reward, self.next_state)
            self.algorithm.store_transition(transition)
            self.state = self.next_state

        # get action from RL module
        self.action = self.algorithm.select_action(state)

        action_list = ['acc', 'maintain', 'brake']
        print("action: ", action_list[self.action])
        # print('throttle: ', self.action_space[self.action][0])
        # print('brake: ', self.action_space[self.action][1])

        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = self.action_space[self.action][0]
        control.brake = self.action_space[self.action][1]
        control.hand_brake = False

        print('throttle:', control.throttle)
        print('steer:', control.steer)

        return control

