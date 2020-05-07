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

# import carla.ColorConverter

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
# local frame origin
# carla.transform
# Location(x=5.421652, y=120.832970, z=0.000000)
# Rotation(pitch=360.000000, yaw=269.637451, roll=0.000000)
"""

class RLAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file, episode_index):
        super(RLAgent, self).__init__(path_to_conf_file)  # useless, no config file
        # RLagent net
        self.algorithm = None
        # current global plans to reach a destination
        self._global_plan = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.state = None
        self.next_state = None
        self.reward = None

        # original input and action space for RL
        # self.image_shape = (3, 300, 200)
        self.action_space = []
        self.action_shape = None
        # self.steer_space = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        self.steer_space = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.acc_space = [0.0, 0.5, 1.0]

        # agent's initialization
        self.setup(path_to_conf_file)

        # generate action space
        # self.generate_action_space()
        # image index???
        self.index = 0



        # ==================================================
        # Action space only consider longitudinal control
        self.acc_space = self.acc_space = [0.0, 0.5, 1.0]
        # todo: use a dense action space with number n
        self.action_shape = len(self.acc_space)

        self.get_action_space2()

        # ==================================================

        self.episode_index = episode_index

        # ==================================================
        # waypoint buffer for navigation and state representation
        # waypoints queue
        self._waypoints_queue = deque(maxlen=100000)  # maximum waypoints to store in current route
        # waypoint buffer
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # near waypoint
        self.near_waypoint_queue = deque(maxlen=5)
        # min distance
        # MIN_DISTANCE_PERCENTAGE = 0.9
        # default params
        # self._dt = 1.0 / 20.0
        # self._target_speed = 20.0  # Km/h
        # self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        # self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._min_distance = 3  # min distance of reaching waypoints

        # ==================================================
        # set lateral controller
        self.lateral_controller = None  # lateral controller will be set later by env script
        # self.set_lateral_controller()

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.ALL_SENSORS : LIDAR, cameras, GPS and speed sensor allowed
            Track.CAMERAS : Only cameras and GPS allowed
            Track.ALL_SENSORS_HDMAP_WAYPOINTS : All sensors and HD Map and waypoints allowed
            Track.SCENE_LAYOUT : No sensors allowed, the agent receives a high-level representation of the scene.
        """
        self.track = Track.CAMERAS

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
            'dt': 0.03}

        # todo: check if get dict correctly
        # init controller
        if not args_lateral:
            args_lateral = args_lateral_dict

        self.lateral_controller = PIDLateralController(self.ego_vehicle, **args_lateral)

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

    def generate_action_space(self):
        """
        根据steer_num和acc_num划分离散动作空间
        :return:
        """
        for item in product(self.steer_space, self.acc_space):
            self.action_space.append(item)
        self.action_shape = len(self.action_space)

    def get_action_space2(self):
        """
        Get action space of only longitudinal action.
        """
        self.action_space = self.acc_space
        self.action_shape = len(self.action_space)

    def buffer_waypoint(self):
        """
        Buffer waypoint
        Get 2 nearest waypoint
        :return:
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

    def run_step(self, state=None):
        """
        Execute one step of navigation.
        :return: control
        """
        # set to zero
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        # ==================================================

        # get info about ego vehicle

        # buffer waypoints and get 2 nearest waypoints
        last_waypoint_location, next_waypoint_location = self.buffer_waypoint()

        # calculate state for RL
        ego_transform = self.ego_vehicle.get_transform()
        lat_offset, diversion_angle = self.get_local_geometry_state(ego_transform,
                                                                    last_waypoint_location, next_waypoint_location)
        print("lat_offset: ", lat_offset)
        print("diversion_angle", diversion_angle)

        # debug
        if lat_offset > 0.95:
            print("debug")

        if diversion_angle > 0.44:
            print("debug")

        # get state about navigation(target waypoint)
        # todo: this should be master_scenario_state in future version
        # vector to target waypoint in local frame
        lon_diatance, lat_diatance = self.get_navigation_state()

        print("lon_diatance: ", lon_diatance)
        print("lat_diatance: ", lat_diatance)

        # todo: add state, velocity projection on local direction
        # lon_vel_nav, lat_vel_nav = self.get_velo_nav()

        # print to debug
        print("lat_offset: ", lat_offset)
        print("diversion_angle: ", diversion_angle)
        print("lon_diatance: ", lon_diatance)
        print("lat_diatance: ", lat_diatance)

        # stack all state into state_dict
        # state_list = [lon_diatance, lat_diatance, lat_offset, diversion_angle]
        # state_dict = {"lon_diatance": lon_diatance,
        #               "lat_diatance": lat_diatance,
        #               "lat_offset": lat_offset,
        #               "diversion_angle": diversion_angle
        #               }
        # state_dict = {"state_1": lon_diatance,
        #               "state_2": lat_diatance,
        #               "state_3": lat_offset,
        #               "state_4": diversion_angle
        #               }
        # print("All state extracted.")

        # utilize stored state
        # transform state into tensor
        # state_tensor = []
        # for state in state_list:  # state stored in list
        #     tensor = torch.tensor([state])
        #     tensor = tensor.unsqueeze(0)
        #     state_tensor.append(tensor)

        # ==================================================
        # try to catch a bug
        # if diversion_angle >= 50:
        #     print("need to check")
        # ==================================================

        if self.state is None:
            self.state = {'state_1': lon_diatance, 'state_2': lat_diatance, 'state_3': lat_offset,
                          'state_4': diversion_angle}
        else:
            self.next_state = {'state_1': lon_diatance, 'state_2': lat_diatance, 'state_3': lat_offset,
                               'state_4': diversion_angle}
            transition = Transition(self.state, self.action, self.reward, self.next_state)
            self.algorithm.store_transition(transition)
            self.state = self.next_state

        state_1 = self.state['state_1']
        state_1_tensor = torch.tensor([state_1])
        state_1_tensor = state_1_tensor.unsqueeze(0)

        state_2 = self.state['state_2']
        state_2_tensor = torch.tensor([state_2])
        state_2_tensor = state_2_tensor.unsqueeze(0)

        state_3 = self.state['state_3']
        state_3_tensor = torch.tensor([state_3])
        state_3_tensor = state_3_tensor.unsqueeze(0)

        state_4 = self.state['state_4']
        state_4_tensor = torch.tensor([state_4])
        state_4_tensor = state_4_tensor.unsqueeze(0)

        state_list = [state_1, state_2]  # float

        state_list_tensor = []
        for state in state_list:
            state = torch.tensor([state])
            state = state.unsqueeze(0)
            state_list_tensor.append(state)

        # get action from RL module
        action_index = self.algorithm.select_action(state_list_tensor[0], state_list_tensor[1], state_3_tensor,
                                                    state_4_tensor)  # get sparse action index

        # print("action_index: ", action_index)  # print selected action index

        # get next transform in buffer as target waypoint
        # format is carla.transform
        target_waypoint_trans = self._waypoint_buffer[0][0]

        # get lateral control by run step controller
        steering = self.lateral_controller.run_step(target_waypoint_trans)

        # # todo: figure out if index>=15
        # # action space bug
        # if action_index == 15:
        #     print("action space error")

        # get corresponding actual action
        self.action = action_index  # keep this to update RL

        # control.steer = self.action_space[action_index][0]  # to be fixed
        # acc = self.action_space[action_index][1]

        acc = self.action_space[action_index]

        control.throttle = self.action_space[action_index]
        control.steer = steering

        # if acc >= 0.0:
        #     control.throttle = acc
        #     control.brake = 0.0
        # else:
        #     control.throttle = 0.0
        #     control.brake = abs(acc)
        # control.throttle = acc
        print('throttle:', control.throttle)
        print('steer:', control.steer)

        return control

    def get_state(self):
        # args of this method has changed

        # ==================================================
        """
        # camera sensor data
        sensor_data = self.sensor_interface.get_data()
        # process image
        image_input = sensor_data['Mid']
        bgr_image = image_input[1][:, :, :3]

        # 注释掉此段，此段用来收集和处理语义分割图像，与现有挑战赛框架不符
        # semcamera = sensor_data['Sem']
        # sem_image = semcamera[1][:,:,:3]
        # self.save_image(50,'sem',sem_image)

        bgr_image = bgr_image[:, :, ::-1]
        self.save_image(50, 'rgb', bgr_image)

        bgr_image = bgr_image.astype(np.float32)
        bgr_image = np.multiply(bgr_image, 1.0 / 255.0)
        bgr_image = np.transpose(bgr_image, (2, 1, 0))

        self.index += 1

        if self.state is None:
            self.state = {'image': bgr_image, 'speedx': speedx, 'speedy': speedy, 'steer': steer}
        else:
            self.next_state = {'image': bgr_image, 'speedx': speedx, 'speedy': speedy, 'steer': steer}
            transition = Transition(self.state, self.action, self.reward, self.next_state)
            self.algorithm.store_transition(transition)
            self.state = self.next_state
        """
        pass

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        # control = self.run_step(self.state)

        control = self.run_step()

        control.manual_gear_shift = False
        return control

    def all_sensors_ready(self):
        return self.sensor_interface.all_sensors_ready()

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

    def get_image_shape(self):
        return self.image_shape

    def get_state_shape(self):
        """

        :return:
        """
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def set_ego_vehicle(self, ego_vehicle):
        """
        Set ego vehicle from env.
        """
        self.ego_vehicle = ego_vehicle

    def set_world(self, world):
        self.world = world

    def get_reward(self, reward):
        self.reward = reward
        print('reward:', self.reward)

    def save_image(self, interval, path, image):
        if self.index % interval == 0:
            im = Image.fromarray(image)
            # im.save('/home/guoyoutian/scenario_runner-0.9.5/DQN/image/'+ path + '/' +str(self.episode_index) + '_%03d.jpeg' %(self.index/interval))
            print("image")

    def get_navigation_state(self):
        """
        The State based on ground truth geometry and kinematics.

        Consider coord frame fixed with local lane.

        Stage 1: follow waypoints plan
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
