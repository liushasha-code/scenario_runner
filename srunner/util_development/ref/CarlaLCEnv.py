# import glob
# import os
# import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
import carla
import numpy as np
import matplotlib.pyplot as plt
# from carla import *
# from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory
from srunner.scenariomanager.timer import GameTime

import random
import time
import weakref
# import re
# import collections
import math
import copy
import traceback
import prettytable as pt
from PIL import Image

import controller2d_m
import configparser
import local_planner
import behavioural_planner
from vehicle import *

from PIL import Image, ImageDraw
COLOR_R = (255, 0, 0)
COLOR_G = (0, 255, 0)
COLOR_B = (0, 0, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_WIHTE = (128, 128, 128)
PIXELS_PER_METER = 12

NUM_PATHS = 1
BP_LOOKAHEAD_BASE = 8.0  # m
BP_LOOKAHEAD_TIME = 1.2  # s
PATH_OFFSET = 0.7  # m
CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
TIME_GAP = 1.0  # s
PATH_SELECT_WEIGHT = 10
A_MAX = 2.0  # m/s^2
SLOW_SPEED = 2.0  # m/s
STOP_LINE_BUFFER = 3.5  # m
LEAD_VEHICLE_LOOKAHEAD = 20.0  # m
LP_FREQUENCY_DIVISOR = 3  # Frequency divisor to make the
# local planner operate at a lower
# frequency than the controller
# (which operates at the simulation
# frequency). Must be a natural
# number.
INTERP_DISTANCE_RES = 0.01  # distance between interpolated points
DESIRE_SPEED = 20.0


class MapImage(object):
    """Class encharged of rendering a 2D image from top view of a carla world. 
    Please note that a cache system is used, so if the OpenDrive content
    of a Carla town has not changed, it will read and use the stored image 
    if it was rendered in a previous execution"""

    def __init__(self, carla_world, carla_map, pixels_per_meter):
        """ Renders the map image generated based on the world, 
        its map and additional flags that provide extra information about the road network"""
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        # Maximum size of a Pygame surface
        width_in_pixels = (1 << 14) - 1

        # Adapt Pixels per meter to make world fit in surface
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > PIXELS_PER_METER:
            surface_pixel_per_meter = PIXELS_PER_METER

        self._pixels_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.image = Image.open('Town05_{}.png'.format(PIXELS_PER_METER))
        # print(self.image.mode)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return (int(x - offset[0]), int(y - offset[1]))


def distance_vehicle(waypoint, vehicle_position):
    dx = waypoint['lat'] - vehicle_position[0]
    dy = waypoint['lon'] - vehicle_position[1]

    return math.sqrt(dx * dx + dy * dy)


def get_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def get_current_pose(transform):
    x = transform.location.x
    y = transform.location.y
    yaw = transform.rotation.yaw
    r_yaw = yaw * math.pi / 180.0
    if r_yaw > math.pi:
        r_yaw -= 2 * math.pi
    elif r_yaw < -math.pi:
        r_yaw += 2 * math.pi
    return x, y, r_yaw


def get_speed(current_pos, prev_pos, delta_time):
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    # dz = current_pos[2] - prev_pos[2]
    # dis = math.sqrt(dx * dx + dy * dy + dz * dz)
    dis = math.sqrt(dx * dx + dy * dy)
    if delta_time == 0:
        return 0
    else:
        return dis / delta_time


def calculate_speed(v):
    speed = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if speed < 1e-2:
        speed = 0
    return speed


def generate_route(world, vehicle_location, hop_resolution=1.0):
    # generate a route for current scenario
    # based on current scenario and map
    current_map = world.get_map()

    # get initial location of ego_vehicle
    start_waypoint = current_map.get_waypoint(vehicle_location)

    # generate a dense route according to current scenario
    # could ref to <scenario_helper> module
    turn_flag = 0  # turn_flag by current scenario
    mid_waypoint = generate_target_waypoint(start_waypoint, turn_flag)
    end_waypoint_list = mid_waypoint.next(20)
    end_waypoint = end_waypoint_list[-1]
    turn_flag = -1  # turn_flag by current scenario
    mid_waypoint = generate_target_waypoint(end_waypoint, turn_flag)
    end_waypoint_list = mid_waypoint.next(20)
    end_waypoint = end_waypoint_list[-1]

    # generate a dense route
    # Setting up global router
    waypoints = [start_waypoint.transform.location, end_waypoint.transform.location]
    gps_route, trajectory = interpolate_trajectory(world, waypoints, hop_resolution)
    return gps_route, trajectory


def draw_trajectory(world, global_plan_world_coord, persistency=-1, vertical_shift=1):
    for index in range(len(global_plan_world_coord)):
        waypoint = global_plan_world_coord[index][0]
        location = waypoint.location + carla.Location(z=vertical_shift)
        world.debug.draw_point(location, size=0.1, color=carla.Color(34, 125, 81), life_time=persistency)


def draw_waypoints(world, waypoints, ego_location, persistency=-1.0, vertical_shift=1.5, color=(255, 0, 0)):
    if waypoints is None or len(waypoints) == 0:
        return
    for position in waypoints:
        location = carla.Location(x=position[0], y=position[1], z=ego_location.z+vertical_shift)
        # height = world.get_map().get_waypoint(location).transform.location.z
        # location = carla.Location(x=position[0], y=position[1], z=height+vertical_shift)
        draw_color = carla.Color(color[0], color[1], color[2])
        if location.distance(ego_location) > 2:
            world.debug.draw_point(location, size=0.1, color=draw_color, life_time=persistency)


def set_sespector(world, ego_trans, wp_angle=None):
    # spectator = world.get_spectator()
    # angle = ego_trans.rotation.yaw
    # d = 5.4
    # a = math.radians(180 + angle)
    # location = carla.Location(d * math.cos(a), d * math.sin(a), 15.0) + ego_trans.location
    # spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-50)))

    # d = 6.4
    # a = math.radians(180 + angle)
    # location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + ego_trans.location
    # spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15)))

    angle = wp_angle if wp_angle is not None else ego_trans.rotation.yaw
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=120),
                                            carla.Rotation(pitch=-90, yaw=180)))


def get_local_coordinate_frame(world, origin_transform, axis_length_scale=3.0, persistency=1.0):
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, origin_transform.location.z + 2])
    # elevate z coordinate
    Origin_location = carla.Location(origin_coord[0], origin_coord[1], origin_coord[2])
    # x axis destination
    x_des_coord = origin_coord + axis_length_scale * np.array([cy, sy, 0])
    x_des = carla.Location(x_des_coord[0], x_des_coord[1], x_des_coord[2])
    # y axis destination
    y_des_coord = origin_coord + axis_length_scale * np.array([-sy, cy, 0])
    y_des = carla.Location(y_des_coord[0], y_des_coord[1], y_des_coord[2])
    # z axis destination
    z_des_coord = origin_coord + axis_length_scale * np.array([0, 0, 1])
    z_des = carla.Location(z_des_coord[0], z_des_coord[1], z_des_coord[2])

    # axis feature
    # thickness = 0.1f
    # arrow_size = 0.1f
    if persistency > 0.0:
        x_axis_color = carla.Color(255, 0, 0)
        y_axis_color = carla.Color(0, 255, 0)
        z_axis_color = carla.Color(0, 0, 255)
        world.debug.draw_arrow(Origin_location, x_des, color=x_axis_color, life_time=persistency)
        world.debug.draw_arrow(Origin_location, y_des, color=y_axis_color, life_time=persistency)
        world.debug.draw_arrow(Origin_location, z_des, color=z_axis_color, life_time=persistency)

    return x_des, y_des, z_des


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        # self.collision = False
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        # self.collision = True
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class CarlaEnv(object):
    def __init__(self, carla_world, traffic_manager=None):
        self.world = carla_world
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.timestamp = None
        self.count_time = 0
        # settings = self.world.get_settings()
        # settings.fixed_delta_seconds = 0.1
        # self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicles = self.blueprint_library.filter('vehicle.*')
        self.cars = [x for x in self.vehicles if int(x.get_attribute('number_of_wheels')) == 4]
        self.spawn_points = self.map.get_spawn_points()

        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.map,
            pixels_per_meter=PIXELS_PER_METER)

        self.actor_list = []
        self.hero = None
        self.sensor = None
        # self.restart()
        self.recording_enabled = False
        self.recording_start = 0
        self._waypoints = None
        self.scenario = None
        self.sub_scenario = None
        self.ctrl_vehicle = None

        self.lp = None
        self.bp = None
        self.controller = None
        self.current_timestamp = 0.0
        self.prev_timestamp = 0.0
        self.count_frame = 0
        # self.vehicles = None
        self._waypoints = None
        self.lane = 1
        self.local_waypoints = None
        self.lead_car_speed = 0.0
        self.desired_speed = 0.0
        self.dis_to_goal = BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * DESIRE_SPEED
        self.run_traffic_light = False
        self.tl_waypoint = None
        self.collision = False
        self.collision_times = 0
        self.collision_his = 0
        self.speed_sum = 0.0
        self.speed_step = 0
        self.lane_change_times = 0
        self.lc_reward = 0.0

        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 0.0
        self.current_control.brake = 1.0
        self.current_control.hand_brake = False
        self.world.on_tick(self._update_timestamp)

        self.traffic_manager = None
        if traffic_manager is not None:
            self.traffic_manager = traffic_manager
            self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)
            # print("traffic_manager set")

    def _update_timestamp(self, snapshot):
        self.timestamp = snapshot.timestamp
        if self.scenario == 8:
            self.count_time += 1

    def restart(self, scenario=None):
        # Spawn the ego car.
        # while self.hero is None:
            # self.setup_hero(ego_model, ego_random_location)
        # if scenario is None:
        #     self.scenario = np.random.randint(6)
        # else:
        self.scenario = scenario
        self.reset()

        def spawn_npc(x, y, model='vehicle.audi.tt'):
            num_npc = 0
            blueprint = random.choice(self.vehicles.filter(model))
            blueprint.set_attribute('role_name', 'autopilot')

            npc = None
            loc = self.multi_waypoints[x][y]
            wp = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]))
            transform = carla.Transform(wp.transform.location + 
                                        carla.Location(z=0.5), wp.transform.rotation)
            npc = self.world.try_spawn_actor(blueprint, transform)
            if npc is not None:
                self.actor_list.append(npc)
                npc_control = carla.VehicleControl(throttle=0.0, steer=0, brake=1)
                npc.apply_control(npc_control)
                num_npc += 1
            return num_npc

        # prepare the scenario
        if self.scenario == 0:  # empty road
            num_npcs = 0
            pass
        elif self.scenario == 1:  # no car in front of the ego car
            num_npcs = 0
            while num_npcs < 4:
                x = np.random.randint(3)
                y = np.random.randint(150)
                if x == self.lane and y > 60:
                    continue
                num_npcs += spawn_npc(x, y)
        elif self.scenario == 2:  # a car in front of the ego car
            num_npcs = 0
            x = self.lane
            y = 65 + np.random.randint(11)
            num_npcs += spawn_npc(x, y)
        elif self.scenario == 3:  # a car in front of the ego car and other random cars
            num_npcs = 0
            x = self.lane
            y = 65 + np.random.randint(11)
            num_npcs += spawn_npc(x, y)
            x_f = self.lane + 2*np.random.randint(1) - 1 if self.lane == 1 else 1
            y_f = np.random.randint(y, 91)
            num_npcs += spawn_npc(x_f, y_f)
            for _ in range(np.random.randint(3)):
                x = np.random.randint(3)
                y = 90 + np.random.randint(31)
                num_npcs += spawn_npc(x, y)
        elif self.scenario == 4:  # cars in front of the ego car in ego lane and neighbor lane
            num_npcs = 0
            x = self.lane
            y = 65 + np.random.randint(11)
            num_npcs += spawn_npc(x, y)
            x = self.lane + 2*np.random.randint(1) - 1
            y = 65 + np.random.randint(21)
            num_npcs += spawn_npc(x, y)
            for _ in range(np.random.randint(3)):
                x = np.random.randint(3)
                y = np.random.randint(50)
                num_npcs += spawn_npc(x, y)
        elif self.scenario == 5:  # cars in front of the ego car in every lane
            num_npcs = 0
            for i in range(3):
                x = i
                y = 65 + np.random.randint(21)
                num_npcs += spawn_npc(x, y)
            for _ in range(np.random.randint(3)):
                x = np.random.randint(3)
                y = np.random.randint(50)
                num_npcs += spawn_npc(x, y)
        elif self.scenario == 6:  # cars in front of the ego car and near front in neighbor lane
            num_npcs = 0
            x = self.lane
            y = 65 + np.random.randint(16)
            num_npcs += spawn_npc(x, y)
            x_f = 1
            y_f = np.random.randint(60, y)
            num_npcs += spawn_npc(x_f, y_f)
            for _ in range(np.random.randint(3)):
                x = np.random.randint(3)
                y = 90 + np.random.randint(31)
                num_npcs += spawn_npc(x, y)
        elif self.scenario == 7:  # cars in front of the ego car and near rear in neighbor lane
            num_npcs = 0
            x = self.lane
            y = 65 + np.random.randint(16)
            num_npcs += spawn_npc(x, y)
            x_f = 1
            y_f = np.random.randint(30, 60)
            num_npcs += spawn_npc(x_f, y_f)
            # for _ in range(np.random.randint(3)):
            #     x = np.random.randint(3)
            #     y = 90 + np.random.randint(31)
            #     num_npcs += spawn_npc(x, y)
        else:  # random spawn cars
            num_npcs = 0
            expect_num_npcs = np.random.randint(6, 13)  # np.random.randint(4, 9)6, 13
            while num_npcs < expect_num_npcs:
                x = np.random.randint(3)
                y = np.random.randint(30, 240)
                num_npcs += spawn_npc(x, y)
        print("number of other cars: {}".format(num_npcs))

        if self.scenario == 7:
            self.sub_scenario = np.random.randint(4)  # np.random.randint(4)
            if self.sub_scenario != 0:
                x = 2 - self.lane
                y = np.random.randint(60, 66)
                spawn_npc(x, y, 'vehicle.nissan.micra')
                for _ in range(20):
                    self.world.tick()
                for actor in self.actor_list[:-1]:
                    actor.set_autopilot(True)
                    dis_to_lead = 5.0 + np.random.randint(10)
                    self.traffic_manager.distance_to_leading_vehicle(actor, dis_to_lead)
                ctrl_vehicle = self.actor_list[-1]
                self.ctrl_vehicle = Vehicle(self.world, self.multi_waypoints, ctrl_vehicle, x)
                self.ctrl_vehicle.DESIRE_SPEED = np.random.randint(5, 20)
                for _ in range(10):
                    self.world.tick()
                    GameTime.on_carla_tick(self.timestamp)
                    timestamp = GameTime.get_time()
                    self.ctrl_vehicle.step_one_loop(timestamp, 0, True)
                if self.sub_scenario != 1:
                    self.ctrl_vehicle.lane = 1
        if self.scenario != 7 or self.sub_scenario == 0:
            for _ in range(20):
                self.world.tick()
            for actor in self.actor_list:
                actor.set_autopilot(True)
                dis_to_lead = 5.0 + np.random.randint(10)
                self.traffic_manager.distance_to_leading_vehicle(actor, dis_to_lead)
                # self.traffic_manager.vehicle_percentage_speed_difference(actor, -100)

        ob, _, _ = self.get_state(self.hero)
        return ob

    def reset(self, ticks=30):
        self.destroy()
        self.lp = local_planner.LocalPlanner(NUM_PATHS,
                                             PATH_OFFSET,
                                             CIRCLE_OFFSETS,
                                             CIRCLE_RADII,
                                             PATH_SELECT_WEIGHT,
                                             TIME_GAP,
                                             A_MAX,
                                             SLOW_SPEED,
                                             STOP_LINE_BUFFER)
        self.bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                         LEAD_VEHICLE_LOOKAHEAD)
        self.controller = None
        self.current_timestamp = 0.0
        self.prev_timestamp = 0.0
        self.count_frame = 0
        # self.vehicles = None
        self._waypoints = None
        self.multi_waypoints = []
        self.local_waypoints = None
        self.lead_car_speed = 0.0
        self.desired_speed = 0.0
        self.run_traffic_light = False
        self.tl_waypoint = None

        # ob = None
        success, trans = self.setup_hero()
        if success:
            self.current_control.steer = 0.0
            self.current_control.throttle = 0.0
            self.current_control.brake = 1.0
            self.current_control.hand_brake = False
            self.hero.apply_control(self.current_control)
            for _ in range(ticks):
                self.world.tick()
            GameTime.on_carla_tick(self.timestamp)
            timestamp = GameTime.get_time()
            self.controller.vars.t_prev = timestamp
            set_sespector(self.world, self.hero.get_transform())
            # ob, _, _ = self.get_state(self.hero)
        # return np.array(ob)

    def setup_hero(self, model='vehicle.lincoln.mkz2017', random_location=False):
        blueprint = random.choice(self.vehicles.filter(model))
        blueprint.set_attribute('role_name', 'hero')
        # self.spawn_points = self.map.get_spawn_points()
        if random_location:
            transform = random.choice(self.spawn_points)
        else:
            # transform = carla.Transform(carla.Location(x=130, y=55.2, z=0.5), carla.Rotation(yaw=180))
            # transform = carla.Transform(carla.Location(x=-410.0, y=12.73, z=0.5), carla.Rotation(yaw=180))
            transform = carla.Transform(carla.Location(x=6.0, y=-204.0, z=0.5), carla.Rotation(yaw=180))
        if self.hero is None:
            # self.hero = self.world.try_spawn_actor(blueprint, transform)
            # gps_route, trajectory = generate_route(self.world, transform.location)
            self.multi_waypoints = []
            start_waypoint = self.map.get_waypoint(transform.location)
            end_waypoint = start_waypoint.next(160)[-1]
            left_start_waypoint = start_waypoint.get_left_lane()
            left_end_waypoint = end_waypoint.get_left_lane()
            right_start_waypoint = start_waypoint.get_right_lane()
            right_end_waypoint = end_waypoint.get_right_lane()
            waypoints = [start_waypoint.transform.location, end_waypoint.transform.location]
            left_waypoints = [left_start_waypoint.transform.location, left_end_waypoint.transform.location]
            right_waypoints = [right_start_waypoint.transform.location, right_end_waypoint.transform.location]
            gps_route, trajectory = interpolate_trajectory(self.world, waypoints, 1.0)
            left_gps_route, left_trajectory = interpolate_trajectory(self.world, left_waypoints, 1.0)
            right_gps_route, right_trajectory = interpolate_trajectory(self.world, right_waypoints, 1.0)
            self.set_global_plan(gps_route, trajectory)
            # draw_trajectory(self.world, self._global_plan_world_coord)
            # draw_trajectory(self.world, left_trajectory)
            # draw_trajectory(self.world, right_trajectory)
            left_waypoints_list = self._get_waypoints(left_trajectory)
            waypoints_list = self._get_waypoints(trajectory)
            right_waypoints_list = self._get_waypoints(right_trajectory)
            self.multi_waypoints.append(left_waypoints_list)
            self.multi_waypoints.append(waypoints_list)
            self.multi_waypoints.append(right_waypoints_list)
            self.multi_waypoints = np.array(self.multi_waypoints)
            # print(self.multi_waypoints.shape())
            
            # self.lane = 1
            # spawn_ego_trans = carla.Transform(trajectory[60][0].location + 
            #     carla.Location(z=0.5), trajectory[60][0].rotation)
            if self.scenario == 4:
                self.lane = 1
            elif self.scenario in [6, 7]:
                self.lane = 2 * np.random.randint(2)
            else:
                self.lane = np.random.randint(3)
            if self.lane == 0:
                spawn_ego_trans = carla.Transform(left_trajectory[60][0].location + 
                    carla.Location(z=0.5), left_trajectory[60][0].rotation)
            elif self.lane == 2:
                spawn_ego_trans = carla.Transform(right_trajectory[60][0].location + 
                    carla.Location(z=0.5), right_trajectory[60][0].rotation)
            else:
                spawn_ego_trans = carla.Transform(trajectory[60][0].location + 
                    carla.Location(z=0.5), trajectory[60][0].rotation)
            self.hero = self.world.try_spawn_actor(blueprint, spawn_ego_trans)
            if self._waypoints is None and self._global_plan is not None:
                self._waypoints = self.multi_waypoints[self.lane]
                self.controller = controller2d_m.Controller2D(self._waypoints)
            if self.hero is not None:
                self.sensor = CollisionSensor(self.hero)
                return True, transform
            else:
                print('creat ego car failed')
                return False, transform
        else:
            self.hero.set_transform(transform)
            # gps_route, trajectory = generate_route(self.world, transform.location)
            start_waypoint = self.map.get_waypoint(transform.location)
            end_waypoint = self.map.get_waypoint(carla.Location(x=9.0, y=124.67))
            waypoints = [start_waypoint.transform.location, end_waypoint.transform.location]
            gps_route, trajectory = interpolate_trajectory(self.world, waypoints, 1.0)
            self.set_global_plan(gps_route, trajectory)
            if self._waypoints is None and self._global_plan is not None:
                self._waypoints = self._get_waypoints(self._global_plan_world_coord)
                self.controller = controller2d_m.Controller2D(self._waypoints)
            return True, transform

    def step(self, action):
        # print("state: {}, action: {}".format(self.bp._state, action))
        self.step_one_loop(action, is_new_action=True)
        if self.bp._state == 1 or self.bp._state == 0:
            for _ in range(10):
                self.step_one_loop(action, is_new_action=False)
        else:
            for _ in range(5):
                self.step_one_loop(action, is_new_action=False)
        ob, reward, done = self.get_state(self.hero, show_message=False)
        return ob, reward, done

    def step_one_loop(self, action, is_new_action=False):
        self.count_frame += 1
        GameTime.on_carla_tick(self.timestamp)
        timestamp = GameTime.get_time()
        delta_time = timestamp - self.prev_timestamp
        # print("timestamp: {}, delta_time: {}".format(timestamp, delta_time))
        if delta_time == 0:
            return
        self.prev_timestamp = self.current_timestamp
        self.current_timestamp = timestamp

        if self.scenario == 7:
            if self.sub_scenario == 3:
                ctrl_vehicle_wp = self.get_vehicle_wp(self.ctrl_vehicle.hero)
                if ctrl_vehicle_wp.lane_id == -2:
                    self.ctrl_vehicle.lane = self.ctrl_vehicle.init_lane
            if self.sub_scenario != 0:
                self.ctrl_vehicle.step_one_loop(timestamp, 0, True)

        trans = self.hero.get_transform()
        current_x, current_y, current_yaw = get_current_pose(trans)
        v_xy = self.hero.get_velocity()
        current_speed = math.sqrt(v_xy.x ** 2 + v_xy.y ** 2 + v_xy.z ** 2)
        ego_wp = self.get_vehicle_wp(self.hero)
        ego_s = ego_wp.s
        ego_lane = ego_wp.lane_id
        # get_local_coordinate_frame(self.world, trans, persistency=0.05)

        lead_car_pos = []
        lead_car_speed = []
        parkedcar_box_pts = []
        # parkedcar_box_pts.append([254.0, 134.5])  # stationary object location
        self.bp._stopsign_fences = []
        # best_path = []

        if self.count_frame % LP_FREQUENCY_DIVISOR == 0 or is_new_action:
            self.count_frame = 1
            # ------------------------- change lane ------------------------- 
            last_lane = self.lane
            if action == 0 and is_new_action:
                self._waypoints = self.multi_waypoints[self.lane]
            elif action == 3 and (ego_lane == -3 or ego_lane ==-2) and is_new_action:
                if self.lane >= 1:
                    self.lane -= 1
                    self.lc_reward = -0.1
                else:
                    self.lc_reward = -1.0
                # self._waypoints = self.multi_waypoints[self.lane]
            elif action == 4 and (ego_lane == -1 or ego_lane ==-2) and is_new_action:
                # self.lane = self.lane + 1 if self.lane <= 1 else self.lane
                if self.lane <= 1:
                    self.lane += 1
                    self.lc_reward = -0.1
                else:
                    self.lc_reward = -1.0
                # self._waypoints = self.multi_waypoints[self.lane]
            intent_lane = -self.lane - 1
            if abs(intent_lane - ego_lane) == 2:
                self.lane = 1
            self._waypoints = self.multi_waypoints[self.lane]
            if last_lane != self.lane:
                self.lane_change_times += 1
            # --------------------------------------------------------------- 

            # ------------------------- get lead car info ------------------------- 
            d_cf = self.bp._lookahead
            lead_car = None
            for i, actor in enumerate(self.world.get_actors().filter('vehicle.*')):
                if actor.id == self.hero.id:
                    continue
                actor_wp = self.get_vehicle_wp(actor)
                actor_lane = actor_wp.lane_id
                actor_s = actor_wp.s
                if actor_lane == (-self.lane-1):
                    if actor_s > ego_s and actor_s - ego_s < d_cf:
                        d_cf = actor_s - ego_s
                        lead_car = actor
            if lead_car is not None:
                lead_loc = lead_car.get_location()
                lead_car_pos.append([lead_loc.x, lead_loc.y])
                lead_v = lead_car.get_velocity()
                lead_car_speed.append(math.sqrt(lead_v.x ** 2 + lead_v.y ** 2 + lead_v.z ** 2))
            # --------------------------------------------------------------------- 

            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(delta_time)
            # print('open_loop_speed: {}'.format(open_loop_speed))
            ego_state = [current_x, current_y, current_yaw, open_loop_speed]
            # print(ego_state)
            self.bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)
            self.bp.transition_state(self._waypoints, ego_state, current_speed, action, is_new_action)

            if len(lead_car_pos) != 0:
                self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0])
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index,
                                                        self.bp._goal_state, self._waypoints, ego_state)

            # ------------------------- draw goal states ------------------------- 
            # for goal_state in goal_state_set:
            #     goal_x = ego_state[0] + goal_state[0]*math.cos(ego_state[2]) - \
            #                                     goal_state[1]*math.sin(ego_state[2])
            #     goal_y = ego_state[1] + goal_state[0]*math.sin(ego_state[2]) + \
            #                                     goal_state[1]*math.cos(ego_state[2])
            #     location = carla.Location(goal_x, goal_y, 1.0)
            #     color = carla.Color(0, 255, 255)
            #     self.world.debug.draw_point(location, size=0.1, color=color, life_time=0.1)
            # --------------------------------------------------------------------- 

            paths, path_validity = self.lp.plan_paths(goal_state_set)
            paths = local_planner.transform_paths(paths, ego_state)
            collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts)
            goal_state = self.bp._goal_state
            best_index = self.lp._collision_checker.select_best_path_index(paths, collision_check_array,
                                                                           goal_state)
            # print('best_index: {}'.format(best_index))
            if best_index == None:
                best_path = self.lp._prev_best_path
            else:
                best_path = paths[best_index]
                self.lp._prev_best_path = best_path
            # print(len(paths))
            # ------------------------- draw paths ------------------------- 
            # for path in paths:
            #     t_path = list(map(list, zip(*path)))
            #     draw_waypoints(self.world, t_path, self.hero.get_location(), 0.1, 1.0, (0, 0, 255))
            # -------------------------------------------------------------- 

            if self.bp._state == behavioural_planner.FOLLOW_LANE:
                self.desired_speed = DESIRE_SPEED
            elif self.bp._state == behavioural_planner.DECELERATE:
                self.desired_speed -= 0.2
                self.desired_speed = max(self.desired_speed, 0.0)
            else:
                self.desired_speed = 0.0
            if len(lead_car_pos) >= 1:
                lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_speed[0]]
            else:
                lead_car_state = []
            decelerate = self.bp._state == behavioural_planner.DECELERATE
            self.local_waypoints = self.lp._velocity_planner.compute_velocity_profile(
                best_path, self.desired_speed, ego_state, current_speed, decelerate,
                lead_car_state, self.bp._follow_lead_vehicle)

            if self.local_waypoints is not None:
                wp_distance = []  # distance array
                self.local_waypoints_np = np.array(self.local_waypoints)
                for i in range(1, self.local_waypoints_np.shape[0]):
                    wp_distance.append(
                        np.sqrt((self.local_waypoints_np[i, 0] - self.local_waypoints_np[i - 1, 0]) ** 2 +
                                (self.local_waypoints_np[i, 1] - self.local_waypoints_np[i - 1, 1]) ** 2))
                wp_distance.append(0)
                wp_interp = []  # interpolated values
                # (rows = waypoints, columns = [x, y, v])
                for i in range(self.local_waypoints_np.shape[0] - 1):
                    wp_interp.append(list(self.local_waypoints_np[i]))
                    num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                     float(INTERP_DISTANCE_RES)) - 1)
                    wp_vector = self.local_waypoints_np[i + 1] - self.local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                        wp_interp.append(list(self.local_waypoints_np[i] + next_wp_vector))
                wp_interp.append(list(self.local_waypoints_np[-1]))
                self.controller.update_waypoints(wp_interp)
        # dis = self.get_dis(self.hero, action)
        # print(dis)

        if self.local_waypoints is not None and self.local_waypoints != []:
            self.controller.update_values(current_x, current_y, current_yaw,
                                          current_speed, self.current_timestamp, self.count_frame)
            self.controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = self.controller.get_commands()
        else:
            cmd_throttle = 0.0
            cmd_steer = 0.0
            cmd_brake = 0.5

        safe_distance = BP_LOOKAHEAD_BASE + (BP_LOOKAHEAD_TIME - 0.5) * current_speed
        if len(lead_car_pos) > 0:
            if get_distance(current_x, current_y, lead_car_pos[0][0], lead_car_pos[0][1]) < safe_distance:
                cmd_throttle = 0.0
                cmd_brake = 0.8

        self.current_control.steer = cmd_steer
        self.current_control.throttle = cmd_throttle
        self.current_control.brake = cmd_brake
        self.current_control.hand_brake = False
        # print(self.current_control)
        # self.current_control = carla.VehicleControl(throttle=0.0, steer=0, brake=1)
        self.hero.apply_control(self.current_control)
        # print("state: {}, speed: {}, desire speed: {}".format(self.bp._state, current_speed, self.desired_speed))
        draw_waypoints(self.world, self.local_waypoints, self.hero.get_location(), 0.05)

        # ------------------------- reset surrounding cars ------------------------- 
        if self.count_time >= 80:
            self.count_time = 0
            num_npc = 0
            count = 0
            while num_npc < np.random.randint(4) and count < 20:
                count += 1
                x = np.random.randint(3)
                y = np.random.randint(30)
                blueprint = random.choice(self.vehicles.filter('vehicle.audi.tt'))
                blueprint.set_attribute('role_name', 'autopilot')

                npc = None
                loc = self.multi_waypoints[x][y]
                wp = self.map.get_waypoint(carla.Location(x=loc[0], y=loc[1]))
                transform = carla.Transform(wp.transform.location + 
                                            carla.Location(z=0.10), wp.transform.rotation)
                npc = self.world.try_spawn_actor(blueprint, transform)
                if npc is not None:
                    velocity = carla.Vector3D(x=-10.0, y=-0.5, z=0.0)
                    npc.set_velocity(velocity)
                    self.actor_list.append(npc)
                    # npc_control = carla.VehicleControl(throttle=0.0, steer=0, brake=1)
                    # npc.apply_control(npc_control)
                    npc.set_autopilot(True)
                    self.world.tick()

                    dis_to_lead = 20.0 + np.random.randint(10)
                    self.traffic_manager.distance_to_leading_vehicle(npc, dis_to_lead)
                    num_npc += 1
        destroy_ids = []
        for i, actor in enumerate(self.world.get_actors().filter('vehicle.audi*')):
            actor_wp = self.get_vehicle_wp(actor)
            actor_s = actor_wp.s
            if actor_s > 800:
                destroy_ids.append(actor.id)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in destroy_ids])
        # -------------------------------------------------------------------------- 

        if ego_lane == -1:
            sespector_wp = ego_wp.get_right_lane()
        elif ego_lane == -3:
            sespector_wp = ego_wp.get_left_lane()
        else:
            sespector_wp = ego_wp
        set_sespector(self.world, sespector_wp.transform, sespector_wp.transform.rotation.yaw)
        self.world.tick()
        # set_sespector(self.world, self.hero.get_transform(), ego_wp.transform.rotation.yaw)

    def get_state(self, vehicle, show_message=False):
        v_xy = vehicle.get_velocity()
        current_speed = math.sqrt(v_xy.x ** 2 + v_xy.y ** 2 + v_xy.z ** 2)
        speed = copy.copy(current_speed)
        # print("speed: {}".format(current_speed))
        trans = vehicle.get_transform()
        loc = trans.location
        rot = trans.rotation
        vehicle_waypoint = self.map.get_waypoint(loc)
        vehicle_lane = vehicle_waypoint.lane_id
        ego_lane = -(vehicle_lane + 2.0)

        dis_to_goal = get_distance(loc.x, loc.y, self._waypoints[-1][0], self._waypoints[-1][1])
        # print(self.dis_to_goal, dis_to_goal)
        if dis_to_goal >= self.dis_to_goal:
            desire_speed = DESIRE_SPEED
        else:
            desire_speed = DESIRE_SPEED * (dis_to_goal / self.dis_to_goal)
        if self.bp._goal_index == len(self._waypoints) - 1:
            target = 1
        else:
            target = -1

        if desire_speed > DESIRE_SPEED:
            desire_speed = DESIRE_SPEED
        elif desire_speed < 0:
            desire_speed = 0.0
        if speed > DESIRE_SPEED:
            speed = DESIRE_SPEED
        speed = speed / DESIRE_SPEED
        target_speed = desire_speed / DESIRE_SPEED
        # if show_message:
        #     print("speed: {}, target speed: {}".format(speed, target_speed))
        reward = -math.fabs(speed - target_speed)
        reward = 1.5 * reward + 0.6 + 0.2 * speed
        # reward += 0.3 *speed
        # reward = 2 * reward + speed
        # if self.bp.trans_reward == -1.0:
        #     reward = -1.0
        reward = reward + self.bp.trans_reward + self.lc_reward
        self.lc_reward = 0.0

        list_vehicles = self.world.get_actors().filter('vehicle.*')
        map_image = copy.copy(self.map_image.image)
        draw = ImageDraw.Draw(map_image)
        self.render_vehicles(draw, list_vehicles, self.map_image.world_to_pixel)
        angle = 0.0 if self.hero is None else rot.yaw + 90.0
        hero_location_screen = self.map_image.world_to_pixel(loc)
        resultIm = map_image.rotate(angle, center=hero_location_screen)
        resultIm_center = hero_location_screen
        box = (resultIm_center[0]-150, resultIm_center[1]-600, 
               resultIm_center[0]+150, resultIm_center[1]+300)
        cropIm = resultIm.crop(box)
        cropIm = cropIm.resize((80, 150), Image.ANTIALIAS)
        ob = cropIm

        # ob = [target, speed, target_speed, self.bp._state - 1.0, ego_lane, 
        #     d_lf, d_lr, d_cf, d_cr, d_rf, d_rr]
        # ob = np.array(ob)
        # print(ob)

        # waypoint = self.map.get_waypoint(loc, project_to_road=True)
        # angle_delta = rot.yaw - waypoint.transform.rotation.yaw
        # if angle_delta > 180:
        #     angle_delta -= 360
        # elif angle_delta < -180:
        #     angle_delta += 360
        # else:
        #     pass
        # if abs(angle_delta) > 60:
        #     reward = -1.0
        #     done = True
        # else:
        #     done = False
        done = False
        if abs(ego_lane) > 1:
            done = True
        closest_len, closest_index = behavioural_planner.get_closest_index(self._waypoints, [loc.x, loc.y])
        if closest_index == len(self._waypoints) - 1:
            done = True
            if speed > 0.2:
                reward = -1.0
        if dis_to_goal < 2.5:
            done = True
            if speed < 0.1:
                reward = 1.0
        if target_speed < 0.40 and speed < 0.05:
            done = True
            reward -= 0.1
        # if not target and speed < 0.05:
        #     done = True
        #     reward = -1.0
        
        # dis = loc.distance(vehicle_waypoint.transform.location)
        # if dis > 2.0 or closest_len > 5.0:
        #     reward = -1.0
        #     done = True
        # # done when running a red light
        # if stop == 1.0 and self.run_traffic_light:
        #     reward = -1.0
        #     done = True
        self.run_traffic_light = False

        reward = -1.0 if reward < -1.0 else reward
        reward = 1.0 if reward > 1.0 else reward
        if self.sensor is not None:
            if len(self.sensor.history) > 0:
                reward = -2.0
                done = True
        if self.collision_his != len(self.sensor.history):
            self.collision_times += 1
            self.collision_his = len(self.sensor.history)
        self.speed_sum += current_speed
        self.speed_step += 1
        if show_message:
            print("------------------------------>")
            print("reward: {:.3}, done: {}".format(reward, done))
            # print("return: {}".format([ob, reward, done]))
            # print("speed\ttarget_speed\tbp_state\tego_lane\td_lf\td_lr\td_cf\td_cr\td_rf\td_rr")
            tb = pt.PrettyTable()
            tb.field_names = ["target", "speed", "target_speed", "bp_state", "ego_lane"]
            tb.add_row([target, speed, target_speed, self.bp._state - 1.0, ego_lane])
            tb.set_style(pt.MSWORD_FRIENDLY)
            tb.float_format = "2.2"
            print(tb)
            # print("d_lf:{:.3}\td_cf:{:.3}\td_rf:{:.3}".format(d_lf, d_cf, d_rf))
            # print("d_lr:{:.3}\td_cr:{:.3}\td_rr:{:.3}".format(d_lr, d_cr, d_rr))
            print("collision times: {}".format(self.collision_times))
            print("<------------------------------")
        return ob, reward, done

    def render_vehicles(self, draw, list_v, world_to_pixel):
        """Renders the vehicles' bounding boxes"""
        corners = [carla.Location(x=-154.6, y=-208.4),
                   carla.Location(x=-153.0, y=-198.2),
                   carla.Location(x=-153.8, y=-197.8),
                   carla.Location(x=-155.3, y=-208.2),
                   carla.Location(x=-154.6, y=-208.4)]
        corners = [world_to_pixel(p) for p in corners]
        draw.polygon(corners, fill=COLOR_R)
        for v in list_v:
            # actor_wp = self.get_vehicle_wp(v)
            actor_v = v.get_velocity()
            actor_speed = math.sqrt(actor_v.x ** 2 + actor_v.y ** 2 + actor_v.z ** 2)
            color_depth = int(128 + 128 * actor_speed / DESIRE_SPEED)
            color_depth = 255 if color_depth > 255 else color_depth
            actor_speed_limit = v.get_speed_limit()
            percentage_v_dif = (1 - 60.0/actor_speed_limit) * 100
            self.traffic_manager.vehicle_percentage_speed_difference(v, percentage_v_dif)
            if v.attributes['role_name'] == 'hero':
                color = (0, 0, color_depth)
            else:
                color = (0, color_depth, 0)
            # Compute bounding box points
            bb = v.bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v.get_transform().transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            draw.polygon(corners, fill=color)

    def get_vehicle_wp(self, vehicle):
        loc = vehicle.get_location()
        waypoint = self.map.get_waypoint(loc, project_to_road=True)
        return waypoint

    def get_dis(self, vehicle, action):
        loc = vehicle.get_location()
        waypoint = self.map.get_waypoint(loc, project_to_road=True)
        dis = 0.0
        if action == 0:
            if waypoint.lane_id == -1:
                center = waypoint.get_right_lane()
                dis = loc.distance(center.transform.location)
                dis = 3.5 - dis
            if waypoint.lane_id == -2:
                center = waypoint.get_left_lane()
                dis = loc.distance(center.transform.location)
        elif action == 1:
            if waypoint.lane_id == -1:
                center = waypoint.get_right_lane()
                dis = -loc.distance(center.transform.location)
            if waypoint.lane_id == -2:
                center = waypoint.get_right_lane()
                dis = loc.distance(center.transform.location)
                dis = 3.5 - dis
            if waypoint.lane_id == -3:
                center = waypoint.get_left_lane()
                dis = loc.distance(center.transform.location)
        elif action == 2:
            if waypoint.lane_id == -2:
                center = waypoint.get_right_lane()
                dis = -loc.distance(center.transform.location)
            if waypoint.lane_id == -3:
                center = waypoint.get_right_lane()
                dis = loc.distance(center.transform.location)
                dis = 3.5 - dis

        return dis

    def destroy_hero(self):
        # if self.sensor is not None:
        #     # print(self.sensor.is_listening)
        #     self.sensor.stop()
        # actors = [self.sensor.sensor, self.hero]
        if self.sensor is not None:
            self.sensor.sensor.destroy()
        if self.hero is not None:
            self.hero.destroy()
        self.hero = None
        self.sensor = None
        # print("destroy hero and sensor done")

    def destroy(self):
        # print('\ndestroying actors')
        self.destroy_hero()
        actor_ids = []
        for v in self.world.get_actors().filter('vehicle*'):
            actor_ids.append(v.id)
        # print(actor_ids)
        # client = carla.Client('localhost', 2000)
        # client.set_timeout(10.0)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_ids])
        self.actor_list = []
        # self.lane = 1
        self.bp = None
        self.lp = None
        self.controller = None
        self.world.tick()
        self.collision_times = 0
        self.collision_his = 0
        self.speed_sum = 0.0
        self.speed_step = 0
        self.lane_change_times = 0
        self.count_time = 0
        # print('done.')

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self._global_plan = global_plan_gps
        self._global_plan_world_coord = global_plan_world_coord

    def _get_waypoints(self, global_plan_world_coord):
        waypoints = []
        for index in range(len(global_plan_world_coord)):
            waypoint = global_plan_world_coord[index][0]
            waypoints.append([waypoint.location.x, waypoint.location.y, DESIRE_SPEED])
        return waypoints
