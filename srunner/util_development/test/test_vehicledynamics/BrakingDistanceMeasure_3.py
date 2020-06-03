"""
    <2020.02.13>
    liuyuqi

    BrakingDistanceMeasure_3.py is a fixed version of BrakingDistanceMeasure_2.py

    To isolate lateral offset function and avoid vibration.

    ==================================================
    todo: set friction and dynamics to verify vehicle sliding and drifting.

    Encapsulated all functions into a class.

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
import traceback
import weakref

from srunner.tools.scenario_helper import *
from srunner.challenge.utils.route_manipulation import interpolate_trajectory
from agents.navigation.local_planner import LocalPlanner

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# ==================================================
# public methods
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

class BrakingDistanceMeasure(object):
    """
        This class contain major methods of measuring braking
        distance of Lincoln Mkz at different speed(km/h).

        About experiment:

        Town04 has a Long straight lane over bridge.
        Ego vehicle can be spawn at the end of lane.

        Some desired spawn points in Town04 are listed below:

        Backup locations:
            After the traffic light:     (x=-351, y=33.5, z=0.5)
            Before the traffic light:    (x=-390, y=33.6, z=0.5)

        Other side of the highway:
            (x=338, y=14, z=0.5) Rotation(pitch=1.194088, yaw=-179.022064, roll=0.000000)
            end point: (x=-363, y=9.0, z=0.5)

        todo: Other functions can be added to measure vehicle physics.
    """

    # spawn location and end location, described as carla.Location
    start_location = carla.Location(x=338, y=14, z=0.5)
    end_location = carla.Location(x=-361, y=9.0, z=0.5)
    stop_location = carla.Location(x=70, y=9.87, z=10.98)

    # test parameters
    speed_lower_limit = 30
    speed_upper_limit = 50
    speed_resolution = 10
    # repetition numbers
    repetition = 5

    def __init__(self, host = 'localhost', port = 2000, client_timeout = 2.0):

        # setup client
        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)
        self.world = self.client.get_world()
        # setup world
        self.setup_world()
        # set the spectator at static location
        self.set_static_spectator()
        # set test, ego vehicle is set in this function
        self.set_test()
        # set ego vehicle
        self.set_ego_vehicle()

        # about result and test setting
        self.result_stat = []
        self.result_detail = []

    def setup_world(self, town_name = 'Town04'):
        """
            Setup carla world with certain parameters.
        :return:
        """
        # get map
        self.map = self.world.get_map()
        # map verification
        if self.map != town_name:
            self.world = self.client.load_world(town_name)
            self.world.tick()

        # world parameters
        frame_rate = 20.0
        # set worldsettings
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / frame_rate
        self.world.apply_settings(settings)

        # set weather
        # use ClearNoon
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

    def draw_waypoints(self, waypoint_list, z=0.5, lifetime = -1.0):
        """
        Draw a list of waypoints at a certain height given in z.

        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
        :return:
        """
        for w in waypoint_list:
            t = w.transform
            begin = t.location + carla.Location(z=z)
            angle = math.radians(t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            self.world.debug.draw_arrow(begin, end, arrow_size=0.3,
                                   color=carla.Color(0, 255, 0), life_time=lifetime)

    def delete_vehicles(self):
        """
            delete all vehicles in selected world
        """
        actors = self.world.get_actors()
        for actor in actors:
            # if actor.type_id == vehicle.*
            id = actor.type_id
            actor_type = id.split(".")
            # destroy all vehicles
            if actor_type[0] == "vehicle":
                actor.destroy()
                print("vehicles are deleted")

    def set_static_spectator(self):
        """
            Set a static spectator for this experiment.
        """
        # several view

        # behind view
        # spectator_transform = carla.Transform(carla.Location(x=380, y=14, z=50),
        #                                       carla.Rotation(pitch=-35, yaw=-180, roll=0))

        # over the bridge view
        # spectator_transform = carla.Transform(carla.Location(x=-3, y=53, z=60),
        #                                       carla.Rotation(pitch=-40, yaw=-35, roll=0))

        # side-behind view
        spectator_transform = spectator_transform = carla.Transform(carla.Location(x=150, y=80, z=50),
                                                      carla.Rotation(pitch=-15, yaw=-110, roll=0))

        # apply transform
        spectator = self.world.get_spectator()
        spectator.set_transform(spectator_transform)

    # check if ego vehicle reach stop waypoint
    # def stop_waypoint_check(self):
    #     # distance check radius
    #     radius = 3
    #     # distance
    #     distance = self.ego_vehicle.get_location().distance(self.stop_location)
    #     if distance <= radius:
    #         self.reach_stop_waypoint = True

    def reset(self):
        # clear vehicles in the world
        # self.delete_vehicles()

        # clear local planner instance
        del self.local_planner # ego vehicle will be deleted by this line as well
        self.world.tick()
        # set test again
        self.set_ego_vehicle()

    def set_test(self):
        """
            Purpose:
            1. Generate some basic info of test.
            2. Generate route and plot

            This method is not included in the reset method.
        """

        # plot the origin of coordinate frame
        # todo: record and plot local coordinate frame for visualization

        # set initial spwan waypoint for ego vehicle
        self.start_waypoint = self.world.get_map().get_waypoint(self.start_location, project_to_road=True,
                                                                lane_type=carla.LaneType.Driving)
        self.end_waypoint = self.world.get_map().get_waypoint(self.end_location, project_to_road=True,
                                                              lane_type=carla.LaneType.Driving)
        self.stop_waypoint = self.world.get_map().get_waypoint(self.stop_location, project_to_road=True,
                                                               lane_type=carla.LaneType.Driving)

        # get lane info
        self.start_lane_id = self.start_waypoint.lane_id
        self.left_lane_id = self.start_waypoint.get_left_lane().lane_id
        self.right_lane_id = self.start_waypoint.get_right_lane().lane_id

        # generate route
        hop_resolution = 1.0
        dao = GlobalRoutePlannerDAO(self.world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        self.route = grp.trace_route(self.start_location, self.end_location)
        # plot route
        if_plot_route = False
        if if_plot_route:
            self.route_waypoint_list = []
            for w in self.route:
                self.route_waypoint_list.append(w[0])
            self.draw_waypoints(waypoint_list=self.route_waypoint_list)

    def set_ego_vehicle(self):
        """
            Set ego vehicle at certain waypoint,
            and apply route for vehicle.
        :param start_waypoint:
        """
        # setup ego vehicle default as Lincoln Mkz
        blueprint_library = self.world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        bp.set_attribute('role_name', 'hero')
        # try spawn
        self.ego_vehicle = self.world.try_spawn_actor(bp, self.start_waypoint.transform)
        self.world.tick()

        # set initial control to keep vehicle static
        control = self.ego_vehicle.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.ego_vehicle.apply_control(control)

        # set local planner
        # todo: current experiment run on same route, if using different setting, encapsule set_route into run_step
        self.local_planner = LocalPlanner(self.ego_vehicle)
        self.local_planner.set_global_plan(self.route)

        # set flag variable
        # status flag
        self.reach_target_speed = False
        # self.stop_waypoint
        self.reach_stop_waypoint = False

    def get_lateral_offset(self, vehicle):
        """
            calculate lateral offset respect to local
            todo: add lane invasion sensor to check if vehicle is in vibration and off the lane

        :param vehicle: keep input parameter
        :return: scalar offset value
        """

        # get current location
        current_location = vehicle.get_transform().location

        # get waypoint of original lane
        local_waypoint = vehicle.get_world().get_map().get_waypoint(current_location,
                            project_to_road=True, lane_type=carla.LaneType.Driving)

        local_lane_id = local_waypoint.lane_id
        lateral_offset = None
        lane_status = None
        if local_lane_id == self.start_lane_id:
            lateral_offset = current_location.distance(local_waypoint.transform.location)
            lane_status = 'original lane'
        elif local_lane_id == self.left_lane_id:
            original_lane_waypoint = local_waypoint.get_right_lane()
            lane_status = 'left'
            lateral_offset = current_location.distance(original_lane_waypoint.get_transform().location)
        elif local_lane_id == self.right_lane_id:
            original_lane_waypoint = local_waypoint.get_left_lane()
            lateral_offset = current_location.distance(original_lane_waypoint.get_transform().location)
            lane_status = 'right'

        return lateral_offset, lane_status

    def run_once(self, target_speed):
        """
            Run once with fixed specified target speed.
        """
        # set target speed of local planner
        self.local_planner._target_speed = target_speed
        # get initial speed
        speed = get_speed(self.ego_vehicle)
        max_speed = speed
        # max lateral offset
        max_lateral_offset_acc = 0.0
        max_lateral_offset_braking = 0.0

        # the status of if off lane
        off_lane_status = False

        # accelerating
        while True:
            # record time
            # world_snapshot = self.world.get_snapshot()
            # timestamp = world_snapshot.timestamp
            # print('========================================')
            # print('frame: ', timestamp.frame)

            # using local planner to get a standard PID control command
            # todo: fix parameters of lateral PID controller
            control = self.local_planner.run_step()

            if control.steer >= 0.05:
                # print(control.steer)
                print(control.steer)
                print(speed)

            # abondon lateral control when speed > 50km/h
            # if speed > 50:
            #     # control = self.accelerating(self.ego_vehicle)
            #     control.steer = 0.0

            # apply control to ego vehicle
            self.ego_vehicle.apply_control(control)

            # check and print control command
            # control = self.ego_vehicle.get_control()
            # print('throttle: ', control.throttle)
            # print('brake: ', control.brake)
            # print('steer: ', control.steer)
            # print('========================================')

            # update speed
            speed = get_speed(self.ego_vehicle)
            # print('speed: ', speed, 'km/h')
            # update max speed
            if speed >= max_speed:
                max_speed = speed

            # check if off lane
            # [lateral_offset, lane_status] = self.get_lateral_offset(self.ego_vehicle)
            # if lane_status != 'original lane' and not off_lane_status:
            #     off_lane_status = True
            #     print('Vehicle is off lane when accelerating!')
            # # update max lateral_offset
            # if lateral_offset > max_lateral_offset_acc:
            #     max_lateral_offset_acc = lateral_offset

            # check if ego vehicle has reached target speed
            if speed >= target_speed:
                self.reach_target_speed = True

            # check if ego vehicle has reached stop location
            radius = 3 # radius to target waypoint
            # get distance
            distance = self.ego_vehicle.get_location().distance(self.stop_location)
            if distance <= radius:
                self.reach_stop_waypoint = True

            # update status
            if self.reach_target_speed and self.reach_stop_waypoint:
                # record final speed
                final_speed = speed
                print('max speed reached, final speed is: ', final_speed)
                # record max speed location
                max_speed_location = self.ego_vehicle.get_location()
                print('final location is: ', max_speed_location)
                # braking by setting target speed to 0
                self.local_planner._target_speed = 0
                # reset off lane status
                off_lane_status = False
                break

        # brake to stop
        while self.reach_target_speed and self.reach_stop_waypoint and speed > 0:
            # get local planner control
            control = self.local_planner.run_step()
            # fix lateral control when speed is high
            # if speed >= 30:
            #     control.steer = 0.0
            control.steer = 0.0
            # apply control
            self.ego_vehicle.apply_control(control)

            # update speed
            speed = get_speed(self.ego_vehicle)
            # print('speed: ', speed, 'km/h')

            # # check if off lane
            # [lateral_offset, lane_status] = self.get_lateral_offset(self.ego_vehicle)
            # if lane_status != 'original lane' and not off_lane_status:
            #     off_lane_status = True
            #     print('Vehicle is off lane when braking!')
            # # update max lateral_offset
            # if lateral_offset > max_lateral_offset_braking:
            #     max_lateral_offset_braking = lateral_offset

        # measure braking distance after finish braking
        if self.reach_target_speed and self.reach_stop_waypoint and speed == 0:
            # hold static after running
            control = self.ego_vehicle.get_control()
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            self.ego_vehicle.apply_control(control)
            # store final location
            stop_location = self.ego_vehicle.get_location()
            # scalar distance, ignore lane direction
            # todo: consider distance along the lane
            braking_distance = max_speed_location.distance(stop_location)
        else:
            braking_distance = None
            print("Error occured! Distance is not recorded.")

        # update server???
        # self.world.tick()
        time.sleep(2.0)

        # result of single run
        single_run_result = {
            'final_speed': final_speed,
            'braking_distance': braking_distance,
            'max_lateral_offset_acc': max_lateral_offset_acc,
            'max_lateral_offset_braking': max_lateral_offset_braking,
            'off_lane_status': off_lane_status,
        }

        return single_run_result


    def run(self):
        # group number as dimension
        speed_group_num = len(range(self.speed_lower_limit, self.speed_upper_limit, self.speed_resolution))
        # create result store array
        result_mean = np.zeros((2, speed_group_num))
        result_var = np.zeros((2, speed_group_num))

        # iter different speed
        for i, target_speed in enumerate(range(self.speed_lower_limit, self.speed_upper_limit, self.speed_resolution)):
            final_speed_list = []
            braking_distance_list = []
            # max_lateral_offset_acc_list = []
            # max_lateral_offset_braking = []

            # run multiple times for each speed and calculate mean and var
            for repetition in range(self.repetition):

                print('==================================================')
                print('Target speed is: ', target_speed, 'km/h, repetition: ', repetition)
                # run route once
                single_run_result = self.run_once(target_speed)
                # print result of this run
                # print('==================================================')
                print('Final speed is ', single_run_result['final_speed'], ' km/h.')
                print('Braking distance: ', single_run_result['braking_distance'],' m')
                print('max_lateral_offset_acc', single_run_result['max_lateral_offset_acc'])
                print('max_lateral_offset_braking', single_run_result['max_lateral_offset_braking'])

                if single_run_result['off_lane_status'] == True:
                    print('Vehicle invaded other lane, vebration might occur, result will not be stored.')
                else:
                    # store result
                    final_speed_list.append(single_run_result['final_speed'])
                    braking_distance_list.append(single_run_result['braking_distance'])

                print('==================================================')
                # reset test and prepare for next run
                self.reset()

            # select not-None data
            final_speed_list = [x for x in final_speed_list if x is not None]
            braking_distance_list = [x for x in braking_distance_list if x is not None]

            # process data of one specified speed
            final_speed_list = np.array(final_speed_list)
            final_speed_mean = np.mean(final_speed_list)
            final_speed_var = np.var(final_speed_list)

            braking_distance_list = np.array(braking_distance_list)
            braking_distance_mean = np.mean(braking_distance_list)
            braking_distance_var = np.var(braking_distance_list)

            result_mean[0, i] = final_speed_mean
            result_mean[1, i] = braking_distance_mean

            result_var[0, i] = final_speed_var
            result_var[1, i] = braking_distance_var


        # analysis final results and print
        # self.reslt_analysis()

        # print final result
        print('==================================================')
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('**************************************************')
        print('Test finished. Total repetition is: ', self.repetition)
        for i, target_speed in enumerate(range(self.speed_lower_limit, self.speed_upper_limit, self.speed_resolution)):
            print('Target speed is ', target_speed, ' km/h')
            print('Final Speed Mean: ', result_mean[0, i], 'Final Speed Variation: ', result_var[0, i])
            print('Braking Distance Mean', result_mean[1, i], 'Braking Distance Variation:', result_var[1, i])
        print('**************************************************')
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('==================================================')

        # print(braking_distance_list)

def main():
    try:
        BrakingDistanceTest = BrakingDistanceMeasure()
        BrakingDistanceTest.run()
        print(BrakingDistanceTest.result_list)
    except:
        traceback.print_exc()
    finally:
        del BrakingDistanceTest

if __name__ == '__main__':
    main()
