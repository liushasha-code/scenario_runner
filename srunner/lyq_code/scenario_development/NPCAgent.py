"""
A npc vehicle using PID controller.

"""

import glob
import os
import sys

# carla 099 API is recommended
sys.path.append("/home/lyq/CARLA_simulator/CARLA_099/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_099/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_099/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# carla module
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import LocalPlanner








class BasicAgent():
    def __init__(self):
        pass

    def spawn_vehicle(self):
        """"""
        location =
        blueprint =

    def run_step(self):
        """"""
        pass



class NPCAgent:
    def __init__(self):
        pass

    def set_controller(self):
        """"""
        # generate route
        hop_resolution = 1.0
        dao = GlobalRoutePlannerDAO(self.world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        self.route = grp.trace_route(self.start_location, self.end_location)

        self.local_planner = LocalPlanner(self.ego_vehicle)
        self.local_planner.set_global_plan(self.route)

        print("PID controller is set.")

    def run_step(self):
        """"""



        control = self.local_planner.run_step()
        control
        pass



if __name__ == '__main__':

    agent = NPCAgent()



