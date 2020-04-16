# a easy version to package npc agent


# todo: inherit NPCagent if necessary
class agent_manager(object):

    def __init__(self, vehicle):
        """
        Generate a manager instance to manage an agent.
        """
        self.agent = None  # todo: add
        self.vehicle = vehicle
        self.controller = None

    def get_state(self):

        self.

    def apply_control(self):
        self.vehicle.applycontrol()


    def set_route(self, route):

        self.route = route



