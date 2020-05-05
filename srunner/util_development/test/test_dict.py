"""
test dict usage for developing test_TrafficFlow.
"""



npc_info = {
            "left": {
                "count": 0,
                "list": ['a'],
                "nearby_npc": ['b'],  # use queue to manage
            },
            "right": {
                "count": None,
                "list": [],
                "nearby_npc": [],  # use queue to manage
            },
            "straight": {
                "count": 0,
                "list": [],
                "nearby_npc": [],  # use queue to manage
            }
        }

x = npc_info['left']['list']
z = npc_info['left']['list'] = ['c']

vv = npc_info['right']['count']

print('d')

