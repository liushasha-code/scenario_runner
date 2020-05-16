"""
Test usage of tuple
"""

from collections import namedtuple


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

t = Transition()

a = len(t)


print('d')
