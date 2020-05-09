"""
Test getting current file name of python script.
"""

import os
import sys

# 绝对路径
print(__file__)
print(sys.argv[0])

# 文件名
print(os.path.basename(__file__))
print(os.path.basename(sys.argv[0]))

class A:
    def __init__(self):
        print('A')
        print(self.__class__.__name__)


class B:
    def __init__(self):
        self.a = A()
        print('B')
        print(self.__class__.__name__)

if __name__ == '__main__':
    b = B()

    print('d')