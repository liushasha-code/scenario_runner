CARLA版本问题梳理

# 1. 多版本carla使用

## 1.1 import carla  

针对直接使用终端和vscode等工具的用户,需要在终端中export carla路径

```
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-<VERSION>.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla
```

针对使用anaconda建立conda环境,搭配pycharm开发的用户,可以在脚本最前端加入如下代码来引入carla  
注意替换相应的目录,使用这种方法要保证anaconda环境的python=3.5 

```
from __future__ import print_function
import glob
import os
import sys

# using carla 098 as an example
sys.path.append("/{your carla unzip folder}/CARLA_098/PythonAPI/carla")
sys.path.append("/{your carla unzip folder}/CARLA_098/PythonAPI/carla/agents")
carla_path = '/{your carla unzip folder}/CARLA_098/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

```
## 1.2 server与egg版本一致
注意导入的.egg文件与实际打开的carla server版本应一致

# 2. 不同版本API可能出现的问题 
## 2.1 wait_for_tick功能
carla 096版本开始,wait_for_tick函数不再适用于synchronous mode，即

    world.settings.synchronous mode=True  
    
解决办法: 当world.synchronous mode=True时

## 2.2 spawn_actor问题

carla 098中，生成actor时有两种函数,分别是

    carla.World.spawn_actor(blueprint, transform, attach_to=None, attachment=Rigid)
以及  

    carla.World.try_spawn_actor(self, blueprint, transform, attach_to=None, attachment=Rigid)

`world.spawn_actor()`会在生成actor失败时返回错误,而`world.try_spawn_actor`则不会报错.

所以在生成关键actor时要使用`world.spawn_actor()`命令,在生成不重要的环境车辆等actor时,可以使用`world.try_spawn_actor`

在carla 098版本中，由于对于碰撞的检测机制不同,生成车辆时需设置一定的初始高度,通常大于１.0即可
否则会出现由于与地面干涉 无法正确生成车辆的情况

