"""
Test:
1. simulation time in carla
2. datetime usage
"""

import datetime
import psutil
import os

import traceback

from srunner.lyq_code.env.BasicEnv import BasicEnv


class TimeTest(BasicEnv):
    """

    """
    def __init__(self, town='Town03', host='localhost', port=2000, client_timeout=2.0):
        super(TimeTest, self).__init__(town=town, host=host, port=port, client_timeout=client_timeout)
        # use sync mode to test simulation time
        self.set_world(sync_mode=True, frame_rate=50.0, no_render_mode=False)
        self.world.tick()
        # test all available maps
        maps = self.client.get_available_maps()
        print(maps)

    def get_time(self):
        """"""
        mode = self.world.get_settings().synchronous_mode

        # get simulation time
        worldsnapshot = self.world.get_snapshot()
        timestamp = worldsnapshot.timestamp

        a = timestamp.frame
        b = timestamp.elapsed_seconds  # 仿真器中已经过去的时间
        c = timestamp.delta_seconds  # 距离上一个frame已经过去的时间
        d = timestamp.platform_timestamp  # 距离开机已经过去的时间
        f = datetime.datetime.fromtimestamp(d)  # 转换成标准格式的UTC时间

        dt = datetime.datetime.fromtimestamp(psutil.boot_time())  # 获得系统开机时间
        dt2 = datetime.datetime.fromtimestamp(d + psutil.boot_time())  # 计算仿真运行到现在的UTC时间

        e = datetime.datetime.now()  # 真实世界现在的时间

        g = e.timestamp()

        self.world.tick()

def main():
    try:

        test = TimeTest()
        #
        while True:
            test.get_time()


        # usage of datetime
        today = datetime.date.today()
        formatted_today = today.strftime('%Y%m%d')

    except:
        traceback.print_exc()
    finally:
        del test


if __name__ == '__main__':
    main()