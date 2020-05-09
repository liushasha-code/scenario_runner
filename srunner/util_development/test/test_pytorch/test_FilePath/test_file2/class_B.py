"""


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from .. test_file1.class_A import TheModelClass
from test_file1.class_A import TheModelClass
from test_file1.class_A import ModelTest


class B:
    def __init__(self):
        """"""
        # initial model
        model = TheModelClass()
        # initialize the optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # use relative path
        name = 'relative_path'
        torch.save(model.state_dict(), './'+name+'_model_state_dict.pth')

        # absolute path
        name = 'absolute_path'
        path = '/home/lyq/PycharmProjects/scenario_runner/srunner/util_development/test/test_pytorch/test_FilePath/test_file2/'
        torch.save(model.state_dict(), path+name+'_model_state_dict.pth')

        print('d')


if __name__ == '__main__':
    # model_test = B()

    # test when class is in original root
    model_test = ModelTest()
