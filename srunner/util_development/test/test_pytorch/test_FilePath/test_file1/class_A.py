import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelTest(object):

    def __init__(self):
        """"""
        # initial model
        model = TheModelClass()
        # initialize the optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # use relative path
        # torch.save(model.state_dict(), './model_state_dict.pth')
        name = 'relative_path'
        torch.save(model.state_dict(), './' + name + '_model_state_dict.pth')

        # absolute path
        name = 'absolute_path'
        path = '/home/lyq/PycharmProjects/scenario_runner/srunner/util_development/test/test_pytorch/test_FilePath/test_file1/testsave/'
        torch.save(model.state_dict(), path+name+'_model_state_dict.pth')

        print('d')


if __name__ == '__main__':
    model_test = ModelTest()


