import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# define model
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

# define model
class modelclass_2(nn.Module):
    def __init__(self):
        super(modelclass_2, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# initial model
model = TheModelClass()

# initialize the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# print the model's state_dict
print("model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, '\t', model.state_dict()[param_tensor].size())

print("\noptimizer's state_dict")
for var_name in optimizer.state_dict():
    print(var_name, '\t', optimizer.state_dict()[var_name])

print("\nprint particular param")
print('\n', model.conv1.weight.size())
print('\n', model.conv1.weight)

print("------------------------------------")
torch.save(model.state_dict(), './model_state_dict.pt')

# ==================================================
# test load dict when NN with different structure
model_3 = modelclass_2()

model_3.load_state_dict(torch.load('./model_state_dict.pt'))

print('d')
# ==================================================



"""


# model_2 = TheModelClass()
# model_2.load_state_dict(torch.load('./model_state_dict'))
# model.eval()
# print('\n',model_2.conv1.weight)
# print((model_2.conv1.weight == model.conv1.weight).size())
## 仅仅加载某一层的参数
conv1_weight_state = torch.load('./model_state_dict.pt')['conv1.weight']
print(conv1_weight_state == model.conv1.weight)

model_2 = TheModelClass()
model_2.load_state_dict(torch.load('./model_state_dict.pt'))
model_2.conv1.requires_grad = False
print(model_2.conv1.requires_grad)
print(model_2.conv1.bias.requires_grad)


"""

