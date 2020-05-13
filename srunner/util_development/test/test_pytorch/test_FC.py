
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dim = 20
action_dim = 3
width = 200

fc = nn.Sequential(
        nn.Linear(state_dim, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, action_dim)
        )

fc.to(device)

state = [x*1.0 for x in range(20)]

state_tensor = []
for item in state:
    item = torch.tensor([item])
    item = item.unsqueeze(0)
    state_tensor.append(item)

input_tensor = state_tensor[0]
for item in state_tensor[1:]:
    input_tensor = torch.cat((input_tensor, item), 1)

bbb = input_tensor.to(device)

result = fc(bbb)

print('d')

