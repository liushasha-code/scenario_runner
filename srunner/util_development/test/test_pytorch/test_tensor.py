import torch

# ==================================================
# test load dict




# ==================================================
# test save path

path = './DQN_save/'

name = 'longitudinal_control'

save_path = path + name + '/dqn.pth'

print('d')

# ==================================================
# test torch sensor dimension

a = 1.0
a_tensor = torch.tensor([a])
a_tensor = a_tensor.unsqueeze(0)

b = 2.0
b_tensor = torch.tensor([b])
b_tensor = b_tensor.unsqueeze(0)

c = 3.0
c_tensor = torch.tensor([b])
c_tensor = c_tensor.unsqueeze(0)

state_list_tensor = []
test = [a, b, c]
for state in test:
    state = torch.tensor([state])
    state = state.unsqueeze(0)
    # state = state + 1
    state_list_tensor.append(state)

aaa = state_list_tensor[1:]
# print(state_list_tensor)
print("d")

# test cat function
tensor = state_list_tensor[0]
for item in state_list_tensor[1:]:
    tensor = torch.cat((tensor, item), 1)

print("d")
