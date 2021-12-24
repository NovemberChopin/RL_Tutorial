import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicQNetwork(nn.Module):
    def __init__(self, input_size, action_size, args):
        super(BasicQNetwork, self).__init__()
        # print("normal weight")
        self.layer1 = nn.Linear(input_size, args.num_units_1)
        self.layer1.weight.data.normal_(0, 1)
        self.layer2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.layer2.weight.data.normal_(0, 1)
        self.layer_out = nn.Linear(args.num_units_2, action_size)
        self.layer_out.weight.data.normal_(0, 1)
    
    def forward(self, q_network_input):
        x = F.relu(self.layer1(q_network_input))
        x = F.relu(self.layer2(x))
        x = self.layer_out(x)
        x = F.relu(x)
        return x