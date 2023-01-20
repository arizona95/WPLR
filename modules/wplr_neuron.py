import torch
from torch import nn
import numpy as np
from Args import Args
args = Args()
from .base_module import BaseModule


class WPLRNeuron(BaseModule) :
    """
    info
    """

    def __init__(self, neuron_name, neuron_type=0, space_num=args.space_num):
        """
        type = 0 : input neuron
        type = 1 : hidden neuron
        tupe = 2 : output neuron
        type = 3 : feedback neuron
        """
        super().__init__()
        self.name = neuron_name
        self.state = nn.Parameter(torch.zeros(args.state_num))
        self.space_num = space_num
        self.X = 0
        self.type = neuron_type
        self.sleep = True
        self.awake = False

        self.node_mode = "basic"
        self.initialize()
        self.debug_mode = False
        self.debug_info = {
            "value_history":list(),
        }


    def initialize(self):
        if self.node_mode == "basic" :
            self.P = torch.from_numpy(np.ones(self.space_num))
            distance = np.zeros((self.space_num,self.space_num))
            for i in range(self.space_num) :
                for j in range(self.space_num) :
                    distance[i][j] = self.space_num*((min(abs(i-j), self.space_num-abs(i-j))/self.space_num)**2)
            distance = np.exp(-distance)/10
            self.D = torch.from_numpy(distance)

            self.P.requires_grad = False
            self.D.requires_grad = False

    def init_probability(self):
        self.P = torch.from_numpy(np.ones(self.space_num))
        self.P.requires_grad = False

    def init_node(self):
        self.init_probability()
        self.init_sleep()
        self.debug_mode = False
        self.debug_info = {
            "value_history":list(),
        }

    def propagate_probability(self, probability):
        self.awake = True
        self.P += probability

    def input(self, value):
        self.X = value
        self.sleep = False

    def init_sleep(self):
        self.sleep = True
        self.awake = False

    def sampling(self):
        self.sleep = not self.awake
        self.awake = False
        if self.sleep == False :
            en = self.P.numpy()
            pr = np.exp(en- np.max(en))
            self.X = np.random.choice(range(self.space_num), 1, p=pr/sum(pr))[0]
            self.debug_info["value_history"].append(self.X)

    def debug(self, debug_mode):
        self.debug_mode = debug_mode


