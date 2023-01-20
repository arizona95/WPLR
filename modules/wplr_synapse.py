import torch
from torch import nn
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
from .base_module import BaseModule
from Args import Args
args = Args()

class WPLRSynapse(BaseModule) :
    """
    info
    """
    def __init__(self, pre_neuron, post_neuron, condition = True):
        super().__init__()
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = nn.Parameter(torch.ones((self.pre_neuron.space_num, self.post_neuron.space_num)))
        self.weight.requires_grad = False
        self.sleep = True
        self.value = 100
        self.condition = 1 if condition is True else -1

    def hebb_training(self):
        if self.sleep == False :
            x = self.pre_neuron.X
            y = self.post_neuron.X
            dw = torch.matmul(self.pre_neuron.D[x].reshape(-1,1), self.post_neuron.D[y].reshape(-1,1).T)
            self.weight += dw * self.value
            self.weight *= 1-args.forget_rate
            self.weight += args.forget_rate

        self.sleep = True
        return

    def propagate_probability(self):
        if self.pre_neuron.sleep == False :
            self.sleep = False
            self.post_neuron.propagate_probability(self.weight[self.pre_neuron.X]*self.condition)

    def visual_synapse(self, angle_x=90, angle_y=0):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20,20))

        # Make data.
        X = np.arange(0, 1, 1/self.pre_neuron.space_num)
        Y = np.arange(0, 1, 1/self.post_neuron.space_num)
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        ax.view_init(angle_x, angle_y)
        surf = ax.plot_surface(X, Y, self.weight.T.numpy(), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        #ax.set_zlim(-1.01, 3.01)
        ax.set_xlabel(self.pre_neuron.name, fontsize=32)
        ax.set_ylabel(self.post_neuron.name, fontsize=32)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        return




