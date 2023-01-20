import torch
from torch import nn

class BaseModule(nn.Module) :
    """
    creates base module from wplr
    """

    def __init__(self):
        super().__init__()

