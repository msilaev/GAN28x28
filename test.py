import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import time
import torch.nn.functional as F

import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchviz import make_dot


if __name__ == "__main__":
    # Define the model using nn.Sequential

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  # Should print 'cuda' or 'cpu'

    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))

    model.to(device)



    # Forward pass through the model


    x = torch.randn(1, 8).to(device)
    y = model(x)

    # Visualize the computation graph
    #dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)

    dot.render("computation_graph", format="png")

    from torchsummary import summary

    summary(model, input_size=(1, 8))


