import torch
import numpy as np
from matplotlib.animation import FuncAnimation
import sympy as sp
from sympy.abc import x
from sympy.parsing.sympy_parser import parse_expr

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # input layer to first hidden layer
        self.fc2 = nn.Linear(20, 40)  # first hidden layer to second hidden layer
        self.fc3 = nn.Linear(40, 1)   # second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# prompt user for input and preprocess the input function
user_input = input("Enter a mathematical function of x (e.g., 'sin(x) + x**2'): ")

# parse the user input using sympy's parse_expr for more complex functions
user_function = parse_expr(user_input, transformations='all')

# generate dataset
x_values = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 200), dim=1)
y_values = torch.tensor([float(user_function.subs(x, val.item())) for val in x_values], dtype=torch.float32).view(-1, 1)

# instantiate neural network, loss function, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)  # adjust learning rate as necessary

# plot setup
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(np.min(y_values.numpy()) - 1, np.max(y_values.numpy()) + 1)
true_function, = ax.plot(x_values.numpy(), y_values.numpy(), 'b^', markersize=5, label='true function')
nn_approximation, = ax.plot([], [], 'r-', lw=2, label='nn approximation')
ax.legend()

def init():
    nn_approximation.set_data([], [])
    return nn_approximation,

def animate(i):
    # perform training step
    y_pred = net(x_values)
    loss = criterion(y_pred, y_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update the nn approximation line
    nn_approximation.set_data(x_values.data.numpy(), y_pred.data.numpy())
    return nn_approximation,

# create animation
ani = FuncAnimation(fig, animate, frames=np.arange(1, 100), init_func=init, blit=True, interval=50, repeat=False)

plt.show()
