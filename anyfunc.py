import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from sympy.abc import x  

# neural network arch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # input layer to first hidden layer
        self.fc2 = nn.Linear(20, 20)  # first hidden layer to second hidden layer
        self.fc3 = nn.Linear(20, 1)   # second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# input function
user_input = input("Vul een wiskundige formule in met x (bijv. '7*x**2 + 2*x + 3'): ")

def check_input_syntax(user_input):
    if 'x' in user_input:
        user_input = user_input.replace('x', '*x')
    if '^' in user_input:
        user_input = user_input.replace('^', '**')
    return user_input

user_input = check_input_syntax(user_input)

user_function = sp.sympify(user_input)

# dataset generation
x_values = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 200), dim=1)
y_values = torch.tensor([float(user_function.subs('x', val)) for val in x_values], dtype=torch.float32).view(-1, 1)

# instance of neural network, loss function, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

# setup
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(np.min(y_values.numpy()) - 1, np.max(y_values.numpy()) + 1)
true_function, = ax.plot(x_values.numpy(), y_values.numpy(), 'b^', markersize=5, label='True function')
nn_approximation, = ax.plot([], [], 'r-', lw=2, label='NN approximation')
ax.legend()

def init():
    nn_approximation.set_data([], [])
    return nn_approximation,

def animate(i):
    # training
    y_pred = net(x_values)
    loss = criterion(y_pred, y_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update nn approximation line
    nn_approximation.set_data(x_values.data.numpy(), y_pred.data.numpy())
    return nn_approximation,

ani = FuncAnimation(fig, animate, frames=np.arange(1, 100), init_func=init, blit=True, interval=50, repeat=False)

plt.show()
