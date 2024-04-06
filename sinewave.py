import torch
import numpy as np
from matplotlib.animation import FuncAnimation

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20) 
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# generate a sine wave dataset
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 200), dim=1)
y = torch.sin(x)

# instantiate the network, loss function, and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.5, 1.5)
true_function, = ax.plot(x.numpy(), y.numpy(), 'b^', markersize=5, label='True function')
nn_approximation, = ax.plot([], [], 'r-', lw=2, label='NN approximation')
ax.legend()

def init():
    nn_approximation.set_data([], [])
    return nn_approximation,

def animate(i):
    # training step
    y_pred = net(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update the NN approximation line
    nn_approximation.set_data(x.data.numpy(), y_pred.data.numpy())
    return nn_approximation,

ani = FuncAnimation(fig, animate, frames=np.arange(1, 100), init_func=init, blit=True, interval=50, repeat=False)

plt.show()
