import numpy as np

from aiga.layers.dense import Dense
from aiga.activations import ReLU
from aiga.networks.sequential import Sequential
from aiga.losses.mse import MSE
from aiga.optimizers import SGD


# Simple linear dataset: y = 2x
x = np.random.randn(100, 1)
y = 2 * x


# Build network
net = Sequential(
    Dense(8, input_size=1),
    ReLU(),
    Dense(1)
)

loss_fn = MSE()
optimizer = SGD(lr=0.01)


# Training loop
for epoch in range(200):
    y_hat = net(x)

    # 
    loss = loss_fn.forward(y_hat, y)

    # compute ∂C/∂y_hat
    delta = loss_fn.backprop()

    # send ∂C/∂y_hat  
    net.backprop(delta)
    net.update(optimizer)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
