__author__ = "Joshua W. Windle"
__version__ = "0.1"


class Sequential:
    def __init__(self, *layers):
        # Store layers in order
        self.layers = list(layers)

    def forward(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backprop(self, delta):
        # Backward pass through all layers (reverse order)
        for layer in reversed(self.layers):
            delta = layer.backprop(delta)
        return delta

    def update(self, optim):
        # Update all trainable layers
        for layer in self.layers:
            layer.update(optim)
