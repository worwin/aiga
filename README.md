# aiga

I'm writing a project from scratch to experiment with machine learning. I want to first program a feedforward network from scratch. I want this to be highly modularized and to be a library for those who want to learn about machine learning. Its focus isn't on efficiency or any of that. It's about understanding the underlying principles of the mathematics behind the models and how they are written. The language this is written in will be Python unless otherwise it is deemed necessary to deviate from this. 

Project to code several machine learning and neural network models from scratch. 

## Optimizers

Adaptive optimizers currently available:

- AdaGrad
- RMSProp
- Adam
- AdamW (Adam with decoupled weight decay)

## Layers

- `Dropout` is available as a regularization layer.
- Dropout is active only during training and is disabled during inference/evaluation.
- It uses inverted dropout scaling (`x * mask / (1 - p)`) during training.
- In feedforward stacks, it is typically placed after activation layers.

# Sequential Networks

### Back-Propagation

