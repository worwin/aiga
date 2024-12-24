aiga/
├── __init__.py
├── layers/
│   ├── __init__.py
│   ├── dense.py
│   └── convolution.py
├── activations/
│   ├── __init__.py
│   ├── relu.py
│   ├── sigmoid.py
│   └── softmax.py
├── losses/
│   ├── __init__.py
│   ├── mse.py
│   ├── cross_entropy.py
│   └── hinge.py
├── optimizers/
│   ├── __init__.py
│   ├── sgd.py
│   └── adam.py
├── metrics/
│   ├── __init__.py
│   ├── accuracy.py
│   ├── precision.py
│   └── recall.py
├── utils/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── normalization.py
│   └── logger.py
├── models/
│   ├── __init__.py
│   ├── feedforward.py
│   └── sequential.py
├── visualization/
│   ├── __init__.py
│   └── plots.py
├── tests/
│   ├── __init__.py
│   ├── test_layers/
│   │   ├── __init__.py
│   │   ├── test_dense.py
│   │   └── test_convolution.py
│   ├── test_activations/
│   │   ├── __init__.py
│   │   ├── test_relu.py
│   │   └── test_sigmoid.py
│   ├── test_losses/
│   │   ├── __init__.py
│   │   ├── test_mse.py
│   │   └── test_cross_entropy.py
│   ├── test_feedforward.py
│   └── test_utils.py
├── docs/
│   ├── tutorials/
│   │   ├── getting_started.md
│   │   ├── feedforward_intro.md
│   │   └── loss_functions.md
│   └── api_reference.md
├── examples/
│   ├── __init__.py
│   ├── simple_ffn.py
│   └── custom_model.py
├── README.md
├── setup.py
├── requirements.txt
├── LICENSE
└── .gitignore