
### Installing dependencies
Install the required Python packages using:

```shell
pip install -r requirements.txt
```

## Directory Structure
The repository is organized as follows:
```
Sabre
├── artifacts             - Output directory for experiment artifacts.
│   ├── attachments        - Directory for any attachments (e.g., figures, graphs).
│   ├── logs               - Experiment log files.
│   └── saved_models       - Directory for storing trained model weights.
├── core                - Implementation of core functionalities.
│   ├── attacks            - Methods for generating adversarial attacks.
│   │   ├── fgsm.py           - Fast Gradient Sign Method attack.
│   │   ├── pgd.py            - Projected Gradient Descent attack.
│   │   └── ... 
│   ├── defenses           - Defense mechanisms.
│   │   └── sabre.py          - Core implementation of the Sabre defense framework.
├── datasets            - Scripts for loading and preprocessing datasets.
│   ├── cifar10.py         - Handles the CIFAR-10 dataset.
│   ├── mnist.py           - Handles the MNIST dataset.
│   └── ...
├── experiments         - Experiment scripts and configurations.
│   └── images             - Image-based experiments
│   │   ├── cifar10.py        - Experiments on the CIFAR-10 dataset.
│   │   ├── mnist.py          - Experiments on the MNIST dataset.
│   │   └── ...
│   ├── setup.py           - Experiment setup and execution.
│   └── utils.py           - Utility functions for experiments.
├── models              - Model definitions.
│   ├── cifar10.py         - CIFAR-10 classification model.
│   ├── mnist.py           - MNIST classification model.
│   ├── denoise.py         - Denoising model implementation.
│   └── helpers.py         - Helper functions for model management.
│   └── ...
├── notebooks           - Jupyter notebooks for demonstrations.
│   ├── cifar10.py         - CIFAR-10 notebook.
│   ├── mnist.py           - MNIST notebook.
│   └── ...
├── utils               - General utility functions and helpers.
│   └── log.py             - Logging utilities for monitoring and debugging.
├── requirements.txt    - Lists the Python package dependencies.
└── README.md           - Provides an overview and instructions for the repository.
```

## Experiments

### Robust classifiers
This section provides instructions for running experiments to evaluate the robustness of the models implemented in the framework. 
We focus on assessing performance against adversarial examples and viability across different datasets. 

#### MNIST
```shell
python3 experiments/images/mnist.py
```

#### CIFAR-10
```shell
python3 experiments/images/cifar10.py
```




