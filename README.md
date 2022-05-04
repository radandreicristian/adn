# Attention Diffusion Network

PyTorch implementation of the Attention Diffusion Network from ["Structured Time Series Prediction without Structural Prior"](https://arxiv.org/pdf/2202.03539v1.pdf).

This project is a work in progress.

# Install

From pip:

`pip install adn-torch`

Or from the source code (in editable / developer mode): 
```
git clone https://github.com/radandreicristian/adn.git
cd adn
pip install -e .[dev]
```

# Usage

The package exposes the ADN model as an API. Example usage:


```
from adn import ADN

model = ADN(d_features = 1,
            d_hidden=32,
            d_feedforward=256,
            n_heads=8,
            p_dropout=0.1,
            batch_size=8,
            n_blocks=3,
            n_nodes=325)
```

The model takes 6 arguments in its forward pass - For both the source and the target 
sequence, the features, the temporal descriptors and the spatial descriptors. A 
detailed example is provided in the `adn/example.py` file.
