# Usage

## Installation with Pip

The project can be installed directly with pip:

```
pip install git+https://github.com/saifullah3396/torchxai.git
```

## Installation from Source

Clone the repository:

```
git clone https://github.com/saifullah3396/torchxai.git
```

Install the project with pip on source directory

```
pip install -e .
```

## Usage

Import the required metric, such as completeness

```
from torchxai.metrics.axiomatic import completeness
from captum.attr import Saliency

# ImageClassifier takes a single input tensor of images Nx3x32x32,
# and returns an Nx10 tensor of class probabilities.
net = ImageClassifier()

saliency = Saliency(net)
input = torch.randn(2, 3, 32, 32, requires_grad=True)
baselines = torch.zeros(2, 3, 32, 32)

# Computes saliency maps for class 3.
attribution = saliency.attribute(input, target=3)

# define a perturbation function for the input
# Computes completeness score for saliency maps
completeness_score = completeness(net, input, attribution, baselines)
```
