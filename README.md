# Computer Vision Toolkits

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-2.x-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![FiftyOne](https://img.shields.io/badge/FiftyOne-ff9d00)](https://voxel51.com/fiftyone/)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

A collection of reusable computer-vision toolkits.

## Install

The whole repository can be installed directly from GitHub with `pip`:

```bash
pip install git+https://github.com/mwangi-clinton/computer-vision-toolkit.git
```

If you use [Poetry](https://python-poetry.org/):

```bash
poetry add git+https://github.com/mwangi-clinton/computer-vision-toolkit.git
```

### Install from a local clone

```bash
git clone https://github.com/mwangi-clinton/computer-vision-toolkit.git
cd computer-vision-toolkit
poetry install
```

## Available toolkits

| Toolkit | Import | Description |
|---------|--------|-------------|
| `gradcam` | `from gradcam import GradCAM, GuidedBackprop` | Grad-CAM / Guided Backpropagation visual explanations for CNNs. |

## Quick start

```python
import torchvision
from gradcam import GradCAM, visualize_gradcam

model = torchvision.models.resnet50(weights="DEFAULT")
model.eval()

cam = GradCAM(model, target_layer=model.layer4[-1])
```

See the `gradcam/` directory for the full Grad-CAM README and examples.

## Adding a new toolkit

1. Create a new top-level directory, e.g. `my_toolkit/`, containing an `__init__.py`.
2. Add it to the `packages` list in `pyproject.toml`:

   ```toml
   packages = [
       { include = "gradcam" },
       { include = "my_toolkit" },
   ]
   ```

3. Run `poetry install` (or `pip install .`) to make it importable.
