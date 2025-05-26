# Code Examples from (and for) Jonathan's Videos

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/jonathandinu?style=social)](https://x.com/jonathandinu)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCi0Hd3U6xb4V0ApUhAIfu9Q?color=%23FF0000&logo=youtube&style=flat-square)](https://www.youtube.com/channel/UCi0Hd3U6xb4V0ApUhAIfu9Q)

<!-- |   May 27, 2025 | [Training your First Neural Network (with PyTorch)](notebooks/pytorch-intro.ipynb)                                                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/youtube/blob/main/notebooks/pytorch-intro.ipynb) | [![YouTube Video Views](https://img.shields.io/youtube/views/c7bbjqiBy38)](https://youtu.be/c7bbjqiBy38) | -->

|           Date | Jupyter                                                                                                                                             | Colab                                                                                                                                                                               |                                                  Video                                                   |
| -------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------: |
|   May 23, 2025 | [Backpropagation and Computational Graphs](notebooks/pytorch-intro.ipynb)                                                                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/youtube/blob/main/notebooks/pytorch-intro.ipynb) | [![YouTube Video Views](https://img.shields.io/youtube/views/c7bbjqiBy38)](https://youtu.be/c7bbjqiBy38) |
|    May 6, 2025 | [PyTorch is Just numpy on GPUs](notebooks/pytorch-intro.ipynb)                                                                                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathandinu/youtube/blob/main/notebooks/pytorch-intro.ipynb) | [![YouTube Video Views](https://img.shields.io/youtube/views/c7bbjqiBy38)](https://youtu.be/c7bbjqiBy38) |
|  April 7, 2021 | [Autoscaling machine learning APIs in Python with Ray](https://github.com/jonathandinu/spark-ray-data-science/blob/main/code/lesson5.ipynb)         |                                                                                                                                                                                     | [![YouTube Video Views](https://img.shields.io/youtube/views/Xa_94PuUYQI)](https://youtu.be/Xa_94PuUYQI) |
| March 24, 2021 | [How does Ray compare to Apache Spark??](https://github.com/jonathandinu/spark-ray-data-science)                                                    |                                                                                                                                                                                     | [![YouTube Video Views](https://img.shields.io/youtube/views/yLKHHiT2nWw)](https://youtu.be/yLKHHiT2nWw) |
| March 16, 2021 | [Stateful Distributed Computing in Python with Ray Actors](https://github.com/jonathandinu/spark-ray-data-science/blob/main/code/lesson4.ipynb)     |                                                                                                                                                                                     | [![YouTube Video Views](https://img.shields.io/youtube/views/a051mbC9zqw)](https://youtu.be/a051mbC9zqw) |
| March 15, 2021 | [Remote functions in Python with Ray](https://github.com/jonathandinu/spark-ray-data-science/blob/main/code/lesson4.ipynb)                          |                                                                                                                                                                                     | [![YouTube Video Views](https://img.shields.io/youtube/views/jua2dFrHSUk)](https://youtu.be/jua2dFrHSUk) |
| March 10, 2021 | [Introduction to Distributed Computing with the Ray Framework](https://github.com/jonathandinu/spark-ray-data-science/blob/main/code/lesson4.ipynb) |                                                                                                                                                                                     | [![YouTube Video Views](https://img.shields.io/youtube/views/cEF3ok1mSo0)](https://youtu.be/cEF3ok1mSo0) |

The easiest way to get started with the code (videos or not), is to use a cloud notebook environment/platform like [Google Colab](https://colab.google/) (or Kaggle, Paperspace, etc.). For convenience I've provided links to the raw Jupyter notebooks for local development, an [NBViewer](https://nbviewer.org/) link if you would like to browse the code without cloning the repo (or you can use the built-in Github viewer), and a Colab link if you would like to interactively run the code without setting up a local development environment (and fighting with CUDA libraries).

> If you find any errors in the code or materials, please open a [Github issue](https://github.com/jonathandinu/youtube/issues) or email [errata@jonathandinu.com](mailto:errata@jonathandinu.com).

### Local Setup

```bash
git clone https://github.com/jonathandinu/youtube.git
cd youtube
```

Code implemented and tested with **Python 3.10.12** (other versions >=3.8 are likely to work fine but buyer beware...). To install all of the packages used across of the notebooks in a local [virtual environment](https://docs.python.org/3/library/venv.html):

#### Standard Library `venv`

```bash
# pyenv install 3.10.12
python --version
# => Python 3.10.12

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Using [`uv`](https://docs.astral.sh/uv/)

```bash
uv venv
uv pip install -r requirements.txt
```

> If using [`pyenv`](https://github.com/pyenv/pyenv) or [`uv`](https://docs.astral.sh/uv/concepts/python-versions/#python-version-files) to manage Python versions, they both should automatically use the version listed in `.python-version` when changing into this directory.

Additionally, the notebooks are setup with a cell to automatically select an appropriate device (GPU) based on what is available. If on a Windows or Linux machine, both NVIDIA and AMD GPUs should work (though this has only been tested with NVIDIA). And if on an Apple Silicon Mac, [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/) will be used.

```python
import torch

# default device boilerplate
device = (
    "cuda" # Device for NVIDIA or AMD GPUs
    if torch.cuda.is_available()
    else "mps" # Device for Apple Silicon (Metal Performance Shaders)
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

If no compatible device can be found, the code will default to a CPU backend. This should be fine for Lessons 1 and 2 but for any of the image generation examples (pretty much everything after lesson 2), not using a GPU will likely be uncomfortably slow—in that case I would recommend using the Google Colab links in the table above.

#### Copyright Notice

©️ 2024 Jonathan Dinu. All Rights Reserved. Removal of this copyright notice or reproduction in part or whole of the text, images, and/or code is expressly prohibited. For permission to use the content please contact copyright@jonathandinu.com.
