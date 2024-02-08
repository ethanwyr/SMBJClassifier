# SMBJClassifier

`SMBJClassifier` is a universal classifier for Single-Molecule Break Junction (SMBJ) experiments. It takes raw current traces measured from SMBJ and automatically classifies them among several (user-specific) targeting single molecules. 

More information regarding the overall approach, methods and results can be found in our publication:

## Overview

`SMBJClassifier` includes:
- A series of preprocessing steps to remove the majority of noises within raw current traces.  
- Conversion of current traces into sampled conductance histograms, which are input data for the classifier. 
- Machine learning models for classifications, based on combining an ensemble learning method XGBoost and a convolutional neural network
- Four different input feature representations: 1D and 2D conductance probability distributions, with and without averaging over the experimental parameters

## Installation & Import

### Installation using pip

This installation method is intended for users who sets up a Python environment without `pipenv`.

```
pip install --upgrade "git+https://github.com/ethanwyr/SMBJClassifier.git"
```

*All dependencies will be installed into your selected environment with the above command. Dependencies can be found in the requirements.txt file.*

### Installation using Pipfile from source

This installation method is intended for users who sets up a Python environment with `pipenv`. `pipenv` allows users to create and activate a virtual environment with all dependencies within the Python project. For more information and installation instructions for `pipenv`, see https://pipenv.pypa.io/en/latest/.

`git clone` the `SMBJClassifier` github repository.

From the base directory of your local `SMBJClassifier` git repo, create a `Pipfile.lock` file from `Pipfile` using:

```
pipenv install requests
```

Activate the virtual environment using:

```
pipenv shell
```

Deactivate the virtual environment using:

```
exit
```

### Import
To import `SMBJClassifier` into your Python script. From the base directory of your local `SMBJClassifier` git repo, use: 

```python
import sys
sys.path.append('../SMBJClassifier')
from SMBJClassifier import DPP, model
```

## Usage

To use `SMBJClassifier`, please refer to our [demo](https://github.com/ethanwyr/SMBJClassifier/tree/main/demo/README.md) with usage examples.