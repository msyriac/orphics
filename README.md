# orphics
## A Library for Theory and Analysis for Cosmology

### Installation

To install while working on the repo (with symlinks):

``pip install -e .``

otherwise

``pip install .``

or

``python setup.py install --user``

### Usage

There are three main subpackages:

1. `orphics.theory` : Cosmology theory modules (mostly interfaces with CAMB)
2. `orphics.analysis` : data analysis helpers
3. `orphics.tools` : miscellaneous tools (e.g. plotting, statistics)

For example, if you wanted to calculate Limber approximation power spectra, you would import:

``
from orphics.theory.cosmology import LimberCosmology
``
