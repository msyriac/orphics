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

Some key modules include:

1. `orphics.cosmology` : Cosmology theory modules (mostly interfaces with CAMB), Limber approximated theory, etc.
2. `orphics.maps` : extensions to `pixell` and other map utilities for power spectra, ILC, etc.
3. `orphics.io` : plotting, config files, file i/o
4. `orphics.stats` : statistics, binning
5. `orphics.mpi` : MPI helpers
6. `orphics.catalogs` : galaxy catalogs, galaxy map-makers
7. `orphics.lensing` : CMB lensing quadratic estimator (use `symlens` for state-of-the-art), lensed pix-pix covariance, NFW kappa profiles, etc.

For example, if you wanted to calculate Limber approximation power spectra, you would import:

``
from orphics.cosmology import LimberCosmology
``
### Known Issues

- If your interactive sessions are rudely killed by an MPI error, disable MPI by exporting the environment variable `DISABLE_MPI=true` as needed.
