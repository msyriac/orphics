! `orphics`+`enlib` Tutorial

!! Working with maps

We need two objects to describe how the information on the sky is stored in a computer. The first is `shape`, the tuple specifying the dimensions of the numpy matrix that stores digitized information about the sky. Typically, this is a 2-tuple giving us (Ny,Nx) the number of pixels in the y and x directions. When working with polarization data (or galaxy shapes), a 3-tuple of the form (3,Ny,Nx) is also common.

The second object is the `wcs` or "World Coordinate System".