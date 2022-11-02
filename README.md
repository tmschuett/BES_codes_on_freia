# BES_codes_on_freia
Collection of python files to work with BES data on freia.

**imports.py** contains all kinds of potentially useful module and function imports, as well as functions for filterin, smoothing, and basic fitting functions. Usually write *from* *imports* *import* * at start of every file.

**BESClass.py** contains the BES data class with its functions to cut the data to South NBI times and calculate the fluctuation data. Also contains the functions needed to fetch the BES data from freia and calculate the channel view locations (which were mostly written by others, though partially adapted by me).

**equilibrium.py** is not mine, but it's useful to get EFIT data (specifically flux surface positions/values, which the BES channels can then be mapped to).

**functions_BES.py** contains a collection of functions useful for BES data analysis, including the cross-correlation function(s), functions to map channels to flux surfaces, functions to process the velocity results, a function to calculate the wavenumber-frequency spectrum (from I. Cziegler), and old functions to calculate coherence, phases, mode powers of signals (these can probably be improved significantly!).

**cctde_velocimetry.py** contains my CCTDE velocimetry function, which can calculate poloidal and radial velocities by taking the flux surfaces into account. I changed this a lot recently to try and find some middle ground between efficiency and readability, I tested the result but there might still be an error or two so please let me know if you find one. 
It's quite customisable, you can choose to calculate both radial and poloidal velocity or just one; remove the poloidal average before calculating the velocity; apply a bandpass filter to the BES data before; choose a cross-correlation length (in number of points) - a bit like a signal-to-noise ratio; decide whether to use a Gaussian fit to find the time delay or just pick the time of maximum correlation; decide on the data range to fit the Gaussian to (as fraction of the cross-correlation length); calculate for all channels or just some.

I have some plotting functions saved in **plotting_functions.py** and the two velocity-specific files, I can write some example use cases for those too.

**BES_viewradius.csv** stores the BES view radius (of the centre of the array) for a number of MAST shots.

**example_use_velocimetry.py** is an example of how to use these functions to calculate the velocities for a MAST shot. This should work if you run it on freia (and have the other files in this project).
