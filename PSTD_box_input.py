from numpy import *
from scipy import *

# Extended PSTD method to solve sound propagation
#
# 2D propagation in a box
#   o box boundaries are rigid or have a finite impedance
#   o box boundaries are located at velocity nodes
#   o box boundaries are locally reacting
#
#Created by Maarten Hornikx, 2023-07, based on a Python script prepared for openPSTD in 2012


#--------------------------------------------------------------------------
# INPUT SECTION
#--------------------------------------------------------------------------
# variable input


freqmax = 1000.             # maximum 1/3 octave band of interest in Hz
print(freqmax)
xdist = 20.             # horizontal dimension of box (integer number of meters)
zdist = 10.             # vertical dimension of box (integer number of meters)
spos=array([-2.,2.])       # horizontal and vertical position of source from center of origin (center of the domain)
rpos=array([[0., 0.],[-1., -1.]])# horizontal and vertical positions of receiver from center of origin (center of the domain)

# normal incident absorption coefficient for left, right, lower and upper boundaries
# The boundary is treated as rigid alfa < 0.005
alfaleft = 0.001
alfaright = 0.001
alfalower = 1
alfaupper = 1
print(freqmax)
calctime = 0.2        # calculation time in s

PMLcells = 20       # number of cells of the damping layer at non-reflective boundary conditions
# should be at least 20 (50 cells is preferable)