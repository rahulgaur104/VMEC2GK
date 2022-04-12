## VMEC2GK
VMEC equilibrium to gyrokinetics 

This script takes a VMEC equilibrium file for an axisymmetric equilibrium and creates the geometric coefficient files required for a gyrokinetics runs with GS2 or GX.

## Requirements
Script should work on Python3 >=3.6.8 
* numpy
* scipy
* matplotlib
* netCDF4
* multiprocessing


## Generating eikcoefs
Edit the relevant parameters in the file ./input\_files/eikcoefs\_final\_input.txt and run the script eikcoefs\_final.py

## Limitations
For the moment:
* the code only works with up-down-symmetric equilibria
* there is no algorithm to handle more than 2 magnetic wells. Non-trivial to add arbitrary wells. This is only a limitation if you want to use GS2.
* only checks for infinite-n ideal-ballooning stability. There is no guarantee that the equilibrium will be peeling or kink-stable.

## Relevant papers
* [Mercier C and Luc N 1974 Technical Report Commission of the European Communities Report No EUR-5127e 140 Brussels]
