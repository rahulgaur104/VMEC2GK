# VMEC2GK cookbook
VMEC equilibrium to gyrokinetics 

This script takes a VMEC equilibrium file for an axisymmetric equilibrium and creates the geometric coefficient files required for a gyrokinetics runs with GS2 or GX.

## Generating eikcoefs
Edit the relevant parameters in the file ./input\_files/eikcoefs\_final\_input.txt and run the script eikcoefs\_final.py

## Limitations
For the moment:
* the code only works with Miller(up-down symmetric) equilibria
* there is no algorithm to handle more than 2 magnetic wells. Non-trivial to add arbitrary wells.
* only checks for infinite-n ideal-ballooning mode. There is no guarantee that the equilibrium with be peeling or kink-stable.

## Relevant papers
* [Mercier C and Luc N 1974 Technical Report Commission of the European Communities Report No EUR-5127e 140 Brussels]
