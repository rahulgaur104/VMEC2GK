## Brief description of different scripts
* eikcoefs\_final.py: This is the main script. All the rest of the scripts are called from this one.
* bishoper\_save\_GX.py: This script saves a netCDF file that can be called directly by the GX code.  
* bishoper\_save\_GS2.py: This script saves a grid.out file in the directory output\_files. GS2 works with (E, \lambda) grid so you may need to tune some knobs in the res\_par variable here to get a satisfactory \lambda grid.
* bishoper\_ball.py: If you want an inf-n ideal ballooning stability analysis of a local equilibrium 
* profile\_plotter.py: If you want to plot the pressure, safety factor and the flux surface profiles, use this script independently.
* bishoper\_dict.pkl: Pickle dictionary containing all the necessary information of a local axisymmetric equilirbrium.


