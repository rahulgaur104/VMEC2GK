#!/usr/bin/env python3

import os   
import time
import re
import pdb
import numpy as np
import pickle
from scipy.integrate import cumtrapz as ctrap
from scipy.interpolate import CubicSpline as cubspl
from scipy.signal import savgol_filter as sf
from netCDF4 import Dataset
#from matplotlib import pyplot as plt
from inspect import currentframe, getframeinfo
import multiprocessing as mp

parnt_dir_nam = os.path.dirname(os.getcwd())

#Dictionary to store input variables read from the text file
variables = {}
fname_in_txt = '{0}/{1}/{2}'.format(parnt_dir_nam,'input_files_vmec', 'eikcoefs_final_input.txt')

with open(fname_in_txt, 'r') as f:
    for line in f:
        try:
            name, value = line.split('=')
            name = name.replace(' ','')
            value, _ = value.split('\n')
            #value = value.strip('\n')
            if name != 'vmec_fname':
                len1 = len(re.findall('\d+.\d*', value))
            else:
                len1 = 0

            if len1 == 1:                   #float
                variables[name] = float(value)
            elif len1 > 1:                  #tuple of floats
                variables[name] = eval(value)
            else:                           #string
                try:
                    variables[name] = eval(value.replace(' ','')) # remove any spaces in the names
                except:
                    variables[name] = (value.replace(' ','')) # remove any spaces in the names
        except:
            continue



vmec_fname = variables["vmec_fname"]

want_to_surf_plot = variables["want_to_surf_plot"]


# get the type of eqbm, i.e., Miller and/or negtri/postri
eqbm_type = vmec_fname.split('_')[2]

if asym_eqbm == 1:
    print("stop using extract essence and use the full range")
    frameinfo = getframeinfo(currentframe())
    print("\n...going into debug mode line number %d"%(frameinfo.lineno))
    pdb.set_trace()



vmec_fname_path = '{0}/{1}/{2}.nc'.format(parnt_dir_nam,'input_files_vmec', vmec_fname)
rtg = Dataset(vmec_fname_path, 'r')

totl_surfs = len(rtg.variables['phi'][:].data)

#surf_idx = 100
#surf_idx = 128
surf_idx = 128
#surf_idx = 306
#surf_idx = 300
#surf_min = surf_idx - 255 # have atleast surf_idx-3 for everything to work properly
#surf_max = surf_idx + 255 # have atlest surf_idx + 3 

surf_min = surf_idx-3
surf_max = surf_idx+3


if totl_surfs < surf_max:
    print("total number of surfaces > maximum surface index") 
    surf_min = totl_surfs-6-1
    surf_max = totl_surfs-1
    surf_idx = totl_surfs-1-3
    print("\n setting surf_max to totl_surfs....\n surf_idx = %d"%(surf_idx))


# fac = 0.5*(no of poloidal points in real space)/(number of modes in Fourier space)
fac = int(2)


P_half = 4*np.pi*1E-7*rtg.variables['pres'][:].data
P_full = 4*np.pi*1E-7*rtg.variables['presf'][:].data

q_vmec_full = -1/rtg.variables['iotaf'][:].data
q_vmec_half = -1/rtg.variables['iotas'][:].data


psi_full = rtg.variables['chi'][:].data
psi_LCFS = rtg.variables['chi'][-1].data
dpsids = rtg.variables['chipf'][:].data
psi_half = psi_full + 0.5/(totl_surfs-1)*dpsids


psi_LCFS = psi_LCFS/(2*np.pi)
psi_half = psi_half/(2*np.pi)
psi_full = psi_full/(2*np.pi)

psi_full = psi_LCFS - psi_full  # shift and flip sign to ensure consistency b/w VMEC & anlyticl
psi_half = psi_LCFS - psi_half  


Phi_f = rtg.variables['phi'][:].data
Phi_LCFS = rtg.variables['phi'][-1].data

dPhids = rtg.variables['phipf'][:].data
Phi_half = Phi_f + 0.5/(totl_surfs-1)*dPhids 


P_vmec_data_4_spl = half_full_combine(P_half, P_full)
q_vmec_data_4_spl = half_full_combine(q_vmec_half, q_vmec_full)
psi_vmec_data_4_spl = half_full_combine(psi_half, psi_full)
Phi_vmec_data_4_spl = half_full_combine(Phi_f, Phi_half)
#rho_vmec_data_4_spl = np.array([np.abs(1-psi_vmec_data_4_spl[i]/psi_LCFS) for i in range(len(psi_vmec_data_4_spl))])
rho_vmec_data_4_spl = np.array([np.sqrt(np.abs(Phi_vmec_data_4_spl[i]/Phi_LCFS)) for i in range(len(psi_vmec_data_4_spl))])


P_spl = cubspl(psi_vmec_data_4_spl[::-1], P_vmec_data_4_spl[::-1])
q_spl = cubspl(psi_vmec_data_4_spl[::-1], q_vmec_data_4_spl[::-1]) 
rho_spl = cubspl(psi_vmec_data_4_spl[::-1], rho_vmec_data_4_spl[::-1])


psi = rtg.variables['chi'][surf_min:surf_max].data
psi_LCFS = rtg.variables['chi'][-1].data

dpsids = rtg.variables['chipf'][surf_min+1:surf_max+1].data
psi_half = rtg.variables['chi'][surf_min+1:surf_max+1] + 0.5/(totl_surfs-1)*dpsids

psi = psi/(2*np.pi)
psi_LCFS = psi_LCFS/(2*np.pi)

psi_half = psi_half/(2*np.pi)
#pdb.set_trace()

psi = psi_LCFS - psi  # shift and flip sign to ensure consistency b/w VMEC & anlyticl
psi_half = psi_LCFS - psi_half  # shift and flip sign to ensure consistency b/w VMEC & anlyticl

Phi_f = rtg.variables['phi'][surf_min:surf_max].data
Phi_LCFS = rtg.variables['phi'][-1].data

dPhids = rtg.variables['phipf'][surf_min:surf_max].data
Phi_half = Phi_f + 0.5/(totl_surfs-1)*dPhids 

# crucial unit conversion being performed here
# MPa to T^2 by multiplying by  \mu  = 4*np.pi*1E-7
P = 4*np.pi*1E-7*rtg.variables['pres'][surf_min+1:surf_max+1].data
q_vmec = -1/rtg.variables['iotaf'][surf_min:surf_max].data
q_vmec_half = -1/rtg.variables['iotas'][surf_min+1:surf_max+1].data

xm = rtg.variables['xm'][:].data
fixdlen = len(xm) 

theta = np.linspace(-np.pi, np.pi, fac*fixdlen+1)

xm_nyq = rtg.variables['xm_nyq'][:].data
R_mag_ax = rtg.variables['raxis_cc'][:].data


rmnc = rtg.variables['rmnc'][surf_min:surf_max].data   # Fourier coeffs of R. Full mesh quantity.
R = ifft_routine(rmnc, xm, 'e', fixdlen, fac)

rmnc_LCFS = rtg.variables['rmnc'][-1].data
R_LCFS =  ifft_routine(rmnc_LCFS, xm, 'e', fixdlen, fac)

zmns_LCFS = rtg.variables['zmns'][-1].data
Z_LCFS =  ifft_routine(zmns_LCFS, xm, 'o', fixdlen, fac)


no_of_surfs = np.shape(R)[0]

zmns = rtg.variables['zmns'][surf_min:surf_max].data  #
Z = ifft_routine(zmns, xm, 'o', fixdlen, fac)

bmnc = rtg.variables['bmnc'][surf_min+1:surf_max+1].data   # Fourier coeffs of B, Half mesh quantity, i.e. specified on the radial points in between the full-mesh points. Must be interpolated to full mesh
B = ifft_routine(bmnc, xm_nyq, 'e', fixdlen, fac)

gmnc = rtg.variables['gmnc'][surf_min+1:surf_max+1].data   # Fourier coeffs of the Jacobian
g_jac = ifft_routine(gmnc, xm_nyq, 'e', fixdlen, fac)

lmns = rtg.variables['lmns'][surf_min+1:surf_max+1].data #half mesh quantity
lmns = ifft_routine(lmns, xm, 'o', fixdlen, fac)

B_sub_zeta = rtg.variables['bsubvmnc'][surf_min+1:surf_max+1].data # half mesh quantity
B_sub_zeta = ifft_routine(B_sub_zeta, xm_nyq, 'e', fixdlen, fac)
B_sub_theta = rtg.variables['bsubumnc'][surf_min+1:surf_max+1].data # half-mesh quantity
B_sub_theta = ifft_routine(B_sub_theta, xm_nyq, 'e', fixdlen, fac)


R_mag_ax = rtg.variables['raxis_cc'][:].data.item()


idx0 = int((xm[-1]+1)*fac/2)

if (R[0][0] < R[0][idx0]):
    Z = np.abs(extract_essence(Z, idx0+1))
    R = extract_essence(R, idx0+1)
    B = extract_essence(B, idx0+1)
    B_sub_zeta = extract_essence(B_sub_zeta, idx0+1)
    B_sub_theta = extract_essence(B_sub_theta, idx0+1)
    g_jac = extract_essence(g_jac, idx0+1)
    F_half = np.array([np.mean(B_sub_zeta[i]) for i in range(no_of_surfs)])
    
    # B_poloidal from B.cdot J (grad s \times grad phi) (signs may be different)
    B_theta_vmec = np.sqrt(np.abs(B_sub_theta*Phi_LCFS/(2*np.pi*np.reshape(q_vmec_half, (-1,1))*g_jac)))
    
    #g_jac = extract_essence(g_jac, idx0+1)
    lmns = extract_essence(lmns, idx0+1)
    
    #F =  np.zeros((no_of_surfs,))
    #F = np.interp(Phi_f, Phi_half, F_half)
    F = np.reshape(F_half, (-1,1))
    u4 = []
    theta_geo = np.array([np.arctan2(Z[i], R[i]-R_mag_ax) for i in range(no_of_surfs)])
    
    
    # All surfaces before surf_min be excluded from our calculations
    fixlen_by_2 = idx0 + 1
    theta_geo_com = np.linspace(0, np.pi, idx0+1)
    theta_vmec = np.linspace(0, np.pi, idx0+1)
    #theta_st = theta_vmec + lmns
    theta_st = theta_vmec - lmns
    B_theta_vmec_2 = np.zeros((no_of_surfs, idx0+1))
else:
    Z = np.abs(extract_essence(Z, idx0+1, 1))
    R = extract_essence(R, idx0+1, 1)
    B = extract_essence(B, idx0+1, 1)
    B_sub_zeta = extract_essence(B_sub_zeta, idx0+1, 1)
    B_sub_theta = extract_essence(B_sub_theta, idx0+1, 1)
    g_jac = extract_essence(g_jac, idx0+1, 1)
    F_half = np.array([np.mean(B_sub_zeta[i]) for i in range(no_of_surfs)])
    
    # B_poloidal from B.cdot J (grad s \times grad phi) (signs may be different)
    B_theta_vmec = np.sqrt(np.abs(B_sub_theta*Phi_LCFS/(2*np.pi*np.reshape(q_vmec_half, (-1,1))*g_jac)))
    
    #g_jac = extract_essence(g_jac, idx0+1)
    lmns = extract_essence(lmns, idx0+1, 1)
    
    #F =  np.zeros((no_of_surfs,))
    #F = np.interp(Phi_f, Phi_half, F_half)
    F = np.reshape(F_half, (-1,1))
    u4 = []
    theta_geo = np.array([np.arctan2(Z[i], R[i]-R_mag_ax) for i in range(no_of_surfs)])
    
    
    # All surfaces before surf_min be excluded from our calculations
    fixlen_by_2 = idx0 + 1
    theta_geo_com = np.linspace(0, np.pi, idx0+1)
    theta_vmec = np.linspace(0, np.pi, idx0+1)
    theta_st = theta_vmec + lmns
    B_theta_vmec_2 = np.zeros((no_of_surfs, idx0+1))

#Get all the relevant quantities from a full-grid onto a half grid by interpolating in the radial direction
for i in np.arange(0, idx0+1):
    R[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), R[:, i])
    Z[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), Z[:, i])


# making sure we choose the right R for a Phi_half
rho_2 = np.array([(np.max(R[i]) - np.min(R[i]))/(np.max(R_LCFS)- np.min(R_LCFS)) for i in range(no_of_surfs)])
#rho = np.array([np.abs(1-psi_half[i]/psi_LCFS) for i in range(no_of_surfs)])
rho = np.array([np.sqrt(np.abs(Phi_half[i]/Phi_LCFS)) for i in range(no_of_surfs)])


for i in range(no_of_surfs):
    if i == 0:
    	B_theta_vmec_2[i] = np.zeros((idx0+1,))
    else:
    	B_theta_vmec_2[i] = np.sqrt(B[i]**2 - (F[i]/R[i])**2) # This B_theta is calculated using a different method. It must be equal to B_theta_vmec. It is important to note that the interpolated R must be in between the full-grid Rs.


#pdb.set_trace()
#rhoc = np.array([(np.max(R[i]) - np.min(R[i]))/(np.max(R[-1]) - np.min(R[-1])) for i in range(510)])
#plt.plot(rhoc, P, '-b', linewidth=2)
#plt.xlabel('rhoc', fontsize=20)
#plt.ylabel('P', fontsize=20)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.tight_layout()
#plt.savefig('../output_files_vmec/NOAH_jet/P_vs_rhoc_negtri.png')
#plt.show()

## For plotting and saving surfaces
#for i in np.arange(0, surf_max-surf_min, 80):
#    plt.plot(R[i], Z[i], '-b', linewidth = 1, alpha= 0.2)

##pdb.set_trace()
##plt.plot(R[30], Z[30], '-b', linewidth = 1, alpha= 0.2)

## for high-beta
##plt.plot(R[310-surf_min], Z[310-surf_min],'-r', R[128-surf_min], Z[128-surf_min], '-r', linewidth=1.8)

## for NOAH_jet postr
##plt.plot(R[13], Z[13],'-r', R[55], Z[55], '-r', R[128], Z[128], '-r', R[243], Z[243], '-r', linewidth=1.8)
## for NOAH_jet postr
#plt.plot(R[14], Z[14],'-r', R[59], Z[59], '-r', R[137], Z[137], '-r', R[257], Z[257], '-r', linewidth=1.8)

#plt.plot(R_mag_ax, 0, 'xk', ms=12, mew=3)
#plt.plot(R_LCFS[Z_LCFS>=0], Z_LCFS[Z_LCFS>=0], '-k', linewidth=2.5)
#plt.xlim([np.min(R_LCFS)-0.02, np.max(R_LCFS)+0.02])
#plt.ylim([0-0.02, np.max(Z_LCFS)+0.02])
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.tight_layout()
#plt.savefig('../output_files_vmec/NOAH_jet/negtri_jet.png', dpi=300)
#plt.show()
#pdb.set_trace()

