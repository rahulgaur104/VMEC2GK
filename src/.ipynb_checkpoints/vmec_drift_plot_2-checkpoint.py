#!/usr/bin/env python3
"""
Created on Fri Apr 10 10:54:48 2020

@author: ralap

The purpose of this code is to take VMEC data and calculate eikcoefs for GS2. Before doing any calculations,we interpolate(Probably incorrectly) R and Z from full grid to the adjacent half grid. All the calculations after that are done on the half grid.

B_poloidal and B_toroidal are used from the VMEC output file. 

Sanity checks:
B_poloidal_VMEC = np.sqrt(B_vmec**2 - B_toroidal_VMEC**2)

B_poloidal_VMEC = -1/R d\psi/drho_ML

shat_VMEC = shat_bishop 

dBdr bishop = dB_vmecdr

dtdr_st = \partial theta_st/ \partial rho_ML should be zero atleast at theta = -pi, pi

The only difference from vmec_drift_plot_4.py is that we are using the data from the collocation points.

(11/6/2020)
Ballooning stability calculation functions will be added soon.

"""
import os   
from datetime import datetime
import time
import re
import pdb
import numpy as np
import pickle
from autograd import  elementwise_grad
import quadpy as qp
from scipy.integrate import quad, solve_ivp, quadrature
from scipy.integrate import cumtrapz as ctrap
from scipy.interpolate import Akima1DInterpolator as akint
from scipy.interpolate import InterpolatedUnivariateSpline as linspl
#from scipy.interpolate import interp1d  as lint
from scipy.interpolate import CubicSpline as cubspl
from scipy.interpolate import CubicHermiteSpline  as chermspl
from scipy.interpolate import BPoly
from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter as sf
from scipy.optimize import fsolve
from scipy.optimize import toms748 as toms
from matplotlib import pyplot as plt
from netCDF4 import Dataset

import multiprocessing as mp
import mpmath
import math
# custom cython lib to find the inverse of a function
from pynverse import inversefunc as invfun

# custom cython lib to find intersection of curves
# the only method we are going from this library to use is called intersection
from intersection_lib  import intersection as intsec 

parnt_dir_nam = os.path.dirname(os.getcwd())

#Dictionary to store input variabled read from the text file
variables = {}
fname_in_txt = '{0}/{1}/{2}'.format(parnt_dir_nam,'input_files_vmec', 'vmec_verify_input.txt')

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
                variables[name] = eval(value.replace(' ','')) # remove any spaces in the names
        except:
            continue


Rmin = variables["Rmin"]
Rmax = variables["Rmax"]
Rhatmax = variables["Rhatmax"]
mu_fac_wo_pi = variables["mu_fac_wo_pi"]
rho_mid = variables["rho_mid"]  
rho_width = variables["rho_width"]
no_of_surfs = int(variables["no_of_surfs"])

vmec_fname = variables["vmec_fname"]


R_res =  variables["R_res"]         # spacing between the R points for Z(Rhat, R), must be greater 
                                    # than or equal to(and a factor of) 5E-3
#R_mag_ax = variables["R_mag_ax"]    # center of the smallest flux surface for Rhatmax 

pp  = variables["pp"]               # pressure parameters --- add no more than 3 digits after
                                    # the decimal
pp_type =  variables["pp_type"]     # polynomial type pressure profile; change to 'other' for 
                                    # non-polynomial
       

bp  = variables["bp"]               # boundary parameters ---  "  "   "  
bp_type = variables["bp_type"] 

psp = variables["psp"]              # dpsidR parameters   ---  "  "   " 
psp_type =  variables["psp_type"]   # change to 'other' for non-polynomial profiles


C = variables["C"]                  # Constant that determines the poloidal current F

fd = variables["fd_scheme"]         # Finite difference scheme

#pdb.set_trace()
Rdiff = Rmax-Rmin
mu_fac =np.pi*mu_fac_wo_pi   # because pressure is in MPa

# check
assert (rho_mid > rho_width and rho_mid <= 1-rho_width)\
       ,"can't have the width of the annulus > its value in the middle or < 1 - value in the middle" 

# creating an array of normalized flux surface values
rho_ann = np.linspace(rho_mid-rho_width, rho_mid+rho_width, no_of_surfs)


def extract_essence(arr, extract_len):
    brr = np.zeros((np.shape(arr)[0], extract_len))
    for i in range(np.shape(arr)[0]):
    	brr[i] = arr[i][0:extract_len][::-1]

    return brr


def ifft_routine(arr, xm, char, fixdlen, fac):
    
    if np.shape(np.shape(arr))[0] > 1:
    	colms =  np.shape(arr)[0]
    	#fac = 8
    	rows =  np.shape(arr)[1]
    	N = fac*rows + 1 #> 2*(rows)-1
    	arr_ifftd = np.zeros((colms, N))
    	theta = np.linspace(-np.pi, np.pi, N)
    	if char == 'e': #even array 
    		for i in range(colms):
    			for j in range(N):
    				for k in range(rows):
    					angle = xm[k]*theta[j]
    					arr_ifftd[i, j] = arr_ifftd[i, j] + np.cos(angle)*arr[i][k] 
    	else:           #odd array
    		for i in range(colms):
    			for j in range(N):
    				for k in range(rows):
    					angle = xm[k]*theta[j]
    					arr_ifftd[i, j] = arr_ifftd[i, j] + np.sin(angle)*arr[i][k] 

    	arr_final = np.zeros((colms, fac*fixdlen))
    	if N > fac*fixdlen+1 : # if longer arrays, interpolate
    		theta_n = np.linspace(-np.pi, np.pi, fac*fixdlen)
    		for i in range(colms):
    			arr_final[i] = np.interp(theta_n, theta, arr_ifftd[i])
    	else:
    		arr_final = arr_ifftd

    else:
    	rows =  len(arr)
    	#fac = 8
    	N = fac*rows + 1 
    	arr_ifftd = np.zeros((N,))
    	theta = np.linspace(-np.pi, np.pi, N)
    	if char == 'e': #even array 
    		for j in range(N):
    			for k in range(rows):
    				angle = xm[k]*theta[j]
    				arr_ifftd[j] = arr_ifftd[j] + np.cos(angle)*arr[k] 
    	else:           #odd array
    		for j in range(N):
    			for k in range(rows):
    				angle = xm[k]*theta[j]
    				arr_ifftd[j] = arr_ifftd[j] + np.sin(angle)*arr[k] 

    	arr_final = np.zeros((fac*fixdlen, ))
    	if N > fac*fixdlen+1 :
    		theta_n = np.linspace(-np.pi, np.pi, fixdlen)
    		arr_final = np.interp(theta_n, theta, arr_ifftd)
    	else:
    		arr_final = arr_ifftd
      

    return arr_final


def derm(arr, ch):
    # Finite difference subroutine
    temp = np.shape(arr)
    if len(temp) == 1 and ch == 'l':
    	#pdb.set_trace()
    	d1, d2 = np.shape(arr)[0], 1
    	arr = np.reshape(arr, (d2,d1))
    	diff_arr = np.zeros((d2,d1))
    	diff_arr[0, 0] = (arr[0, 1] - arr[0, 0]) 
    	diff_arr[0, -1] = (arr[0, -1] - arr[0, -2]) 
    	diff_arr[0, 1:-1] = np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)  

    elif len(temp) == 1 and ch == 'r':
    	#pdb.set_trace()
    	d1, d2 = np.shape(arr)[0], 1
    	diff_arr = np.zeros((d1,d2))
    	arr = np.reshape(arr, (d1,d2))
    	diff_arr[0, 0] = (arr[1, 0] - arr[0, 0]) 
    	diff_arr[-1, 0] = (arr[-1, 0] - arr[-2, 0]) 
    	diff_arr[1:-1, 0] = np.diff(arr[:-1,0], axis=0) + np.diff(arr[1:,0], axis=0)  


    else:
    	d1, d2 = np.shape(arr)[0], np.shape(arr)[1]

    	diff_arr = np.zeros((d1,d2))
    	if ch == 'r':
    		#pdb.set_trace()
    		diff_arr[0, :] = (arr[1,:] - arr[0,:]) 
    		diff_arr[-1, :] = (arr[-1,:] - arr[-2,:]) 
    		diff_arr[1:-1, :] = (np.diff(arr[:-1,:], axis=0) + np.diff(arr[1:,:], axis=0))  

    	elif ch == 'l':
    		#pdb.set_trace()
    		diff_arr[:, 0] = (arr[:, 1] - arr[:, 0]) 
    		diff_arr[:, -1] = (arr[:, -1] - arr[:, -2]) 
    		diff_arr[:, 1:-1] = (np.diff(arr[:,:-1], axis=1) + np.diff(arr[:,1:], axis=1))  

    	elif ch == 'p':
    		diff_arr = np.zeros((d1,d2))
    		#pdb.set_trace()
    		arr = np.reshape(arr, (d1,d2))
    		if no_of_surfs == 3:
    			diff_arr[1, :] =  arr[-1,:] + arr[0,:]
    			diff_arr[-1, :] =  arr[-1, :]
    			diff_arr[0, :] =  arr[0, :]
    		else:
    			diff_arr[1:-1, :] =  (arr[:-1,:] + arr[1:,:])[:-1, :]
    			diff_arr[-1, :] =  arr[-1, :]
    			diff_arr[0, :] =  arr[1, :]

    arr = np.reshape(diff_arr, temp)
    return diff_arr

def half_full_combine(arrh, arrf):
#Function to combine data fronm both the half and the full radial meshes

    len0 = len(arrh)
    arr = np.zeros((2*len0-1,))
    arr[0] = arrf[0]
    
    for i in np.arange(1,len0):
        arr[2*i-2] = arrf[i-1]
        arr[2*i-1] = arrh[i]
    
    arr[2*len0-2] = arrf[len0-1]
    
    return arr
         


vmec_fname_path = '{0}/{1}/{2}.nc'.format(parnt_dir_nam,'input_files_vmec', vmec_fname)
rtg = Dataset(vmec_fname_path, 'r')

totl_surfs = len(rtg.variables['phi'][:].data)

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

P_vmec_data_4_spl = half_full_combine(P_half, P_full)
q_vmec_data_4_spl = half_full_combine(q_vmec_half, q_vmec_full)
psi_vmec_data_4_spl = half_full_combine(psi_half, psi_full)
rho_vmec_data_4_spl = np.array([np.abs(1-psi_vmec_data_4_spl[i]/psi_LCFS) for i in range(len(psi_vmec_data_4_spl))])



P_spl = cubspl(psi_vmec_data_4_spl[::-1], P_vmec_data_4_spl[::-1])
q_spl = cubspl(psi_vmec_data_4_spl[::-1], q_vmec_data_4_spl[::-1]) 
rho_spl = cubspl(psi_vmec_data_4_spl[::-1], rho_vmec_data_4_spl[::-1])


surf_idx = 280
surf_min = surf_idx-5
surf_max = surf_idx+5

psi = rtg.variables['chi'][surf_min:surf_max].data
psi_LCFS = rtg.variables['chi'][-1].data

dpsids = rtg.variables['chipf'][surf_min:surf_max].data
psi_half = psi + 0.5/(totl_surfs-1)*dpsids

psi = psi/(2*np.pi)
psi_LCFS = psi_LCFS/(2*np.pi)

psi_half = psi_half/(2*np.pi)



psi = psi_LCFS - psi  # shift and flip sign to ensure consistency b/w VMEC & anlyticl
psi_half = psi_LCFS - psi_half  # shift and flip sign to ensure consistency b/w VMEC & anlyticl


Phi_f = rtg.variables['phi'][surf_min:surf_max].data
Phi_LCFS = rtg.variables['phi'][-1].data

dPhids = rtg.variables['phipf'][surf_min:surf_max].data
Phi_half = Phi_f + 0.5/(totl_surfs-1)*dPhids 

# crucial unit conversion being performed here
# MPa to T^2 by multiplying by  \mu  = 4*np.pi*1E-7
P = 4*np.pi*1E-7*rtg.variables['pres'][surf_min:surf_max].data
q_vmec = -1/rtg.variables['iotaf'][surf_min:surf_max].data
q_vmec_half = -1/rtg.variables['iotas'][surf_min:surf_max].data

xm = rtg.variables['xm'][:].data
fixdlen = len(xm) 

fac = int(12)
theta = np.linspace(-np.pi, np.pi, fac*fixdlen+1)

xm_nyq = rtg.variables['xm_nyq'][:].data
R_mag_ax = rtg.variables['raxis_cc'][:].data


rmnc = rtg.variables['rmnc'][surf_min:surf_max].data   # Fourier coeffs of R. Full mesh quantity.
R = ifft_routine(rmnc, xm, 'e', fixdlen, fac)

rmnc_LCFS = rtg.variables['rmnc'][-1].data
R_LCFS =  ifft_routine(rmnc_LCFS, xm, 'e', fixdlen, fac)

no_of_surfs = np.shape(R)[0]

zmns = rtg.variables['zmns'][surf_min:surf_max].data  #
Z = ifft_routine(zmns, xm, 'o', fixdlen, fac)

bmnc = rtg.variables['bmnc'][surf_min:surf_max].data   # Fourier coeffs of B, Half mesh quantity, i.e. specified on the radial points in between the full-mesh points. Must be interpolated to full mesh
B = ifft_routine(bmnc, xm_nyq, 'e', fixdlen, fac)

gmnc = rtg.variables['gmnc'][surf_min:surf_max].data   # Fourier coeffs of the Jacobian
g_jac = ifft_routine(gmnc, xm_nyq, 'e', fixdlen, fac)

lmns = rtg.variables['lmns'][surf_min:surf_max].data #half mesh quantity
lmns = ifft_routine(lmns, xm, 'o', fixdlen, fac)

B_sub_zeta = rtg.variables['bsubvmnc'][surf_min:surf_max].data # half mesh quantity
B_sub_zeta = ifft_routine(B_sub_zeta, xm_nyq, 'e', fixdlen, fac)
B_sub_theta = rtg.variables['bsubumnc'][surf_min:surf_max].data # half-mesh quantity
B_sub_theta = ifft_routine(B_sub_theta, xm_nyq, 'e', fixdlen, fac)


R_mag_ax = rtg.variables['raxis_cc'][:].data.item()



idx0 = int((xm[-1]+1)*fac/2)

Z = np.abs(extract_essence(Z, idx0+1))
R = extract_essence(R, idx0+1)
B = extract_essence(B, idx0+1)
B_sub_zeta = extract_essence(B_sub_zeta, idx0+1)
B_sub_theta = extract_essence(B_sub_theta, idx0+1)
g_jac = extract_essence(g_jac, idx0+1)
F_half = np.array([np.mean(B_sub_zeta[i]) for i in range(no_of_surfs)])


# B_poloidal from B.cdot J (grad s \times grad phi) (signs may be different)
B_theta_vmec = np.sqrt(np.abs(B_sub_theta*Phi_LCFS/(2*np.pi*np.reshape(q_vmec_half, (-1,1))*g_jac)))

#B_theta_vmec = extract_essence(B_theta_vmec, idx0+1)
#F = np.reshape(F,(-1,1))
#g_jac = extract_essence(g_jac, idx0+1)
lmns = extract_essence(lmns, idx0+1)

#theta_geo = extract_essence(theta_geo, idx0+1)
#F =  np.zeros((no_of_surfs,))
#F = np.interp(Phi_f, Phi_half, F_half)
F = np.reshape(F_half, (-1,1))
u4 = []
theta_geo = np.array([np.arctan2(Z[i], R[i]-R_mag_ax) for i in range(no_of_surfs)])


# All surfaces before surf_min be excluded from our calculations
fixlen_by_2 = idx0 + 1
theta_geo_com = np.linspace(0, np.pi, idx0+1)
theta_vmec = np.linspace(0, np.pi, idx0+1)
theta_st = theta_vmec - lmns
B_theta_vmec_2 = np.zeros((no_of_surfs, idx0+1))

#Get all the relevant quantities from a full-grid onto a half grid by interpolating in the radial direction
for i in np.arange(0, idx0+1):
    R[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), R[:, i])
    Z[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), Z[:, i])



# making sure we choose the right R for a Phi_half
rho_2 = np.array([(np.max(R[i-1]) - np.min(R[i-1]))/(np.max(R_LCFS)- np.min(R_LCFS)) for i in range(no_of_surfs)])
rho = np.array([np.abs(1-psi_half[i]/psi_LCFS) for i in range(no_of_surfs)])


for i in range(no_of_surfs):
    if i == 0:
    	B_theta_vmec_2[i] = np.zeros((idx0+1,))
    else:
    	B_theta_vmec_2[i] = np.sqrt(B[i]**2 - (F[i]/R[i-1])**2) # This B_theta is calculated using a different method. It must be equal to B_theta_vmec. It is important to note that the interpolated R must be in between the full-grid Rs.


#pdb.set_trace()

psi = psi_half
##P_spl = cubspl(psi[::-1], P[::-1])
##q_spl = cubspl(psi[::-1], q_vmec_half[::-1]) 
##rho_spl = cubspl(psi[::-1], rho[::-1])
F_spl = cubspl(psi[::-1], F[::-1])

# various derivatives
dRi = np.zeros((no_of_surfs, fixlen_by_2)) # ML distances
dZi = np.zeros((no_of_surfs, fixlen_by_2))
dRj = np.zeros((no_of_surfs, fixlen_by_2))
dZj = np.zeros((no_of_surfs, fixlen_by_2))

dRh = np.zeros((no_of_surfs, fixlen_by_2)) # cartesian distances
dZh = np.zeros((no_of_surfs, fixlen_by_2))
dRv = np.zeros((no_of_surfs, fixlen_by_2))
dZv = np.zeros((no_of_surfs, fixlen_by_2))

drdR = np.zeros((no_of_surfs, fixlen_by_2))
drdZ = np.zeros((no_of_surfs, fixlen_by_2))
dtdZ = np.zeros((no_of_surfs, fixlen_by_2))
dtdR = np.zeros((no_of_surfs, fixlen_by_2))
dti = np.zeros((no_of_surfs, fixlen_by_2))
dtdr = np.zeros((no_of_surfs, fixlen_by_2))
dtdr_st = np.zeros((no_of_surfs, fixlen_by_2))
dB_cart = np.zeros((no_of_surfs, fixlen_by_2))
dB2l = np.zeros((no_of_surfs, fixlen_by_2))
dBdr =  np.zeros((no_of_surfs, fixlen_by_2))
dBdt =  np.zeros((no_of_surfs, fixlen_by_2))
dPdr =  np.zeros((no_of_surfs, fixlen_by_2))
dqdr =  np.zeros((no_of_surfs, fixlen_by_2))
B20 =  np.zeros((no_of_surfs, fixlen_by_2))
B0 =  np.zeros((no_of_surfs, fixlen_by_2))
B_p_t =  np.zeros((no_of_surfs, fixlen_by_2))
B_p_t_cart =  np.zeros((no_of_surfs, fixlen_by_2))
B_p =  np.zeros((no_of_surfs, fixlen_by_2))
dpsidr =  np.zeros((no_of_surfs, fixlen_by_2))
cvdrift = np.zeros((no_of_surfs, fixlen_by_2))
gbdrift = np.zeros((no_of_surfs, fixlen_by_2))
gbdrift0 = np.zeros((no_of_surfs, fixlen_by_2))
cvdrift0 = np.zeros((no_of_surfs, fixlen_by_2))
phi = np.zeros((fixlen_by_2,))
phi_n = np.zeros((no_of_surfs, fixlen_by_2))
dthdl = np.zeros((no_of_surfs, fixlen_by_2))
dt = np.zeros((no_of_surfs, fixlen_by_2))
dr = np.zeros((no_of_surfs, fixlen_by_2))
u_ML  = np.zeros((no_of_surfs, fixlen_by_2))
u_ML_diff  = np.zeros((no_of_surfs, fixlen_by_2))
curv = np.zeros((no_of_surfs, fixlen_by_2))   
nu_til = np.zeros((no_of_surfs, fixlen_by_2))
theta_u = np.zeros((no_of_surfs, fixlen_by_2))
theta_d = np.zeros((no_of_surfs, fixlen_by_2))
theta_h_u = np.zeros((no_of_surfs, fixlen_by_2))
theta_h_d = np.zeros((no_of_surfs, fixlen_by_2))
theta_v_u = np.zeros((no_of_surfs, fixlen_by_2))
theta_v_d = np.zeros((no_of_surfs, fixlen_by_2))
dR_dpsi = np.zeros((no_of_surfs, fixlen_by_2))
dR_dt = np.zeros((no_of_surfs, fixlen_by_2))
dZ_dpsi = np.zeros((no_of_surfs, fixlen_by_2))
dZ_dt = np.zeros((no_of_surfs, fixlen_by_2))
dpsidR = np.zeros((no_of_surfs, fixlen_by_2))
dpsidZ = np.zeros((no_of_surfs, fixlen_by_2))
dt_dR =  np.zeros((no_of_surfs, fixlen_by_2))
dt_dZ =  np.zeros((no_of_surfs, fixlen_by_2))
jac = np.zeros((no_of_surfs, fixlen_by_2))
jac_m = np.zeros((no_of_surfs, fixlen_by_2))


m= np.zeros((no_of_surfs, fixlen_by_2))
dBr = np.zeros((no_of_surfs, fixlen_by_2))
u5 = []

tic3 = time.perf_counter() 

#VMEC B_p here
#B_p = np.sqrt(np.abs(B_theta_vmec/g_jac))/np.reshape(q_vmec, (-1,1))
B_p =  B_theta_vmec


#theta_st_com = np.linspace(0, np.pi, idx0+1)
psi_diff = derm(psi, 'r')
dt_vmec_l = derm(theta_vmec, 'l')

dR_dpsi = derm(R, 'r')/psi_diff
dR_dt = derm(R, 'l')/dt_vmec_l
dZ_dpsi = derm(Z, 'r')/psi_diff
dZ_dt = derm(Z, 'l')/dt_vmec_l

jac = dR_dpsi*dZ_dt - dZ_dpsi*dR_dt

dpsidR = dZ_dt/jac
dpsidZ = -dR_dt/jac
dt_dR = -dZ_dpsi/jac
dt_dZ =  dR_dpsi/jac


for i in range(no_of_surfs):
	 dRj[i, :] = derm(R[i,:], 'l')
	 dZj[i, :] = derm(Z[i,:], 'l') 
	 phi = np.arctan2(dZj[i,:], dRj[i,:])
	 phi = np.concatenate((phi[phi>=0]-np.pi/2, phi[phi<0]+3*np.pi/2)) 
	 phi_n[i,:] = phi

u_ML = np.arctan2(-derm(Z, 'l'), derm(R, 'l'))


dl = np.sqrt(derm(R, 'l')**2 + derm(Z, 'l')**2)
# du_ML/dl is negative and dphi = -du_ML so R_c = -du_ML/dl > 0
R_c = dl/(2*np.concatenate((np.diff(phi_n, axis=1), np.reshape(np.diff(phi_n)[:, -1],(-1,1))), axis=1))
B2 = B**2

#plt.plot(theta_geo_m[10], theta_st_com_m, theta_geo_com, theta_st[10]); plt.show() 

B0 = [np.sqrt(B_p_t_cart[i]**2 + (F[i]/R[i,:])**2) for i in np.arange(surf_min, no_of_surfs)]

q_int = np.zeros((no_of_surfs,))
L = np.cumsum(dl, axis =1)/2

#plt.plot(q_vmec[:-1], q_int[:-1], q_vmec[:-1], q_vmec[:-1]); plt.show()
#q_int = np.abs(q_vmec)




####################----------------EIKCOEFS CALCULATION ---------------------------##################


#pdb.set_trace()
#B_p = F/(R**2*np.reshape(q_int,(-1,1))*dt_st_l/dl)
#B_p = np.sqrt(B**2 - (F/R)**2)

rel_surf_idx = surf_idx - surf_min -1 # -1 in the end becuase of python's 0 based indexing 

theta_st_com = theta_st[rel_surf_idx]

shat = rho[rel_surf_idx]/q_vmec_half[rel_surf_idx]*q_spl.derivative()(psi[rel_surf_idx])/rho_spl.derivative()(psi[rel_surf_idx])


# the factor of 2 in the front cancels with the the 2 in 2*pi in the expression for shat_test
a_s = -(2*q_vmec_half[rel_surf_idx]/F[rel_surf_idx]*theta_st_com + 2*F[rel_surf_idx]*q_vmec_half[rel_surf_idx]*ctrap(1/(R[rel_surf_idx]**2*B_p[rel_surf_idx]**2), theta_st_com, initial=0))  
b_s = -(2*q_vmec_half[rel_surf_idx]*ctrap(1/(B_p[rel_surf_idx]**2), theta_st_com, initial=0 ))
c_s =  -(2*q_vmec_half[rel_surf_idx]*ctrap((2*np.sin(u_ML[rel_surf_idx])/R[rel_surf_idx] +  2/R_c[rel_surf_idx])*1/(R[rel_surf_idx]*B_p[rel_surf_idx]), theta_st_com, initial=0))
dl = np.sqrt(derm(R, 'l')**2 + derm(Z, 'l')**2)

dFdpsi = F_spl.derivative()(psi[rel_surf_idx])
dPdpsi = P_spl.derivative()(psi[rel_surf_idx])
dqdpsi = q_spl.derivative()(psi[rel_surf_idx])

dpsidrho = 1/rho_spl.derivative()(psi[rel_surf_idx])
##B_N =  1
##a_N = 1
B_N = np.max(B[rel_surf_idx])
a_N = np.sqrt(psi_LCFS/B_N)

shat_test = -(1/(2*np.pi*q_vmec_half[rel_surf_idx]))*(rho[rel_surf_idx]*dpsidrho)*(a_s[-1]*dFdpsi +b_s[-1]*dPdpsi - c_s[-1])
##shat_test_2 = -(1/(2*np.pi*q_vmec_half[rel_surf_idx]))*(rho[rel_surf_idx]*dpsidrho)*(a_s[-1]*dFdpsi +b_s[-1]*dPdpsi + c_s[-1])

dFdpsi_4_ball = (-shat*2*np.pi*q_vmec_half[rel_surf_idx]/(rho[rel_surf_idx]*dpsidrho) - b_s[-1]*dPdpsi + c_s[-1])/a_s[-1]
##dFdpsi_4_ball_2 = (shat*2*np.pi*q_vmec_half[rel_surf_idx]/(rho[rel_surf_idx]*dpsidrho) - b_s[-1]*dPdpsi - c_s[-1])/a_s[-1]

# The 0.5 in the front is to cancel the factor of 2 in that has been used in a_s, b_s and c_s
##aprime_bish = R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi_4_ball +b_s*dPdpsi - c_s)*0.5
aprime_bish_1 = R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi_4_ball +b_s*dPdpsi - c_s)*0.5
##aprime_bish_3 = R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi_4_ball_2 +b_s*dPdpsi + c_s)*0.5


aprime_bish = R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi +b_s*dPdpsi - c_s)*0.5



dpsi_dr = -np.sqrt(dpsidR**2 + dpsidZ**2)
dtdr_vmec = -(dt_dR*dpsidR + dt_dZ*dpsidZ)/np.sqrt(dpsidR**2 + dpsidZ**2)
dtdr_st = dtdr_vmec - (derm(lmns, 'r')/psi_diff*dpsi_dr + derm(lmns, 'l')/dt_vmec_l*dtdr_vmec)


dpsi_dr = -R*B_p

dqdr = dqdpsi*dpsi_dr
dPdr = dPdpsi*dpsi_dr 

aprime = -(np.reshape(q_vmec_half, (-1,1))*dtdr_st + dqdr*theta_st_com)


plt.plot(theta_st_com, aprime[rel_surf_idx], theta_st_com, -aprime_bish_1, theta_st_com, -aprime_bish); plt.legend(['aprime_fd', 'aprime_bish_corr', 'aprime_bish'])
plt.show()

pdb.set_trace()

##plt.plot(theta_st_com, -aprime_bish, theta_st_com, -aprime_bish_1, theta_st_com, -aprime_bish_2,theta_st_com, -aprime_bish_3, theta_st_com, aprime[rel_surf_idx]); plt.legend(['1','2','3','4']); plt.show()

##plt.plot(theta_st_com, -aprime_bish, theta_st_com, aprime[rel_surf_idx], theta_st_com, -aprime_bish_1)
##plt.legend(['bish','fd', 'adj']); plt.show()

dBdr_bish = B_p/B*(-B_p/R_c + dPdpsi*R + F**2*np.sin(u_ML)/(R**3*B_p))

dtdr_st_bish = (aprime_bish/dpsidrho - dqdr*theta_st_com)/np.reshape(q_int, (-1,1))

gds2 =  (dpsidrho)**2*(1/R[rel_surf_idx]**2 + (dqdr*theta_st_com)**2 + (np.reshape(q_vmec_half,(-1,1)))**2*(dtdr_st**2 + (dt_st_l/dl)**2)+ 2*np.reshape(q_vmec_half,(-1,1))*dqdr*theta_st_com*dtdr_st)*1/(a_N*B_N)**2

gds21 = dpsidrho*dqdpsi*dpsidrho*(dpsi_dr*aprime)/(a_N*B_N)**2

gds22 = (dqdpsi*dpsidrho)**2*np.abs(dpsi_dr)**2/(a_N*B_N)**2
#dtdr_st = dt_st_i/dr     
#dtdr_st = (dt_dR*dpsidR + dt_dZ*dpsidZ)/np.sqrt(dpsidR**2 + dpsidZ**2)

#diffq = np.reshape(np.append(np.diff(q_int), np.diff(q_int)[-1]),(-1,1))
if fd=='one-sided':
    diffq = np.reshape(np.append(np.diff(q_int), np.diff(q_int)[-1]),(-1,1))
else:    
    diffq = derm(q_int, 'r')



if fd == 'one-sided':
	dB2l = np.concatenate((np.diff(B2, axis=1), np.reshape(np.diff(B2, axis=1)[:,-1],(-1,1))), axis=1)
	dBl = np.concatenate((np.diff(B, axis=1), np.reshape(np.diff(B, axis=1)[:,-1],(-1,1))), axis=1)
	#dB2r = np.concatenate((np.diff(B2, axis=0), np.reshape(np.diff(B2, axis=0)[-1,:],(1,-1))), axis=0)
	dB_cart = np.concatenate((np.diff(B, axis=0), np.reshape(np.diff(B, axis=0)[-1,:],(1,-1))), axis=0)

	diffP = 2*np.diff(P)
	diffP = np.append(diffP, diffP[-1])
	diffP = np.reshape(diffP, (-1,1))

	diffq = np.diff(q_int)
	diffq = np.append(diffq, diffq[-1])
	diffq = np.reshape(diffq, (-1,1))

	diffrho = np.diff(rho)
	diffrho = np.append(diffrho, diffrho[-1])
	# B2 decreases with theta for a given r
	# theta decreases with l
else:
	dB2l = derm(B2, 'l')
	#dB2_cart = derm(B2, 'r')
	dBl = derm(B, 'l')
	dB_cart = derm(B, 'r')
	diffF = derm(np.reshape(F, (-1,)), 'r')
	diffP = 2*derm(P, 'r') # factor of two from 2 mu p/B_a**2
	diffq = derm(q_int, 'r')
	diffrho = derm(rho, 'r')

##B_p_diff = np.zeros((no_of_surfs, fixlen_by_2))

##dpsi_dr = np.sign(psi_diff)*np.sqrt(dpsidR**2 + dpsidZ**2)

##dB_dr =  np.zeros((no_of_surfs, fixlen_by_2))
    #theta_st[i,:] = np.linspace(0, np.pi, idx0+1) + lmns[i]
    #theta_geo[i] = np.arctan2(Z[i], R[i]-R_mag_ax)
    #dBr[j] = B_p_t[j]**2/B[j]*(curv[j] - diffP[j]/(2*psi_diff[j])*R[j]/B_p_t[j] + F[j]*diffF[j]/(psi_diff[j]*dpsidr[j])) + F[j]**2/(B[j]*R[j]**3)*(-np.sin(u_ML[j]) + R[j]*dpsi_dr[j]*diffF[j]/(psi_diff[j]*F[j]))


dB_dr = derm(B, 'l')/dt_st_l*dtdr_st + derm(B, 'r')/psi_diff*dpsi_dr

gbdrift_bish = dpsidrho/(B**3)*(2*B**2*dBdr_bish/dpsi_dr + aprime_bish*(1/dpsidrho)*F/R*dB2l/dl*1/B)

#gbdrift = dpsidrho/(B**3)*sf(-2*B**2*dBdr_bish/dpsi_dr + aprime*F/R*dBl/dl, 37,1, mode='mirror')
gbdrift = dpsidrho*(-2/B*dBdr_bish/dpsi_dr + 2*aprime*F/R*1/B**3*dBl/dl)

##gbdrift = sf(dpsidrho*(-2/B*dBdr_bish/dpsi_dr + 2*aprime*F/R*1/B**3*dBl/dl), 31,1)

#cvdrift =  dpsidrho/np.abs(B**3)*(-2*B2*(2*dPdr/(2*B) + dBdr_bish)/dpsi_dr + aprime*F/R*dB2l/dl*1/B)
cvdrift =  dpsidrho/np.abs(B)*(-2*(2*dPdr/(2*B))/dpsi_dr) + gbdrift

# Remove the -1 in from of gbdrift0. Only there to match signs with FIGG output
gbdrift0 =  1*2/(B**3)*dpsidrho*np.reshape(F, (-1,1))/R*(dqdr*dBl/dl)
##gbdrift0 = sf(gbdrift0, 39, 1, mode='mirror')

gradpar = a_N/(R*B)*(-dpsi_dr)*(derm(theta_st_com, 'l')/dl) # gradpar is b.grad(theta)



##plt.plot(theta_st_com, dB_dr[5] - dBdr_bish[5]); plt.show() 


######################------BALLOONING CALCULATION IN FLUX COORDINATES----------------##############

def reflect_n_append(arr, ch):
	"""
	The purpose of this function is to increase the span of an array from [0, np.pi] to [-np.pi,np.pi). ch can either be 'e' or 'o' depending upon the parity of the input array.
	"""
	rows = 1
	brr = np.zeros((2*len(arr)-1, ))
	if ch == 'e':
		for i in range(rows):
			brr = np.concatenate((arr[::-1][:-1], arr))
	else :
		for i in range(rows):
			brr = np.concatenate((-arr[::-1][:-1], arr))
	return brr





 
##data = np.zeros((ntheta+1, 10))
##for i in range(ntheta+1):
##	if i == 0:
##		# drhodpsi = -1.0
##		data[0, :5] = np.array([int((ntheta-1)/2), 0., shat, -1.0 , q_vmec[rel_surf_idx]])
##	else:
##		data[i, :] = np.array([theta_ball[i-1], B_ball[i-1], gradpar_ball[i-1], gds2_ball[i-1], gds21_ball[i-1], gds22_ball[i-1], cvdrift_ball[i-1], gbdrift0_ball[i-1], gbdrift_ball[i-1],  gbdrift0_ball[i-1]])
##
#np.savetxt("D_eikcoefs_surf_270_300.txt", data, fmt ='%.9f' )
#with open('revD_eikcoefs_surf_320.txt', 'w') as txt_file:
#	for line in data:
#		txt_file.write("  ".join(str(line)) + '\n')

def check_ball(shat_n, dPdpsi_n):

	dFdpsi_n = (-shat_n*2*np.pi*q_vmec_half[rel_surf_idx]/(rho[rel_surf_idx]*dpsidrho) - b_s[-1]*dPdpsi + c_s[-1])/a_s[-1]
	dqdpsi_n = shat_n*q_vmec_half[rel_surf_idx]/rho[rel_surf_idx]*1/dpsidrho
	aprime_n = -R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi_n +b_s*dPdpsi_n - c_s)*0.5
	dqdr_n = dqdpsi_n*dpsi_dr
	dtdr_st_n = aprime_n + dqdr_n*theta_st_com
	gds2_n =  (dpsidrho)**2*(1/R[rel_surf_idx]**2 + (dqdr_n*theta_st_com)**2 + (np.reshape(q_vmec_half,(-1,1)))**2*(dtdr_st_n**2 + (dt_st_l/dl)**2)+ 2*np.reshape(q_vmec_half,(-1,1))*dqdr_n*theta_st_com*dtdr_st_n)*1/(a_N*B_N)**2

	dBdr_bish_n = B_p/B*(-B_p/R_c + dPdpsi*R + F**2*np.sin(u_ML)/(R**3*B_p))
	gbdrift = dpsidrho*(-2/B*dBdr_bish_n/dpsi_dr + 2*aprime_n*F/R*1/B**3*dBl/dl)
	dPdr_n = dPdpsi*dpsidr
	cvdrift_n =  dpsidrho/np.abs(B)*(-2*(2*dPdr_n/(2*B))/dpsi_dr) + gbdrift

	gradpar_n = a_N/(R*B)*(-dpsi_dr)*(dt_st_l/dl) # gradpar is b.grad(theta)
	
	diff = 0.
	one_m_diff = 1. - diff

	gradpar_ball = reflect_n_append(gradpar_n[rel_surf_idx], 'e')

	theta_ball = reflect_n_append(theta_st_com, 'o')
	cvdrift_ball = reflect_n_append(cvdrift_n[rel_surf_idx], 'e')
	R_ball = reflect_n_append(R[rel_surf_idx], 'e')
	R_ball = R_ball/a_N
	B_ball = reflect_n_append(B[rel_surf_idx], 'e')
	B_ball = B_ball/B_N
	gds2_ball = reflect_n_append(gds2_n[rel_surf_idx], 'e')
	ntheta = len(theta_ball)
	
	delthet = np.diff(theta_ball)
	
	#average of dbetadrho
	#dbetadrho = -np.sum((cvdrift-gbdrift)*bmag**2)/ntheta 
	
	#dbetadrho = beta_prime_input
	
	
	g = gds2_ball/((R_ball*B_ball)**2)
	c = -dPdpsi*dpsidrho*cvdrift_ball*R_ball**2*q_vmec_half[rel_surf_idx]**2/(F[rel_surf_idx]/(a_N*B_N))**2
	
	ch = np.zeros((ntheta,))
	gh = np.zeros((ntheta,))
	
	for i in np.arange(1, ntheta):
	    ch[i] = 0.5*(c[i] + c[i-1]) 
	    gh[i] = 0.5*(g[i] + g[i-1])
	
	cflmax = np.max(np.abs(delthet**2*ch[1:]/gh[1:]))
	
	
	c1 = np.zeros((ntheta,))
	for ig in np.arange(1, ntheta-1):
	    c1[ig] = -delthet[ig]*(one_m_diff*c[ig]+0.5*diff*ch[ig+1])\
	             -delthet[ig-1]*(one_m_diff*c[ig]+0.5*diff*ch[ig])\
	             -delthet[ig-1]*0.5*diff*ch[ig]
	    c1[ig]=0.5*c1[ig]
	
	
	c2 = np.zeros((ntheta,))
	g1 = np.zeros((ntheta,))
	g2 = np.zeros((ntheta,))
	
	for ig in np.arange(1, ntheta):
	    c2[ig] = -0.25*diff*ch[ig]*delthet[ig-1]
	    g1[ig] = gh[ig]/delthet[ig-1]
	    g2[ig] = 1.0/(0.25*diff*ch[ig]*delthet[ig-1]+gh[ig]/delthet[ig-1])
	
	
	psi = np.zeros((ntheta,))
	psi[0] = 0.
	psi[1]=delthet[0]
	psi_prime=psi[1]/g2[1]
	
	for ig in np.arange(1,ntheta-1):
	    psi_prime=psi_prime+c1[ig]*psi[ig]+c2[ig]*psi[ig-1]
	    psi[ig+1]=(g1[ig+1]*psi[ig]+psi_prime)*g2[ig+1]
	
	isunstable = 0
	for ig in np.arange(1,ntheta-1):
	    if(psi[ig]*psi[ig+1] <= 0 ):
	        isunstable = 1
	        #print("instability detected... please choose a different equilibrium")
	return isunstable

print("isunstable for the original eq =",check_ball(shat, dPdpsi))

pdb.set_trace()
want_a_scan = 0
if want_a_scan == 1:
	len1 = 50
	len2 = 100
	#shat_grid = np.linspace(shat*0.1, 10*shat, len1)
	shat_grid = np.linspace(0.10, 0.25, len1)
	#dp_dpsi_grid = np.linspace(dPdpsi*0.1, 10*dPdpsi, len2)
	dp_dpsi_grid = np.linspace(0.08, 0.12, len2)
	ball_scan_arr = np.zeros((len1, len2))
	for i in range(len1):
		for j in range(len2):
			ball_scan_arr[i,j] = check_ball(shat_grid[i], dp_dpsi_grid[j]) 

	from mpl_toolkits import mplot3d
	from matplotlib import pyplot as plt
	
	X, Y = np.meshgrid(shat_grid, dp_dpsi_grid)
	Z = ball_scan_arr
	ax = plt.axes(projection='3d')
	
	#ax.plot3d(shat, dPdpsi, 1, 'or')
	ax.contour3D(X.T, Y.T, ball_scan_arr, 1, cmap='binary')
	plt.show()
pdb.set_trace()
