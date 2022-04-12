#!/usr/bin/env python3
"""
This script takes an axisymmetric, up-down symmetric VMEC equilibrium and calculates the geometric coefficients required for a gyrokinetics run (with GX/GS2).
"""
import os   
import re
import pdb
import numpy as np
import pickle
from netCDF4 import Dataset
from scipy.integrate import cumtrapz as ctrap
from scipy.interpolate import CubicSpline as cubspl
from scipy.signal import savgol_filter as sf
from matplotlib import pyplot as plt
from inspect import currentframe, getframeinfo
from utils import *

parnt_dir_nam = os.path.dirname(os.getcwd())

#Dictionary to store input variables read from the text file
variables = {}
fname_in_txt = '{0}/{1}/{2}'.format(parnt_dir_nam,'input_files', 'eikcoefs_final_input.txt')

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


vmec_fname        = variables["vmec_fname"]
surf_idx          = int(variables["surf_idx"])
want_to_save_GX   = variables["want_to_save_GX"]
want_to_save_GS2  = variables["want_to_save_GS2"]
want_to_ball_scan = variables["want_to_ball_scan"]
want_foms         = variables["want_foms"]
theta_4_jac       = variables["which_theta"]
nperiod           = int(variables['nperiod'])
high_res_fac      = 42

try:
    pres_scale = eval(vmec_fname.split('_')[-1])
except:
    pres_scale = 0

# get the type of eqbm, i.e., Miller and/or negtri/postri
eqbm_type = vmec_fname.split('_')[2]


vmec_fname_path = '{0}/{1}/{2}.nc'.format(parnt_dir_nam,'input_files', vmec_fname)
rtg = Dataset(vmec_fname_path, 'r')

totl_surfs = len(rtg.variables['phi'][:].data)

surf_min = surf_idx-3
surf_max = surf_idx+3

if totl_surfs < surf_max:
    print("total number of surfaces > maximum surface index") 
    surf_min = totl_surfs-6-1
    surf_max = totl_surfs-1
    surf_idx = totl_surfs-1-3
    print("\n setting surf_max to totl_surfs....\n surf_idx = %d"%(surf_idx))


# fac = 0.5*(no of poloidal points in real space)/(number of modes in Fourier space)
fac = int(4)


P_half = 4*np.pi*1E-7*rtg.variables['pres'][:].data
P_full = 4*np.pi*1E-7*rtg.variables['presf'][:].data

q_vmec_full = -1/(rtg.variables['iotaf'][:].data - 1E-16)
q_vmec_half = -1/(rtg.variables['iotas'][:].data - 1E-16) 


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


P_vmec_data_4_spl   = half_full_combine(P_half, P_full)
q_vmec_data_4_spl   = half_full_combine(q_vmec_half, q_vmec_full)
psi_vmec_data_4_spl = half_full_combine(psi_half, psi_full)
Phi_vmec_data_4_spl = half_full_combine(Phi_f, Phi_half)
#rho_vmec_data_4_spl = np.array([np.abs(1-psi_vmec_data_4_spl[i]/psi_LCFS) for i in range(len(psi_vmec_data_4_spl))])
rho_vmec_data_4_spl = np.array([np.sqrt(np.abs(Phi_vmec_data_4_spl[i]/Phi_LCFS)) for i in range(len(psi_vmec_data_4_spl))])


P_spl   = cubspl(psi_vmec_data_4_spl[::-1], P_vmec_data_4_spl[::-1])
q_spl   = cubspl(psi_vmec_data_4_spl[::-1], q_vmec_data_4_spl[::-1]) 
rho_spl = cubspl(psi_vmec_data_4_spl[::-1], rho_vmec_data_4_spl[::-1])


psi      = rtg.variables['chi'][surf_min:surf_max].data
psi_LCFS = rtg.variables['chi'][-1].data

dpsids   = rtg.variables['chipf'][surf_min+1:surf_max+1].data
psi_half = rtg.variables['chi'][surf_min+1:surf_max+1] + 0.5/(totl_surfs-1)*dpsids

psi      = psi/(2*np.pi)
psi_LCFS = psi_LCFS/(2*np.pi)

psi_half = psi_half/(2*np.pi)
#pdb.set_trace()

psi      = psi_LCFS - psi  # shift and flip sign to ensure consistency b/w VMEC & anlyticl
psi_half = psi_LCFS - psi_half  # shift and flip sign to ensure consistency b/w VMEC & anlyticl

Phi_f    = rtg.variables['phi'][surf_min:surf_max].data
Phi_LCFS = rtg.variables['phi'][-1].data

dPhids   = rtg.variables['phipf'][surf_min:surf_max].data
Phi_half = Phi_f + 0.5/(totl_surfs-1)*dPhids 

# crucial unit conversion being performed here
# MPa to T^2 by multiplying by  \mu  = 4*np.pi*1E-7
P           = 4*np.pi*1E-7*rtg.variables['pres'][surf_min+1:surf_max+1].data
q_vmec      = -1/rtg.variables['iotaf'][surf_min:surf_max].data
q_vmec_half = -1/rtg.variables['iotas'][surf_min+1:surf_max+1].data

xm      = rtg.variables['xm'][:].data
fixdlen = len(xm) 

theta = np.linspace(-np.pi, np.pi, fac*fixdlen+1)

xm_nyq   = rtg.variables['xm_nyq'][:].data
R_mag_ax = rtg.variables['raxis_cc'][:].data


rmnc = rtg.variables['rmnc'][surf_min:surf_max].data   # Fourier coeffs of R. Full mesh quantity.
R    = ifft_routine(rmnc, xm, 'e', fixdlen, fac)

rmnc_LCFS = rtg.variables['rmnc'][-1].data
R_LCFS    =  ifft_routine(rmnc_LCFS, xm, 'e', fixdlen, fac)

zmns_LCFS = rtg.variables['zmns'][-1].data
Z_LCFS    =  ifft_routine(zmns_LCFS, xm, 'o', fixdlen, fac)


no_of_surfs = np.shape(R)[0]

zmns = rtg.variables['zmns'][surf_min:surf_max].data  #
Z    = ifft_routine(zmns, xm, 'o', fixdlen, fac)

bmnc = rtg.variables['bmnc'][surf_min+1:surf_max+1].data   # Fourier coeffs of B, Half mesh quantity, i.e. specified on the radial points in between the full-mesh points. Must be interpolated to full mesh
B    = ifft_routine(bmnc, xm_nyq, 'e', fixdlen, fac)

gmnc  = rtg.variables['gmnc'][surf_min+1:surf_max+1].data   # Fourier coeffs of the Jacobian
g_jac = ifft_routine(gmnc, xm_nyq, 'e', fixdlen, fac)

lmns = rtg.variables['lmns'][surf_min+1:surf_max+1].data #half mesh quantity
lmns = ifft_routine(lmns, xm, 'o', fixdlen, fac)

B_sub_zeta  = rtg.variables['bsubvmnc'][surf_min+1:surf_max+1].data # half mesh quantity
B_sub_zeta  = ifft_routine(B_sub_zeta, xm_nyq, 'e', fixdlen, fac)
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
    g_jac  = extract_essence(g_jac, idx0+1)
    F_half = np.array([np.mean(B_sub_zeta[i]) for i in range(no_of_surfs)])
    
    # B_poloidal from B.cdot J (grad s \times grad phi) (signs may be different)
    B_theta_vmec = np.sqrt(np.abs(B_sub_theta*Phi_LCFS/(2*np.pi*np.reshape(q_vmec_half, (-1,1))*g_jac)))
    
    #g_jac = extract_essence(g_jac, idx0+1)
    lmns = extract_essence(lmns, idx0+1)
    
    F  = np.reshape(F_half, (-1,1))
    u4 = []
    theta_geo = np.array([np.arctan2(Z[i], R[i]-R_mag_ax) for i in range(no_of_surfs)])
    
    
    # All surfaces before surf_min be excluded from our calculations
    fixlen_by_2    = idx0 + 1
    theta_geo_com  = np.linspace(0, np.pi, idx0+1)
    theta_vmec     = np.linspace(0, np.pi, idx0+1)
    #theta_st = theta_vmec + lmns
    theta_st       = theta_vmec - lmns
    B_theta_vmec_2 = np.zeros((no_of_surfs, idx0+1))
else:
    Z           = np.abs(extract_essence(Z, idx0+1, 1))
    R           = extract_essence(R, idx0+1, 1)
    B           = extract_essence(B, idx0+1, 1)
    B_sub_zeta  = extract_essence(B_sub_zeta, idx0+1, 1)
    B_sub_theta = extract_essence(B_sub_theta, idx0+1, 1)
    g_jac       = extract_essence(g_jac, idx0+1, 1)
    F_half      = np.array([np.mean(B_sub_zeta[i]) for i in range(no_of_surfs)])
    
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
    fixlen_by_2    = idx0 + 1
    theta_geo_com  = np.linspace(0, np.pi, idx0+1)
    theta_vmec     = np.linspace(0, np.pi, idx0+1)
    theta_st       = theta_vmec + lmns
    B_theta_vmec_2 = np.zeros((no_of_surfs, idx0+1))

#Get all the relevant quantities from a full-grid onto a half grid by interpolating in the radial direction
for i in np.arange(0, idx0+1):
    R[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), R[:, i])
    Z[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), Z[:, i])


# making sure we choose the right R for a Phi_half
rho_2 = np.array([(np.max(R[i]) - np.min(R[i]))/(np.max(R_LCFS)- np.min(R_LCFS)) for i in range(no_of_surfs)])
#rho = np.array([np.abs(1-psi_half[i]/psi_LCFS) for i in range(no_of_surfs)])
rho   = np.array([np.sqrt(np.abs(Phi_half[i]/Phi_LCFS)) for i in range(no_of_surfs)])


for i in range(no_of_surfs):
    if i == 0:
    	B_theta_vmec_2[i] = np.zeros((idx0+1,))
    else:
    	B_theta_vmec_2[i] = np.sqrt(B[i]**2 - (F[i]/R[i])**2) # This B_theta is calculated using a different method. It must be equal to B_theta_vmec.
                                                              #It is important to note that the interpolated R must be in between the full-grid Rs.



psi = psi_half
F_half_2 = np.abs(np.array([1/ctrap(2*g_jac[i]/(Phi_LCFS*R[i]**2), theta_vmec, initial = 0)[-1] for i in range(no_of_surfs)]))
F_spl = cubspl(psi[::-1], F[::-1])

#fixlen_by_2 = (2*(nperiod-1)+1)*fixlen_by_2
# various derivatives
dRj        = np.zeros((no_of_surfs, fixlen_by_2)) # ML distances
dZj        = np.zeros((no_of_surfs, fixlen_by_2))

dtdr_st    = np.zeros((no_of_surfs, fixlen_by_2))
dB_cart    = np.zeros((no_of_surfs, fixlen_by_2))
dB2l       = np.zeros((no_of_surfs, fixlen_by_2))
dBdr       = np.zeros((no_of_surfs, fixlen_by_2))
dBdt       = np.zeros((no_of_surfs, fixlen_by_2))
dPdr       = np.zeros((no_of_surfs, fixlen_by_2))
dqdr       = np.zeros((no_of_surfs, fixlen_by_2))
B0         = np.zeros((no_of_surfs, fixlen_by_2))
B_p_t      = np.zeros((no_of_surfs, fixlen_by_2))
B_p_t_cart = np.zeros((no_of_surfs, fixlen_by_2))
B_p        = np.zeros((no_of_surfs, fixlen_by_2))
dpsidr     = np.zeros((no_of_surfs, fixlen_by_2))
phi        = np.zeros((fixlen_by_2,))
phi_n      = np.zeros((no_of_surfs, fixlen_by_2))
dt         = np.zeros((no_of_surfs, fixlen_by_2))
u_ML       = np.zeros((no_of_surfs, fixlen_by_2))
u_ML_diff  = np.zeros((no_of_surfs, fixlen_by_2))
curv       = np.zeros((no_of_surfs, fixlen_by_2))   
dR_dpsi    = np.zeros((no_of_surfs, fixlen_by_2))
dR_dt      = np.zeros((no_of_surfs, fixlen_by_2))
dZ_dpsi    = np.zeros((no_of_surfs, fixlen_by_2))
dZ_dt      = np.zeros((no_of_surfs, fixlen_by_2))
dpsidR     = np.zeros((no_of_surfs, fixlen_by_2))
dpsidZ     = np.zeros((no_of_surfs, fixlen_by_2))
dt_dR      = np.zeros((no_of_surfs, fixlen_by_2))
dt_dZ      = np.zeros((no_of_surfs, fixlen_by_2))
jac        = np.zeros((no_of_surfs, fixlen_by_2))


dBr        = np.zeros((no_of_surfs, fixlen_by_2))
u5 = []

psi_diff = derm(psi, 'r')

#VMEC B_p here
B_p =  B_theta_vmec

# Setting the 0th element to 0. Sometimes it's 1E-17 
theta_st[:, 0] = np.zeros((no_of_surfs,)) 
theta_st_com   = theta_st[int(no_of_surfs/2)-1].copy()
#pdb.set_trace()

B_original = B[int(no_of_surfs/2)-1].copy()
#B_local_max_0_idx =np.where(B_original[:3] == np.max(B_original[:3]))[0][0]
B_local_max_0_idx = find_peaks(B_original[:6])[0]


override0 = 0
if len(B_local_max_0_idx) > 0 and override0 == 0:
    mag_local_peak    = "True"
    B_local_max_0_idx = int(B_local_max_0_idx)
else:
    mag_local_peak = "False"
    B_local_max_0_idx = int(0)
override1 = 0
local_minimas = find_peaks(-B_original[B_local_max_0_idx:])[0]

#pdb.set_trace()
if len(local_minimas) > 0 and override1 == 0:
    mag_well = "True"
else:
    mag_well = "False"


B_original2 = B[int(no_of_surfs/2)-1].copy()
#theta_st_com = np.sort(symmetrize(theta_st_com.copy(), B[int(no_of_surfs/2)-1]))
theta_st_com = theta_st[int(no_of_surfs/2)-1].copy()
#print("theta_st_com is not uniformly spaced so derm can't be central difference")
print("Note that theta_st is not uniformly spaced so derm can't be central difference\n")
#pdb.set_trace()


for i in range(0,no_of_surfs):
    R[i]         = np.interp(theta_st_com, theta_st[i], R[i]) # Interpolation of full grid quantities to a half-grid surfaces requires i - > i-1. theta_st_is naturally on half-grid but R was interpolated(radially) to a half-grid.
    Z[i]         = np.interp(theta_st_com, theta_st[i], Z[i])
    B[i]         = np.interp(theta_st_com, theta_st[i], B[i])
    B_p[i]       = np.interp(theta_st_com, theta_st[i], B_p[i])
    theta_st[i]  = theta_st_com
    theta_geo[i] = np.arctan2(Z[i], R[i]-R_mag_ax)
  

dt_st_l = derm(theta_st, 'l', 'o')


dR_dpsi = derm(R, 'r')/psi_diff
dR_dt   = dermv(R, theta_st, 'l', 'e')
dZ_dpsi = derm(Z, 'r')/psi_diff
dZ_dt   = dermv(Z, theta_st, 'l', 'o')

jac = dR_dpsi*dZ_dt - dZ_dpsi*dR_dt

dpsidR = dZ_dt/jac
dpsidZ = -dR_dt/jac
dt_dR  = -dZ_dpsi/jac
dt_dZ  =  dR_dpsi/jac


for i in range(no_of_surfs):
     dRj[i, :]  = derm(R[i,:], 'l', 'e')
     dZj[i, :]  = derm(Z[i,:], 'l', 'o') 
     phi        = np.arctan2(dZj[i,:], dRj[i,:])
     phi_n[i,:] = np.concatenate((phi[phi>=0]-np.pi/2, phi[phi<0]+3*np.pi/2)) 

u_ML = np.arctan2(derm(Z, 'l', 'o'), derm(R, 'l', 'e'))


dl  = np.sqrt(derm(R, 'l', 'e')**2 + derm(Z, 'l', 'o')**2)
# The way u is defined in the notes, it should decrease with increasing l which is different from the value being calculated here
R_c = dl/derm(phi_n, 'l', 'o')
B2  = B**2

#plt.plot(theta_geo_m[10], theta_st_com_m, theta_geo_com, theta_st[10]); plt.show() 


for i in np.arange(no_of_surfs):
    B_p_t_cart[i, :] = np.sqrt(dpsidR[i]**2+ dpsidZ[i]**2)/R[i]

B_p = B_p_t_cart

B0 = [np.sqrt(B_p_t_cart[i]**2 + (F[i]/R[i,:])**2) for i in np.arange(surf_min, no_of_surfs)]

q_int = np.zeros((no_of_surfs,))
L     = np.cumsum(dl, axis =1)/2

for i in np.arange(no_of_surfs):

    q_int[i] = 2*np.trapz(F[i]/(2*R[i]*np.pi*np.abs(np.sqrt(dpsidR[i]**2 + dpsidZ[i]**2))), L[i])
    jac_chk  = F[i]*jac/(R[i]*q_int[i])
    #print((np.max(jac_chk[i])-np.min(jac_chk[i]))/np.mean(np.abs(jac_chk[i])), sep = '   ')
    #q_int[i] = 2*np.trapz(F[i]/(2*R[i]*np.pi*np.abs(R[i]*B_p_t[i])), L[i])


#plt.plot(q_vmec[:-1], q_int[:-1], q_vmec[:-1], q_vmec[:-1]); plt.show()
spl_st_to_col_theta = cubspl(theta_st_com, theta[theta>=0])
spl_st_to_geo_theta = cubspl(theta_st_com, theta_geo[int(no_of_surfs/2)-1])



####################----------------EIKCOEFS CALCULATION ---------------------------##################


rel_surf_idx = surf_idx - surf_min -1 # -1 in the end becuase of python's 0 based indexing 

shat = rho[rel_surf_idx]/q_vmec_half[rel_surf_idx]*q_spl.derivative()(psi[rel_surf_idx])/rho_spl.derivative()(psi[rel_surf_idx])

B_p_ex          = nperiod_data_extend(B_p[rel_surf_idx], nperiod)
R_ex            = nperiod_data_extend(R[rel_surf_idx], nperiod)
Z_ex            = nperiod_data_extend(Z[rel_surf_idx], nperiod, istheta=0, par='o')
u_ML_ex         = nperiod_data_extend(u_ML[rel_surf_idx], nperiod)
R_c_ex          = nperiod_data_extend(R_c[rel_surf_idx], nperiod)
theta_st_com_ex = nperiod_data_extend(theta_st_com, nperiod, istheta=1)

L_st_ex         = np.concatenate((np.array([0.]), np.cumsum(np.sqrt(np.diff(R_ex)**2 + np.diff(Z_ex)**2))))


# the factor of 2 in the front cancels with the the 2 in 2*pi in the expression for shat_test
a_s = -(2*q_vmec_half[rel_surf_idx]/F[rel_surf_idx]*theta_st_com_ex + 2*F[rel_surf_idx]*q_vmec_half[rel_surf_idx]*ctrap(1/(R_ex**2*B_p_ex**2), theta_st_com_ex, initial=0))  
b_s = -(2*q_vmec_half[rel_surf_idx]*ctrap(1/(B_p_ex**2), theta_st_com_ex, initial=0))
c_s =  (2*q_vmec_half[rel_surf_idx]*ctrap((2*np.sin(u_ML_ex)/R_ex -  2/R_c_ex)*1/(R_ex*B_p_ex), theta_st_com_ex, initial=0))
    
dl = np.sqrt(derm(R, 'l', 'e')**2 + derm(Z, 'l', 'o')**2)

dl_ex = nperiod_data_extend(dl[rel_surf_idx], nperiod)

dFdpsi   = F_spl.derivative()(psi[rel_surf_idx])
dPdpsi   = P_spl.derivative()(psi[rel_surf_idx])
dqdpsi   = q_spl.derivative()(psi[rel_surf_idx])

dpsidrho = 1/rho_spl.derivative()(psi[rel_surf_idx])

#only for debugging
#B_N =  1
#a_N = 1

#area_LCFS = np.abs(ctrap(Z_LCFS, R_LCFS, initial=0)[-1])
#a_N = np.sqrt(area_LCFS/np.pi)
#B_N = np.abs(Phi_LCFS/area_LCFS)

area_LCFS = np.abs(ctrap(Z_LCFS, R_LCFS, initial=0)[-1])
a_N1      = np.sqrt(area_LCFS/np.pi)
a_N       = rtg.variables['Aminor_p'][:].data # imported from VMEC, same as the a_N 3 lines above
B_N       = np.abs(Phi_LCFS/(np.pi*a_N**2))
# corresponds to dPhidrho = 1

shat_test = -(1/(2*np.pi*(2*nperiod-1)*q_vmec_half[rel_surf_idx]))*(rho[rel_surf_idx]*dpsidrho)*(a_s[-1]*dFdpsi +b_s[-1]*dPdpsi - c_s[-1])

dFdpsi_4_ball = (-shat*2*np.pi*(2*nperiod-1)*q_vmec_half[rel_surf_idx]/(rho[rel_surf_idx]*dpsidrho) - b_s[-1]*dPdpsi + c_s[-1])/a_s[-1]

# The 0.5 in the front is to cancel the factor of 2 in that has been used in a_s, b_s and c_s
##aprime_bish = R[rel_surf_idx]*B_p[rel_surf_idx]*(a_s*dFdpsi_4_ball +b_s*dPdpsi - c_s)*0.5
aprime_bish_1 = -R_ex*B_p_ex*(a_s*dFdpsi_4_ball +b_s*dPdpsi - c_s)*0.5


aprime_bish = R_ex*B_p_ex*(a_s*dFdpsi +b_s*dPdpsi - c_s)*0.5

dpsidR_ex = nperiod_data_extend(dpsidR[rel_surf_idx], nperiod)
dt_dR_ex  = nperiod_data_extend(dt_dR[rel_surf_idx], nperiod, par = 'o')
dt_dZ_ex  = nperiod_data_extend(dt_dZ[rel_surf_idx], nperiod)
dpsidZ_ex = nperiod_data_extend(dpsidZ[rel_surf_idx], nperiod, istheta=0, par = 'o')

dtdr_st_ex = -(dt_dR_ex*dpsidR_ex + dt_dZ_ex*dpsidZ_ex)/np.sqrt(dpsidR_ex**2 + dpsidZ_ex**2)

dpsi_dr_ex = -R_ex*B_p_ex
dqdr_ex    = dqdpsi*dpsi_dr_ex
dPdr_ex    = dPdpsi*dpsi_dr_ex 
aprime     = -(np.reshape(q_vmec_half, (-1,1))*dtdr_st_ex + dqdr_ex*theta_st_com_ex)


#plt.plot(theta_st_com, aprime[rel_surf_idx], theta_st_com, -aprime_bish_1, theta_st_com, -aprime_bish); plt.legend(['aprime_fd', 'aprime_bish_corr', 'aprime_bish'])
#plt.show()
#pdb.set_trace()

B_ex = nperiod_data_extend(B[rel_surf_idx], nperiod)
B_ex_original = nperiod_data_extend(B_original, nperiod)
#B_ex = nperiod_data_extend(B[rel_surf_idx], nperiod)

dt_st_l_ex = nperiod_data_extend(dt_st_l[rel_surf_idx], nperiod) 
# always use this method of calculating dt_st_l_ex instead of derm(theta_st_com_ex, 'l','o'), future me

dBdr_bish    = B_p_ex/B_ex*(-B_p_ex/R_c_ex + dPdpsi*R_ex - F**2*np.sin(u_ML_ex)/(R_ex**3*B_p_ex))
dtdr_st_bish = -(aprime_bish_1 + dqdr_ex*theta_st_com_ex)/np.reshape(q_vmec_half, (-1,1))

gds2  =  (dpsidrho)**2*(1/R_ex**2 + (dqdr_ex*theta_st_com_ex)**2 + (np.reshape(q_vmec_half,(-1,1)))**2*(dtdr_st_ex**2 + (dt_st_l_ex/dl_ex)**2)+\
          2*np.reshape(q_vmec_half,(-1,1))*dqdr_ex*theta_st_com_ex*dtdr_st_ex)*1/(a_N*B_N)**2

gds21 = dpsidrho*dqdpsi*dpsidrho*(dpsi_dr_ex*aprime)/(a_N*B_N)**2
gds22 = (dqdpsi*dpsidrho)**2*np.abs(dpsi_dr_ex)**2/(a_N*B_N)**2
grho  = 1/dpsidrho*dpsi_dr_ex*a_N


dB2l_dl_ex = dermv(B_ex**2, L_st_ex, 'l', 'e')[0]
#dB2l       = derm(B2, 'l', 'e')
#dB2l_dl_ex = nperiod_data_extend(dB2l[rel_surf_idx], nperiod, istheta=0, par='o')/dl_ex
dBl        = derm(B, 'l', 'e')


want_2_filter = 1
if want_2_filter == 1:
    print('filter-L2-norm-err = %.4f'%(np.linalg.norm(dBl[2]-np.concatenate((np.array([0]),sf(dBl[2], 9,2)[1:])))))
    dBl = np.concatenate((np.zeros((len(B),1)), sf(dBl, 9,2)[:, 1:]), axis=1)
else:
    dBl =  derm(B, 'l', 'e') 

dBl_ex    = nperiod_data_extend(dBl[rel_surf_idx], nperiod, istheta=0, par='o')
dBl_dl_ex = dermv(B_ex, L_st_ex, 'l', 'e')[0]

dB_dr     = derm(B_ex, 'l', 'e')/dt_st_l_ex*dtdr_st_ex + nperiod_data_extend(derm(B, 'r')[rel_surf_idx], nperiod)/psi_diff[rel_surf_idx]*dpsi_dr_ex

#plt.plot(theta_st_com_ex, dB_dr[0], theta_st_com_ex, dBdr_bish[2]); plt.show() 
#pdb.set_trace()

#gbdrift_bish = dpsidrho/(B_ex**3)*(-2*B_ex**2*dBdr_bish/dpsi_dr_ex + aprime_bish_1*F/R_ex*sf(dB2l_ex/dl_ex, 9, 2)*1/B_ex)
gbdrift_bish = dpsidrho/(B_ex**3)*(-2*B_ex**2*dBdr_bish/dpsi_dr_ex + aprime_bish_1*F/R_ex*dB2l_dl_ex*1/B_ex)
#gbdrift = dpsidrho*(-2/B_ex*dBdr_bish/dpsi_dr_ex + 2*aprime*F/R_ex*1/B_ex**3*dBl_ex/dl_ex)
gbdrift  = dpsidrho*(-2/B_ex*dBdr_bish/dpsi_dr_ex + 2*aprime*F/R_ex*1/B_ex**3*dBl_dl_ex)
gbdrift0 =  1*2/(B_ex**3)*dpsidrho*np.reshape(F, (-1,1))/R_ex*(dqdr_ex*dBl_dl_ex)

cvdrift  =  dpsidrho/np.abs(B_ex)*(-2*(2*dPdpsi/(2*B_ex))) + gbdrift

gradpar = a_N/(R_ex*B_ex)*(-dpsi_dr_ex)*(dt_st_l_ex/dl_ex) # gradpar is b.grad(theta)
#gradpar = a_N/(R_ex*B_ex)*(-dpsi_dr_ex)*1/dermv(L_st_ex, theta_st_com_ex, 'l', 'o') # gradpar is b.grad(theta)
gradpar_lim   = gradpar[theta_st_com_ex <= np.pi]
B_lim         = B_ex[theta_st_com_ex <= np.pi]
B_p_lim       = B_p_ex[theta_st_com_ex <= np.pi]
theta_lim     = theta_st_com_ex[theta_st_com_ex <= np.pi]
L_eqarc       = ctrap(B_p_lim/(B_lim*gradpar_lim), theta_lim, initial=0)
gradpar_eqarc = np.pi/ctrap(1/(gradpar_lim), theta_lim, initial=0)[-1]
theta_eqarc   = ctrap(B_lim/B_p_lim*gradpar_eqarc, L_eqarc, initial=0)

spl_st_to_eqarc_theta = cubspl(theta_st_com_ex[theta_st_com_ex <=np.pi], theta_eqarc)
    
    

bishop_dict = {'a_N':a_N, 'B_N':B_N, 'mag_well':mag_well, 'mag_local_peak':mag_local_peak,  'B_local_peak':int(B_local_max_0_idx), 'pres_scale':pres_scale,\
               'eqbm_type':eqbm_type, 'surf_idx':surf_idx, 'high_res_fac':high_res_fac, 'qfac':q_vmec_half[rel_surf_idx], 'shat':shat, 'dqdpsi':dqdpsi,\
               'P':P[rel_surf_idx], 'dPdpsi':dPdpsi, 'F':F[rel_surf_idx], 'dFdpsi':dFdpsi, 'rho': rho[rel_surf_idx], 'dpsidrho':dpsidrho, 'theta_st':theta_st_com_ex,\
               'nperiod':nperiod, 'a_s':a_s, 'b_s':b_s, 'c_s':c_s, 'R_ex':R_ex, 'Z_ex':Z_ex, 'R_c_ex':R_c_ex, 'B_p_ex' :B_p_ex, 'B_ex':B_ex, 'dBl_ex':dBl_ex,\
               'dt_st_l_ex': dt_st_l_ex, 'dtdr_st_ex' : dtdr_st_ex, 'dl_ex':dl_ex,'u_ML_ex':u_ML_ex, 'gds2_ex':gds2, 'spl_st_to_geo_theta':spl_st_to_geo_theta,\
               'spl_st_to_eqarc_theta':spl_st_to_eqarc_theta}


#print("rhoc=", (np.max(R[rel_surf_idx])-np.min(R[rel_surf_idx]))/(np.max(R_LCFS)-np.min(R_LCFS)))
#pdb.set_trace()

# saving bishop dict only once
if want_to_ball_scan == 1 or want_to_save_GS2 == 1 or want_foms == 1:
    dict_file = open("bishop_dict.pkl", 'wb')
    pickle.dump(bishop_dict, dict_file)
    dict_file.close()


if want_to_ball_scan == 1:
    # run another script from this script
    os.system('python3 bishoper_ball.py bishop_dict.pkl')
    #print("HERE I AM")
    #time.sleep(10)

if want_to_save_GS2 == 1:
    # run another script from this script
    os.system('python3 bishoper_save_GS2.py bishop_dict.pkl')


if want_to_save_GX == 1:
    # run another script from this script
    os.system('python3 bishoper_save_GX.py bishop_dict.pkl')



















