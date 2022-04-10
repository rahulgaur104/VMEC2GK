#!/usr/bin/env python3
"""
This scripts calculates and saves some of the figures of merit we are looking at in context of the ultra high beta equilibria. 
Presently, we calulate:
<omega_d> - bounce averaged drift(precession frequency)
local_mag_shr - local magnetic shear
trapped particle fraction
other eikcoefs like cvdrift


The difference from bishoper_foms.py is that we try to incorporate the trapped particle fraction in the calculation of the precession drift <omega_D>

The trapped particle fraction tp = sqrt{1 - B/B_max} weighted <omega_D> will be
\int <omega_D> sqrt(1-B/B_max)/\int sqrt(1-B/B_max)
"""

import os   
import time
import sys
import pdb
import numpy as np
import pickle
from scipy.integrate import quad
from scipy.interpolate import CubicSpline as cubspl
from scipy.interpolate import interp1d as linspl
from scipy.signal import find_peaks
from scipy.special import erf as erfi
from scipy.optimize import newton 
from matplotlib import pyplot as plt
from inspect import currentframe, getframeinfo
import multiprocessing as mp
import scipy


bishop_dict = sys.argv[1]

dict_file = open(bishop_dict, 'rb')
bishop_dict = pickle.load(dict_file)

qfac = bishop_dict['qfac']
dqdpsi = bishop_dict['dqdpsi']
shat = bishop_dict['shat']


rho = bishop_dict['rho']
dpsidrho = bishop_dict['dpsidrho']
drhodpsi = 1/dpsidrho

F = bishop_dict['F']
dFdpsi = bishop_dict['dFdpsi']
dPdpsi = bishop_dict['dPdpsi']

a_s = bishop_dict['a_s']
b_s = bishop_dict['b_s'] 
c_s = bishop_dict['c_s']

R_ex = bishop_dict['R_ex']
R_c_ex = bishop_dict['R_c_ex']
B_p_ex = bishop_dict['B_p_ex']
B_ex = bishop_dict['B_ex']

dtdr_st = bishop_dict['dtdr_st_ex']
dt_st_l_ex = bishop_dict['dt_st_l_ex']
dl_ex = bishop_dict['dl_ex']
u_ML_ex = bishop_dict['u_ML_ex']
R_c_ex = bishop_dict['R_c_ex']

a_N = bishop_dict['a_N']
B_N = bishop_dict['B_N']
mag_well = bishop_dict['mag_well']
mag_local_peak = bishop_dict['mag_local_peak']
B_local_peak_idx = bishop_dict['B_local_peak']

theta_st_com_ex = bishop_dict['theta_st']
theta_st_com = theta_st_com_ex[theta_st_com_ex <= np.pi]
nperiod = bishop_dict['nperiod']
high_res_fac = bishop_dict['high_res_fac']

spl_st_to_geo_theta = bishop_dict['spl_st_to_geo_theta']
spl_st_to_eqarc_theta = bishop_dict['spl_st_to_eqarc_theta']

pres_scale = bishop_dict['pres_scale']

def reflect_n_append(arr, ch):
	"""
	The purpose of this function is to increase the span of an array from [0, np.pi] to [-np.pi,np.pi). ch can either be 'e'(even) or 'o'(odd) depending upon the parity of the input array.
	"""
	rows = 1
	brr = np.zeros((2*len(arr)-1, ))
	if ch == 'e':
		for i in range(rows):
                    brr = np.concatenate((arr[::-1][:-1], arr[0:]))
	else :
		for i in range(rows):
                    brr = np.concatenate((-arr[::-1][:-1],np.array([0.]), arr[1:]))
	return brr

def equalize(arr, b_arr, extremum, len1, spacing=1):
    #len1 is the required len
    #pdb.set_trace()
    theta_arr1 = np.interp(b_arr[extremum.item()+1:][::spacing], b_arr[extremum.item():], arr[extremum.item():])
    theta_arr2 = np.interp(b_arr[0:extremum.item()][::spacing], b_arr[:extremum.item()+1][::-1], arr[:extremum.item()+1][::-1])
    len2 = len(arr)
    #pdb.set_trace()
    return np.sort(np.concatenate((theta_arr2, np.array([arr[extremum.item()]]), theta_arr1)))


def symmetrize(theta_arr, b_arr, mode=1, spacing=1):
    # this function symmetrizes the theta grid about a local extremum in B
    # such a symmetrization makes it easy to specify lambda = 1/b_arr
    # This routine is specifically written for the case where B has <= 1 trough in 
    # the range (-np.pi, np.pi)
    # Not tested for any other case
    # mode = 1 gives only the new theta array
    #pdb.set_trace()
    all_peaks = np.concatenate((find_peaks(b_arr)[0], find_peaks(-b_arr)[0]))
    if len(all_peaks) == 0:
        return theta_arr
        if mode == 1:
            return np.sort(np.unique(theta_new_arr))
        else:
            return [theta_new_arr]
    else:
        split_b_arr = np.split(b_arr, all_peaks)
        split_theta_arr = np.split(theta_arr, all_peaks)
        #pdb.set_trace()
        num_arrays = np.shape(split_b_arr)[0]
        theta_new_arr = split_theta_arr[0]

        for i in range(num_arrays):
            if i == 0:
                 # theta1 s.t B(theta1) = B[0]
                 theta_new_arr1 =np.concatenate((theta_new_arr, theta_arr[all_peaks], np.interp(split_b_arr[i], split_b_arr[i+1], split_theta_arr[i+1]))) #array containing the theta points around the extremum
            elif i == num_arrays-1:
                 #array containing the theta points other than the ones around the extremum
                 theta_new_arr2 =   split_theta_arr[i][split_b_arr[i]>np.max(split_b_arr[i-1])]

                 theta_new_arr = np.concatenate((theta_new_arr1, theta_new_arr2))

            else:
                print("symmetrize only desgned for 1 extremum! something's wrong")
        if mode == 1:
            return np.concatenate((np.sort(np.unique(theta_new_arr1)), theta_new_arr2))
        else:
            theta_new_arr1 = equalize(np.sort(np.unique(theta_new_arr1)), np.concatenate((split_b_arr[0],b_arr[all_peaks], split_b_arr[0][::-1])), all_peaks, len(theta_arr)-len(theta_new_arr2), spacing)

            return [theta_new_arr1, theta_new_arr2]


def omega_d_bavg(u_ML_arr, dB_dr_arr, B_arr, R_arr, dpsi_dr_arr, theta_arr, mag_well='False', mag_local_well='False', B_local_well_idx=0):
    #Calculates the quantity <omega_d>
    #aprime_arr = aprime_arr[theta_arr <= np.pi]
    u_ML_arr2 = u_ML_arr.copy()
    dB_dr_arr2 = dB_dr_arr.copy()
    B_arr2 = B_arr.copy()
    R_arr2 = R_arr.copy()

    dpsi_dr_arr2 =  dpsi_dr_arr.copy()

    # theta_arr limit cut should be the last
    if mag_well == 'False':
        theta_arr = theta_arr[theta_arr <= np.pi]
        theta_arr = reflect_n_append(theta_arr, 'o')
        theta_arr2 = theta_arr[theta_arr>=0]
        len1 = len(theta_arr)
        len2 = int((len(theta_arr)-1)/2)
        out_arr = np.zeros((len2+1,))
        spl_theta = cubspl(B_arr2, theta_arr[theta_arr>=0])
        spl_B = cubspl(theta_arr[theta_arr>=0], B_arr2)
        spl_trap_frac = cubspl(theta_arr[theta_arr>=0],np.sqrt(1-B_arr2/np.max(B_arr2)))
        spl_u_ML = cubspl(theta_arr[theta_arr>=0], u_ML_arr2)
        spl_dB_dr = cubspl(theta_arr[theta_arr>=0], dB_dr_arr2)
        spl_R = cubspl(theta_arr[theta_arr>=0], R_arr2)
        spl_dpsi_dr = cubspl(theta_arr[theta_arr>=0], dpsi_dr_arr2)
        for i in np.arange(1, len2+1):
            pdb.set_trace()
            #print(i)
            denom = quad(lambda  x,lam =1/B_arr2[i], c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1, 1, weight='cos', wvar=np.pi/200, limit=200)[0]
            numer = quad(lambda x, lam=1/B_arr2[i], c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr: c6(c*np.abs(x))*y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1, 1, weight='cos', wvar=np.pi/100, limit=200)[0]
            #numer_tol = quad(lambda x, lam=1/B_arr2[i], c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi, c4=qfac, c5 = F, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1, 1, weight='cos', wvar=np.pi/100, limit=200)[1]
            #print(' nt=%.3E'%(numer_tol), end=' ')
            if np.isnan(denom) == 1 or i== len2:
                denom = quad(lambda  x,lam =1/B_arr2[i], c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1+1E-5, 1-1E-5, weight='cos', wvar=np.pi/400, limit=400)[0]
                #numer = -quad(lambda x, lam=1/B_arr2[i], c=theta_arr2[i], y=spl_B, z1=spl_R, z2=spl_aprime: 1/z1(np.abs(x)*c)**3*z2(np.abs(x)*c)/y(np.abs(x)*c)**2*(y.derivative()(np.abs(x)*c))*(0.5*lam/np.sqrt(1-lam*y(np.abs(x)*c)) + np.sqrt(1-lam*y(np.abs(x)*c))/y(np.abs(x)*c)), -1+1E-4, 1-1E-4, weight='cos', wvar=np.pi/100, limit=100)[0]
                numer = quad(lambda x, lam=1/B_arr2[i], c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi, c4=qfac, c5 = F, c6 = spl_trap_frac, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:   c6(c*np.abs(x))*y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1+1E-5, 1-1E-5, weight='cos', wvar=np.pi/200, limit=400)[0]
            out_arr[i] = numer/denom
        return out_arr

    elif mag_well == "True" and mag_local_well == "True":

        [theta_arr1, theta_arr2] = symmetrize(theta_st_com[B_local_well_idx:], B_arr2[B_local_well_idx:], mode=2, spacing=1)
        len1 = len(theta_arr1)

        len3 = len(theta_arr2)
        len2 = int((len(theta_arr1)-1)/2+ len(theta_arr2))
        theta_b_array = np.concatenate((theta_arr1[int((len1-1)/2):], theta_arr2))
        theta_t_array = np.concatenate((theta_st_com[:B_local_well_idx], theta_arr1, theta_arr2))
        theta_t_array[0] = 0 # TEMPORARY FIX
        out_arr = np.zeros((len2,))
        #spl_B = linspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0], B_arr2))
        spl_B = linspl(theta_st_com, B_arr2)
        spl_u_ML = linspl(theta_st_com, u_ML_arr2)
        spl_dB_dr = linspl(theta_st_com, dB_dr_arr2)
        spl_R = linspl(theta_st_com, R_arr2)
        spl_dpsi_dr = linspl(theta_st_com, dpsi_dr_arr2)
        #plt.plot(theta_t_array,spl_B(theta_t_array) , '-sr', ms=4); plt.plot(theta_st_com, spl_B(theta_st_com), '-og', ms=3); plt.show()

        for i in range(int((len1-1)/2)):
            theta_lim1 = theta_arr1[i:len1-i]
            #spl_theta = cubspl(B_arr2, theta_arr[theta_arr>=0])
            #denom = np.trapz(B_lim/np.sqrt(1-B_lim/(np.max(B_lim)+1E-3))*1/R_lim**2, theta_lim1)
            if i == 0:
                denom = quad(lambda  x,lam =1/B_arr2[B_local_well_idx+i], y=spl_B, z=spl_R: y(x)*z(x)**2/np.sqrt(1-y(x)*lam), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, weight='cos', wvar=np.pi/200, limit=200)[0]
                denom_tol = quad(lambda  x,lam =1/B_arr2[B_local_well_idx+i], y=spl_B, z=spl_R: y(x)*z(x)**2/np.sqrt(1-y(x)*lam), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, weight='cos', wvar=np.pi/200, limit=200)[1]
                #print(denom_tol)   
            #print(denom)
                numer = quad(lambda x, lam=1/B_arr2[B_local_well_idx+i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac,y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:   c6(c*np.abs(x))*y(x)*z1(x)**2*(-0.5*1/np.sqrt(1-lam*y(x))*z3(x)/(z4(x)*y(x)) + np.sqrt(1-lam*y(x))*(1.5*z3(x)/(z4(x)*y(x)) + c2/c4 + 2*np.sin(z2(x))/(z4(x)*z1(x)) - c1/c5)), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, limit=200)[0]
                numer_tol = quad(lambda x, lam=1/B_arr2[B_local_well_idx+i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac,y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:   c6(c*np.abs(x))*y(x)*z1(x)**2*(-0.5*1/np.sqrt(1-lam*y(x))*z3(x)/(z4(x)*y(x)) + np.sqrt(1-lam*y(x))*(1.5*z3(x)/(z4(x)*y(x)) + c2/c4 + 2*np.sin(z2(x))/(z4(x)*z1(x)) - c1/c5)), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, limit=200)[1]
                #print(numer_tol)   

            else:
                denom, denom_tol = quad(lambda  x,lam =1/B_arr2[B_local_well_idx+i], y=spl_B, z=spl_R: y(x)*z(x)**2/np.sqrt(1-y(x)*lam), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, weight='cos', wvar=np.pi/200, limit=200)
                #print(denom_tol)   
                numer, numer_tol = quad(lambda x, lam=1/B_arr2[B_local_well_idx+i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac,y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(x)*z1(x)**2*(-0.5*1/np.sqrt(1-lam*y(x))*z3(x)/(z4(x)*y(x)) + np.sqrt(1-lam*y(x))*(1.5*z3(x)/(z4(x)*y(x)) + c2/c4 + 2*np.sin(z2(x))/(z4(x)*z1(x)) - c1/c5)), theta_lim1[0]+5E-7, theta_lim1[-1]-5E-7, limit=200)
                #print(numer_tol)   
            #pdb.set_trace()
            out_arr[i] = numer/denom
        #out_arr[len2-int((len1-1)/2):-1] = out_arr[:int((len1-1)/2)]
        #pdb.set_trace()
        out_arr[:int((len1-1)/2)] = out_arr[:int((len1-1)/2)][::-1]
        #out_arr[:int((len1-1)/2)] = out_arr[:int((len1-1)/2)]
        len4 = len(B_arr)
        for i in np.arange(0, len3):
            #pdb.set_trace()
            denom, denom_tol = quad(lambda  x, lam =1/spl_B(theta_arr2[i]), c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1+5E-7, 1-5E-7, weight='cos', wvar=np.pi/200, limit=200)
            #print(denom_tol)   

            numer, numer_tol = quad(lambda x, lam =1/spl_B(theta_arr2[i]), c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac,y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1+5E-7, 1-5E-7, weight='cos', wvar=np.pi/100, limit=200)
            #print(numer_tol)   

            out_arr[i+int((len1-1)/2)] = numer/denom
            #pdb.set_trace()
            if i == len3-1:
                denom = quad(lambda  x, lam =1/spl_B(theta_arr2[i]), c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1+5E-7, 1-5E-7, weight='cos', wvar=np.pi/200, limit=200)[0]
                numer = quad(lambda x, lam =1/spl_B(theta_arr2[i]), c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, c6 = spl_trap_frac,y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1+5E-7, 1-5E-7, weight='cos', wvar=np.pi/100, limit=200)[0]
                out_arr[i+int((len1-1)/2)] = numer/denom


    else:
        [theta_arr1, theta_arr2] = symmetrize(theta_st_com, B_arr2, mode=2)
        len1 = len(theta_arr1)

        len3 = len(theta_arr2)
        len2 = int((len(theta_arr1)-1)/2+ len(theta_arr2))
        #pdb.set_trace()
        theta_b_array = np.concatenate((theta_arr1[int((len1-1)/2):], theta_arr2))
        theta_t_array = np.concatenate((theta_arr1, theta_arr2))
        theta_t_array[0] = 0 # TEMPORARY FIX
        out_arr = np.zeros((len2+1,))
        spl_B = cubspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0], B_arr2))
        spl_u_ML = cubspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0], u_ML_arr2))
        spl_dB_dr = cubspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0],dB_dr_arr2))
        spl_R = cubspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0], R_arr2))
        spl_dpsi_dr = cubspl(theta_t_array, np.interp(theta_t_array,theta_arr[theta_arr>=0],dpsi_dr_arr2))

        for i in range(int((len1-1)/2)):
            theta_lim1 = theta_arr1[i:len1-i]
            #spl_theta = cubspl(B_arr2, theta_arr[theta_arr>=0])
            #denom = np.trapz(B_lim/np.sqrt(1-B_lim/(np.max(B_lim)+1E-3))*1/R_lim**2, theta_lim1)
            denom = quad(lambda  x,lam =1/B_arr2[i], y=spl_B, z=spl_R: y(x)*z(x)**2/np.sqrt(1-y(x)*lam), theta_lim1[0], theta_lim1[-1], weight='cos', wvar=np.pi/200, limit=200)[0]
            denom_tol = quad(lambda  x,lam =1/B_arr2[i], y=spl_B, z=spl_R: y(x)*z(x)**2/np.sqrt(1-y(x)*lam), theta_lim1[0], theta_lim1[-1], weight='cos', wvar=np.pi/200, limit=200)[1]
            #print(denom_tol)   
            #print(denom)
            numer = quad(lambda x, lam=1/B_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(x)*z1(x)**2*(-0.5*1/np.sqrt(1-lam*y(x))*z3(x)/(z4(x)*y(x)) + np.sqrt(1-lam*y(x))*(1.5*z3(x)/(z4(x)*y(x)) + c2/c4 + 2*np.sin(z2(x))/(z4(x)*z1(x)) - c1/c5)), theta_lim1[0], theta_lim1[-1], limit=200)[0]
            numer_tol = quad(lambda x, lam=1/B_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(x)*z1(x)**2*(-0.5*1/np.sqrt(1-lam*y(x))*z3(x)/(z4(x)*y(x)) + np.sqrt(1-lam*y(x))*(1.5*z3(x)/(z4(x)*y(x)) + c2/c4 + 2*np.sin(z2(x))/(z4(x)*z1(x)) - c1/c5)), theta_lim1[0], theta_lim1[-1], limit=200)[1]
            #print(numer_tol)   
            out_arr[i] = numer/denom
        #out_arr[len2-int((len1-1)/2):-1] = out_arr[:int((len1-1)/2)]
        #pdb.set_trace()
        out_arr[:int((len1-1)/2)] = out_arr[:int((len1-1)/2)][::-1]
        #out_arr[:int((len1-1)/2)] = out_arr[:int((len1-1)/2)]
        len4 = len(B_arr)
        for i in np.arange(0, len3):
            #pdb.set_trace()
            denom = quad(lambda  x, lam =1/spl_B(theta_arr2[i]), c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1, 1, weight='cos', wvar=np.pi/200, limit=200)[0]
            #print(denom_tol)   

            numer = quad(lambda x, lam =1/spl_B(theta_arr2[i]), c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1, 1, weight='cos', wvar=np.pi/100, limit=200)[0]

            out_arr[i+int((len1-1)/2)] = numer/denom

            if i == len3-1:
                denom = quad(lambda  x, lam =1/spl_B(theta_arr2[i]), c = theta_arr2[i], y=spl_B, z=spl_R: y(c*np.abs(x))*z(c*np.abs(x))**2/np.sqrt(1-y(np.abs(x)*c)*lam), -1+1E-5, 1-1E-5, weight='cos', wvar=np.pi/200, limit=200)[0]
                numer = quad(lambda x, lam =1/spl_B(theta_arr2[i]), c=theta_arr2[i], c1=dFdpsi, c2=dqdpsi,  c4=qfac, c5 = F, y=spl_B, z1=spl_R, z2=spl_u_ML, z3=spl_dB_dr, z4=spl_dpsi_dr:  y(c*np.abs(x))*z1(c*np.abs(x))**2*(-0.5*1/np.sqrt(1-lam*y(c*np.abs(x)))*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + np.sqrt(1-lam*y(c*np.abs(x)))*(1.5*z3(c*np.abs(x))/(z4(c*np.abs(x))*y(c*np.abs(x))) + c2/c4 + 2*np.sin(z2(c*np.abs(x)))/(z4(c*np.abs(x))*z1(c*np.abs(x))) - c1/c5)), -1+1E-5, 1-1E-5, weight='cos', wvar=np.pi/100, limit=200)[0]
                out_arr[i+int((len1-1)/2)] = numer/denom


    #pdb.set_trace()



    #plt.plot(spl_st_to_geo_theta(theta_b_array[1:]), -out_arr[:]); plt.xlabel(r'$\theta_b$', fontsize=16);
    #plt.label(r'$<\omega_D>$', fontsize=16); plt.title('negtri_ps100', fontsize=16); plt.show()

    return theta_b_array, out_arr




def bishop_omega_d(shat_n, dPdpsi_n):
        #shat_n = 2.50
        #dPdpsi_n = 1.00
        dFdpsi_n = (-shat_n*2*np.pi*(2*nperiod-1)*qfac/(rho*dpsidrho) - b_s[-1]*dPdpsi_n + c_s[-1])/a_s[-1]
        dqdpsi_n = shat_n*qfac/rho*1/dpsidrho
        aprime_n = -R_ex*B_p_ex*(a_s*dFdpsi_n +b_s*dPdpsi_n - c_s)*0.5
        dpsi_dr_ex = -R_ex*B_p_ex
        dqdr_n = dqdpsi_n*dpsi_dr_ex
        dtdr_st_n = -(aprime_n + dqdr_n*theta_st_com_ex)/qfac
        #dtdr_st_n = dtdr_st[rel_surf_idx]
        gradpar_n = a_N/(B_ex)*(-B_p_ex)*(dt_st_l_ex/dl_ex) # gradpar is b.grad(theta)

        gds2_n =  (dpsidrho)**2*(1/R_ex**2 + (dqdr_n*theta_st_com_ex)**2 + (qfac)**2*(dtdr_st_n**2 + (dt_st_l_ex/dl_ex)**2)+ 2*qfac*dqdr_n*theta_st_com_ex*dtdr_st_n)*1/(a_N*B_N)**2
        gds21_n = dpsidrho*dqdpsi_n*dpsidrho*(dpsi_dr_ex*aprime_n)/(a_N*B_N)**2
        gds22_n = (dqdpsi_n*dpsidrho)**2*np.abs(dpsi_dr_ex)**2/(a_N*B_N)**2
        #pdb.set_trace()
        dBdr_bish_n = B_p_ex/B_ex*(-B_p_ex/R_c_ex + dPdpsi_n*R_ex - F**2*np.sin(u_ML_ex)/(R_ex**3*B_p_ex))
        dPdr_n = dPdpsi_n*dpsi_dr_ex


        B = B_ex[theta_st_com_ex <= np.pi]
    #aprime_bounce =  np.interp(theta_bounce, theta_st_com, aprime[rel_surf_idx][theta_st_com_ex <= np.pi])
        print(mag_well)
        if mag_well=="True" and mag_local_peak == "True":
            theta_bounce = theta_st_com
            theta_bounce[0] = 0 # TEMPORARY FIX WHAT IS GOING ON !?
            B_bounce = np.interp(theta_bounce, theta_st_com, B)
            u_ML_bounce =np.interp(theta_bounce, theta_st_com, u_ML_ex[theta_st_com_ex <= np.pi])
            dB_dr_bounce  =  np.interp(theta_bounce, theta_st_com, dBdr_bish_n[theta_st_com_ex <= np.pi])
            R_bounce = np.interp(theta_bounce, theta_st_com_ex, R_ex)
            dpsi_dr_bounce = np.interp(theta_bounce, theta_st_com_ex, dpsi_dr_ex)
            theta_bounce_mod, omega_d_bouce_avg = omega_d_bavg(u_ML_bounce, dB_dr_bounce, B, R_bounce, dpsi_dr_bounce, theta_st_com, mag_well, mag_local_peak, B_local_peak_idx)
            return theta_bounce_mod[1:], omega_d_bouce_avg
            #plt.plot(theta_bounce_mod, omega_d_bouce_avg, '-or', ms=2); plt.show()
        elif mag_well=="True" and mag_local_peak == "False":
            theta_bounce = theta_st_com
            theta_bounce[0] = 0 # TEMPORARY FIX WHAT IS GOING ON !?
            B_bounce = np.interp(theta_bounce, theta_st_com, B)
            u_ML_bounce =np.interp(theta_bounce, theta_st_com, u_ML_ex[theta_st_com_ex <= np.pi])
            dB_dr_bounce  =  np.interp(theta_bounce, theta_st_com, dBdr_bish_n[theta_st_com_ex <= np.pi])
            R_bounce = np.interp(theta_bounce, theta_st_com_ex, R_ex)
            dpsi_dr_bounce = np.interp(theta_bounce, theta_st_com_ex, dpsi_dr_ex)
            theta_bounce_mod, omega_d_bouce_avg = omega_d_bavg(u_ML_bounce, dB_dr_bounce, B, R_bounce, dpsi_dr_bounce, theta_st_com, mag_well)
            return theta_bounce_mod[1:], omega_d_bouce_avg
        else:
            #theta_bounce = symmetrize(theta_st_com, B)
            theta_bounce = theta_st_com
            B_bounce = np.interp(theta_bounce, theta_st_com, B)
            u_ML_bounce =np.interp(theta_bounce, theta_st_com, u_ML_ex[theta_st_com_ex <= np.pi])
            dB_dr_bounce  =  np.interp(theta_bounce, theta_st_com, dBdr_bish_n[theta_st_com_ex <= np.pi])
            R_bounce = np.interp(theta_bounce, theta_st_com_ex, R_ex)
            dpsi_dr_bounce = np.interp(theta_bounce, theta_st_com_ex, dpsi_dr_ex)
            omega_d_bouce_avg = omega_d_bavg(u_ML_bounce, dB_dr_bounce, B_bounce, R_bounce, dpsi_dr_bounce, theta_bounce, mag_well)
            #plt.plot(theta_bounce[1:], -omega_d_bouce_avg[1:], '-or', ms=2); plt.show()
        #plt.plot(1/B_bounce, omega_d_bouce_avg, '-or', ms=2); plt.show()
        #pdb.set_trace()
            return theta_bounce, omega_d_bouce_avg

#x1, y1 = bishop_omega_d(shat, 4*dPdpsi)
#bish_fac = 1
#pdb.set_trace()
#plt.plot(x1, y1); plt.show();
#x2, y2 = bishop_omega_d(shat, dPdpsi)
x3, y3 = bishop_omega_d(shat, 2*dPdpsi)
#x4, y4 = bishop_omega_d(shat, 4*dPdpsi)
#plt.plot(x2[1:], -y2[1:]); 
plt.plot(x3[1:], -y3[1:]); 
#plt.plot(x4[1:], -y4[1:]); 
plt.title(r'$\rho = 0.71, \delta < 0$, ps$%d$'%(pres_scale), fontsize=14)
plt.xlabel(r'$\theta_{geo}$', fontsize=14)
plt.ylabel(r'$\langle\omega_{D}\rangle$', fontsize=14)
#plt.legend(['bish1', 'bish2', 'bish4'], fontsize=14)
#plt.savefig('../output_files_vmec/misc_figures/precessn_drift_negtri_rhop71_ps%d.png'%(pres_scale), dpi=200)
#x3, y3 = bishop_omega_d(shat, 2*dPdpsi)
#ky = np.linspace(0.5, 10, 12) # k_perp rho_i
ky = 3
#thetab = x1[1:] # theta bounce
#omega_D = y1[1:]
#vthi = 3.05*1E4 # for ps10
#fprim = 1.787
#fprim = 10.8
#tprim = 1.037
#eta = tprim/fprim # L_N/L_T
#omegas = ky*vthi*fprim/a_N
#B_spl = cubspl(thetab, np.interp(thetab, theta_st_com_ex, B_ex))
#Bmax = np.max(B_ex)
plt.show()

#gam1 = scipy.special.iv(0, ky**2)*scipy.exp(-ky**2)
#gam2 = scipy.special.iv(1, ky**2)*scipy.exp(-ky**2)

#newton(lambda omega: np.exp(-omega/omegas)*np.sqrt(-omega/omegas)*(np.pi*omegas/omega_D[1])*(eta*(3/2-omega/omega_D[1])-1) + omegas/omega_D[1]*np.exp(omega/omega_D[1])*np.sqrt(np.pi)*(1 + eta*(-1+omega/omega_D[1]))/np.sqrt(-omega/omega_D[1]) + np.exp(omega/omega_D[1])*np.sqrt(np.pi)*np.sqrt(-omega/omega_D[1])+np.pi*omega/omega_D[1] + np.pi*(-omega/omega_D[1] + omegas/omega_D[1]*(1-3/2*eta+eta*omega/omega_D[1])*erfi(np.sqrt(-omega/omega_D[1])))-(2-ky**2*(-gam1 + gam2)*eta*omegas/omega - gam1*(1 + omegas/omega))==0, 10)


#print(newton(lambda omega, omegas = omegas, omega_D= -omega_D[25], gam1 = gam1, gam2 = gam2, eta = eta: scipy.exp(-omega/omegas)*scipy.sqrt(-omega/omegas)*(scipy.pi*omegas/omega_D)*(eta*(3/2-  omega/omega_D)-1) + omegas/omega_D*scipy.exp(omega/omega_D)*scipy.sqrt(scipy.pi)*(1 + eta*(-1+omega/omega_D))/scipy.sqrt(-omega/omega_D) + scipy.exp(omega/omega_D)*scipy.sqrt(np.pi)*scipy.sqrt(-omega/omega_D)+scipy.pi*omega/omega_D + scipy.pi*(-omega/omega_D + omegas/omega_D*(1-3/2*eta+eta*omega/omega_D)*erfi(scipy.sqrt(-omega/omega_D)))-(2-ky**2*(-gam1 + gam2)*eta*omegas/omega - gam1*(1 + omegas/omega)), 0.1))

#TEM growth rate calculation


#plt.rcParams.update({'font.size': 16});
#plt.plot(spl_st_to_eqarc_theta(x1[1:]), -y1[1:], '-og', spl_st_to_eqarc_theta(x2[1:]), -y2[1:], '-or', spl_st_to_eqarc_theta(x3[1:]), -y3[1:], '-ob', ms=1.5);plt.legend(['0.1xp', '1xp', '2xp'], fontsize=16); plt.title('negtri_ps10', fontsize = 16); plt.show()



