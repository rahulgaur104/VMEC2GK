#!/usr/bin/env python3
"""
The purpose of this script is to plot and/or save the s-alpha balooning stability
diagrams for an equilibrium. This script needs the name of the dictionary containing all
the information of a local equilibrium.
Instead of alpha we choose dPdpsi
"""

import os
import sys
import pdb
import numpy as np
import pickle
from inspect import currentframe, getframeinfo
import multiprocessing as mp
from scipy.interpolate import CubicSpline as cubspl
from scipy.sparse.linalg import eigs

# from matplotlib import pyplot as plt
from utils import *

bishop_dict = sys.argv[1]

parnt_dir_nam = os.path.dirname(os.getcwd())

dict_file = open(bishop_dict, "rb")
bishop_dict = pickle.load(dict_file)

qfac = bishop_dict["qfac"]
dqdpsi = bishop_dict["dqdpsi"]
shat = bishop_dict["shat"]

eqbm_type = bishop_dict["eqbm_type"]
surf_idx = bishop_dict["surf_idx"]
pres_scale = bishop_dict["pres_scale"]

rho = bishop_dict["rho"]
dpsidrho = bishop_dict["dpsidrho"]

F = bishop_dict["F"]
dFdpsi = bishop_dict["dFdpsi"]
P = bishop_dict["P"]
dPdpsi = bishop_dict["dPdpsi"]

R_ex = bishop_dict["R_ex"]
R_c_ex = bishop_dict["R_c_ex"]
B_p_ex = bishop_dict["B_p_ex"]
B_ex = bishop_dict["B_ex"]

dtdr_st = bishop_dict["dtdr_st_ex"]
dt_st_l_ex = bishop_dict["dt_st_l_ex"]
dBl_ex = bishop_dict["dBl_ex"]
dl_ex = bishop_dict["dl_ex"]
u_ML_ex = bishop_dict["u_ML_ex"]
R_c_ex = bishop_dict["R_c_ex"]

a_N = bishop_dict["a_N"]
B_N = bishop_dict["B_N"]

theta_st_com_ex = bishop_dict["theta_st"]
nperiod = bishop_dict["nperiod"]

a_s = bishop_dict["a_s"]
b_s = bishop_dict["b_s"]
c_s = bishop_dict["c_s"]


def check_ball(shat_n, dPdpsi_n):
    # shat_n = 2.50
    # dPdpsi_n = 1.00
    dFdpsi_n = (
        -shat_n * 2 * np.pi * (2 * nperiod - 1) * qfac / (rho * dpsidrho)
        - b_s[-1] * dPdpsi_n
        + c_s[-1]
    ) / a_s[-1]
    dqdpsi_n = shat_n * qfac / rho * 1 / dpsidrho
    aprime_n = -R_ex * B_p_ex * (a_s * dFdpsi_n + b_s * dPdpsi_n - c_s) * 0.5
    dpsi_dr_ex = -R_ex * B_p_ex
    dqdr_n = dqdpsi_n * dpsi_dr_ex
    dtdr_st_n = -(aprime_n + dqdr_n * theta_st_com_ex) / qfac
    gradpar_n = (
        a_N / (B_ex) * (-B_p_ex) * (dt_st_l_ex / dl_ex)
    )  # gradpar is b.grad(theta)

    gds2_n = (
        (dpsidrho) ** 2
        * (
            1 / R_ex**2
            + (dqdr_n * theta_st_com_ex) ** 2
            + (qfac) ** 2 * (dtdr_st_n**2 + (dt_st_l_ex / dl_ex) ** 2)
            + 2 * qfac * dqdr_n * theta_st_com_ex * dtdr_st_n
        )
        * 1
        / (a_N * B_N) ** 2
    )
    gds21_n = (
        dpsidrho * dqdpsi_n * dpsidrho * (dpsi_dr_ex * aprime_n) / (a_N * B_N) ** 2
    )
    gds22_n = (dqdpsi_n * dpsidrho) ** 2 * np.abs(dpsi_dr_ex) ** 2 / (a_N * B_N) ** 2
    # pdb.set_trace()
    dBdr_bish_n = (
        B_p_ex
        / B_ex
        * (
            -B_p_ex / R_c_ex
            + dPdpsi_n * R_ex
            - F**2 * np.sin(u_ML_ex) / (R_ex**3 * B_p_ex)
        )
    )
    gbdrift_n = dpsidrho * (
        -2 / B_ex * dBdr_bish_n / dpsi_dr_ex
        + 2 * aprime_n * F / R_ex * 1 / B_ex**3 * dBl_ex / dl_ex
    )
    dPdr_n = dPdpsi_n * dpsi_dr_ex
    cvdrift_n = dpsidrho / np.abs(B_ex) * (-2 * (2 * dPdpsi_n / (2 * B_ex))) + gbdrift_n
    gbdrift0_n = 1 * 2 / (B_ex**3) * dpsidrho * F / R_ex * (dqdr_n * dBl_ex / dl_ex)

    R_ball = reflect_n_append(R_ex, "e") / a_N
    B_ball = reflect_n_append(B_ex, "e") / B_N
    theta_ball = reflect_n_append(theta_st_com_ex, "o")
    gradpar_ball = reflect_n_append(gradpar_n, "e")
    cvdrift_ball = reflect_n_append(cvdrift_n, "e")
    gbdrift_ball = reflect_n_append(gbdrift_n, "e")
    gbdrift0_ball = reflect_n_append(gbdrift0_n, "o")
    gds2_ball = reflect_n_append(gds2_n, "e")
    gds21_ball = reflect_n_append(gds21_n, "o")
    gds22_ball = reflect_n_append(gds22_n, "e")
    ntheta = len(theta_ball)
    # pdb.set_trace()

    delthet = np.diff(theta_ball)
    diff = 0.0
    one_m_diff = 1 - diff
    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.

    g = gds2_ball / ((R_ball * B_ball) ** 2)
    c = (
        -1
        * dPdpsi_n
        * dpsidrho
        * cvdrift_ball
        * R_ball**2
        * qfac**2
        / (F / a_N) ** 2
    )
    f = gds2_ball / B_ball**2 * qfac**2 * (a_N * B_N) ** 2 / F**2 * R_ball**2
    # g = (gds2_ball*gradpar_ball/B_ball)
    # c = -1*(dPdpsi_n/B_N**2)*dpsidrho*cvdrift_ball/(gradpar_ball*B_ball)

    ch = np.zeros((ntheta,))
    gh = np.zeros((ntheta,))
    fh = np.zeros((ntheta,))

    for i in np.arange(1, ntheta):
        ch[i] = 0.5 * (c[i] + c[i - 1])
        gh[i] = 0.5 * (g[i] + g[i - 1])
        fh[i] = 0.5 * (f[i] + f[i - 1])

    cflmax = np.max(np.abs(delthet**2 * ch[1:] / gh[1:]))

    c1 = np.zeros((ntheta,))
    f1 = np.zeros((ntheta,))

    for ig in np.arange(1, ntheta - 1):
        c1[ig] = (
            -delthet[ig] * (one_m_diff * c[ig] + 0.5 * diff * ch[ig + 1])
            - delthet[ig - 1] * (one_m_diff * c[ig] + 0.5 * diff * ch[ig])
            - delthet[ig - 1] * 0.5 * diff * ch[ig]
        )
        c1[ig] = -delthet[ig] * (one_m_diff * c[ig]) - delthet[ig - 1] * (
            one_m_diff * c[ig]
        )
        f1[ig] = -delthet[ig] * (one_m_diff * f[ig]) - delthet[ig - 1] * (
            one_m_diff * f[ig]
        )
        c1[ig] = 0.5 * c1[ig]
        f1[ig] = 0.5 * f1[ig]

    c2 = np.zeros((ntheta,))
    f2 = np.zeros((ntheta,))
    g1 = np.zeros((ntheta,))
    g2 = np.zeros((ntheta,))

    for ig in np.arange(1, ntheta):
        c2[ig] = -0.25 * diff * ch[ig] * delthet[ig - 1]
        f2[ig] = -0.25 * diff * fh[ig] * delthet[ig - 1]
        g1[ig] = gh[ig] / delthet[ig - 1]
        g2[ig] = 1.0 / (
            0.25 * diff * ch[ig] * delthet[ig - 1] + gh[ig] / delthet[ig - 1]
        )

    psi_t = np.zeros((ntheta,))
    # psi_t[int((ntheta-1)/2)] = 0
    psi_t[1] = delthet[0]
    psi_prime = psi_t[1] / g2[1]
    # psi_prime= 10
    # psi_t[int((ntheta-1)/2) + 1] = 0 + psi_prime*delthet[0]

    # for ig in np.arange(1,ntheta-1):
    #    psi_prime=psi_prime+c1[ig]*psi_t[ig]+c2[ig]*psi_t[ig-1]
    #    psi_t[ig+1]=(g1[ig+1]*psi_t[ig]+psi_prime)*g2[ig+1]

    gamma = 0
    # for ig in np.arange(int((ntheta-1)/2),ntheta-1):
    for ig in np.arange(1, ntheta - 1):
        # pdb.set_trace()
        psi_prime = (
            psi_prime
            + 1 * c1[ig] * psi_t[ig]
            + c2[ig] * psi_t[ig - 1]
            + gamma * (f1[ig] * psi_t[ig] + f2[ig] * psi_t[ig - 1])
        )
        psi_t[ig + 1] = (g1[ig + 1] * psi_t[ig] + psi_prime) * g2[ig + 1]

    # pdb.set_trace()
    if np.isnan(np.sum(psi_t)) or np.isnan(np.abs(psi_prime)):
        print("warning NaN  balls")

    isunstable = 0
    for ig in np.arange(1, ntheta - 1):
        if psi_t[ig] * psi_t[ig + 1] <= 0:
            isunstable = 1
            # print("instability detected... please choose a different equilibrium")
    return isunstable


def gamma_ball(shat_n, dPdpsi_n):
    # shat_n = 2.50
    # dPdpsi_n = 1.00
    # nperiod = 2
    # theta_st_com_ex = nperiod_data_extend(theta_st_com_ex, nperiod, istheta=1)
    dFdpsi_n = (
        -shat_n * 2 * np.pi * (2 * nperiod - 1) * qfac / (rho * dpsidrho)
        - b_s[-1] * dPdpsi_n
        + c_s[-1]
    ) / a_s[-1]
    dqdpsi_n = shat_n * qfac / rho * 1 / dpsidrho
    aprime_n = -R_ex * B_p_ex * (a_s * dFdpsi_n + b_s * dPdpsi_n - c_s) * 0.5
    dpsi_dr_ex = -R_ex * B_p_ex
    dqdr_n = dqdpsi_n * dpsi_dr_ex
    dtdr_st_n = -(aprime_n + dqdr_n * theta_st_com_ex) / qfac
    # dtdr_st_n = dtdr_st[rel_surf_idx]
    gradpar_n = (
        a_N / (B_ex) * (-B_p_ex) * (dt_st_l_ex / dl_ex)
    )  # gradpar is b.grad(theta)

    gds2_n = (
        (dpsidrho) ** 2
        * (
            1 / R_ex**2
            + (dqdr_n * theta_st_com_ex) ** 2
            + (qfac) ** 2 * (dtdr_st_n**2 + (dt_st_l_ex / dl_ex) ** 2)
            + 2 * qfac * dqdr_n * theta_st_com_ex * dtdr_st_n
        )
        * 1
        / (a_N * B_N) ** 2
    )
    gds21_n = (
        dpsidrho * dqdpsi_n * dpsidrho * (dpsi_dr_ex * aprime_n) / (a_N * B_N) ** 2
    )
    gds22_n = (dqdpsi_n * dpsidrho) ** 2 * np.abs(dpsi_dr_ex) ** 2 / (a_N * B_N) ** 2
    # pdb.set_trace()
    dBdr_bish_n = (
        B_p_ex
        / B_ex
        * (
            -B_p_ex / R_c_ex
            + dPdpsi_n * R_ex
            - F**2 * np.sin(u_ML_ex) / (R_ex**3 * B_p_ex)
        )
    )
    gbdrift_n = dpsidrho * (
        -2 / B_ex * dBdr_bish_n / dpsi_dr_ex
        + 2 * aprime_n * F / R_ex * 1 / B_ex**3 * dBl_ex / dl_ex
    )
    dPdr_n = dPdpsi_n * dpsi_dr_ex
    cvdrift_n = dpsidrho / np.abs(B_ex) * (-2 * (2 * dPdpsi_n / (2 * B_ex))) + gbdrift_n
    gbdrift0_n = 1 * 2 / (B_ex**3) * dpsidrho * F / R_ex * (dqdr_n * dBl_ex / dl_ex)

    R_ball = reflect_n_append(R_ex, "e") / a_N
    B_ball = reflect_n_append(B_ex, "e") / B_N
    theta_ball = reflect_n_append(theta_st_com_ex, "o")

    gradpar_ball = reflect_n_append(gradpar_n, "e")
    cvdrift_ball = reflect_n_append(cvdrift_n, "e")
    gbdrift_ball = reflect_n_append(gbdrift_n, "e")
    gbdrift0_ball = reflect_n_append(gbdrift0_n, "o")
    gds2_ball = reflect_n_append(gds2_n, "e")
    gds21_ball = reflect_n_append(gds21_n, "o")
    gds22_ball = reflect_n_append(gds22_n, "e")
    ntheta = len(theta_ball)
    # pdb.set_trace()

    # initial value integrator
    delthet = np.diff(theta_ball)
    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    # All the variables of the type *_ball have been normalized
    # g = gds2_ball/((R_ball*B_ball)**2)
    # c = -1*dPdpsi_n*dpsidrho*cvdrift_ball*R_ball**2*qfac**2/(F/a_N)**2
    # f = gds2_ball/B_ball**2*qfac**2*(a_N*B_N)**2/F**2*R_ball**2

    g = np.abs(gradpar_ball) * gds2_ball / (B_ball)
    c = (
        -1
        * dPdpsi_n
        * 1
        / B_N**2
        * dpsidrho
        * cvdrift_ball
        * 1
        / (np.abs(gradpar_ball) * B_ball)
    )
    f = gds2_ball / B_ball**2 * 1 / (np.abs(gradpar_ball) * B_ball)

    g_spl = cubspl(theta_ball, g)
    c_spl = cubspl(theta_ball, c)
    f_spl = cubspl(theta_ball, f)
    # pdb.set_trace()
    len1 = 4 * int(len(g))
    len2 = int((len1 - 1) / 2)

    # Uniform half theta ball

    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)
    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)

    # uniform theta_ball on half points with half the size,
    # i.e., only from [0, (2*nperiod-1)*np.pi]
    theta_ball_u_half = np.concatenate(
        [np.array([0.0]), (theta_ball_u[len2 + 1 :] + theta_ball_u[len2:-1]) / 2]
    )
    h = np.diff(theta_ball_u_half)[1]
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g)
    c_u1 = c_u[len2:]
    f_u1 = f_u[len2:]
    A = np.zeros((len2, len2))
    M = np.zeros((len2, len2))

    for i in range(len2):
        if i == 0:
            A[i, i : i + 2] = np.array(
                [
                    -2 * g_u_half[i + 1] / f_u1[i] * 1 / h**2 + c_u1[i] / f_u1[i],
                    2 * g_u_half[i + 1] / f_u1[i] * 1 / h**2,
                ]
            )
        elif i == len2 - 1:
            A[i, i - 1 : i + 1] = np.array(
                [
                    g_u_half[i] / f_u1[i] * 1 / h**2,
                    -(g_u_half[i] + g_u_half[i + 1]) / f_u1[i] * 1 / h**2
                    + c_u1[i] / f_u1[i],
                ]
            )
        else:
            A[i, i - 1 : i + 2] = np.array(
                [
                    g_u_half[i] / f_u1[i] * 1 / h**2,
                    -(g_u_half[i] + g_u_half[i + 1]) / f_u1[i] * 1 / h**2
                    + c_u1[i] / f_u1[i],
                    g_u_half[i + 1] / f_u1[i] * 1 / h**2,
                ]
            )

    # w, v = np.linalg.eig(A)
    # print(np.max(w))
    # faster compared to np.linalg.eig
    w, v = eigs(A, 3, sigma=1.0, tol=1e-8, OPpart="r")
    idx_max = np.where(w == np.max(w))[0]
    # pdb.set_trace()
    # plt.plot(theta_ball_u[theta_ball_u>0][1:], v[:, idx_max]); plt.show()
    return w[idx_max][0].real


if check_ball(shat, dPdpsi) == 0:
    print(
        "The nominal equilibrium is inf-n ideal ballooning stable.\n"
        "Doing a gamma scan now..."
    )
else:
    print(
        "The nominal equilibrium is inf-n ideal ballooning unstable.\n"
        "Doing a gamma scan now..."
    )


# gamma_ball(shat, dPdpsi)


# No of shat points
len1 = 100
# No of alpha_MHD(proportional to dpdpsi) points
len2 = 100
shat_grid = np.linspace(-3, 10, len1)
dp_dpsi_grid = np.linspace(0, 2, len2)

# to keep the marginal stability data
ball_scan_arr1 = np.zeros((len1, len2))

# to keep the growth rate data
ball_scan_arr2 = np.zeros((len1, len2))

# Setting the number of threads to 1. We don't want multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

nop = int(8)

frameinfo = getframeinfo(currentframe())
print(
    f"Using {nop} processes. To change the number of processes go to line "
    f"{frameinfo.lineno-2} of the bishoper_ball.py routine"
)

pool = mp.Pool(processes=nop)
results1 = np.array(
    [
        [
            pool.apply_async(check_ball, args=(shat_grid[i], dp_dpsi_grid[j]))
            for i in range(len1)
        ]
        for j in range(len2)
    ]
)

for i in range(len2):
    ball_scan_arr1[:, i] = np.array([results1[i, j].get() for j in range(len1)])


results2 = np.array(
    [
        [
            pool.apply_async(gamma_ball, args=(shat_grid[i], dp_dpsi_grid[j]))
            for i in range(len1)
        ]
        for j in range(len2)
    ]
)
for i in range(len2):
    ball_scan_arr2[:, i] = np.array([results2[i, j].get() for j in range(len1)])


from matplotlib import pyplot as plt

X, Y = np.meshgrid(shat_grid, dp_dpsi_grid)
cs = plt.contour(X.T, Y.T, ball_scan_arr1, levels=[0.0])
cs2 = plt.contourf(X.T, Y.T, ball_scan_arr2, cmap="hot")
plt.colorbar()
plt.plot(shat, dPdpsi, "x", color="limegreen", mew=5, ms=8)
rand_idx = 42
path = "{0}/output_files/s-alpha-plots/{1}-{2}.png".format(
    parnt_dir_nam, "s-alpha", rand_idx
)
plt.savefig("%s" % (path))
print("balllooning s-alpha curve successfully saved at %s\n" % (path))
# plt.show()
