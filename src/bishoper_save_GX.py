#!/usr/bin/env python3
"""
This script creates and saves a grid.out file for a GS2 run using local equilibrium data
from a dictionary. There is also provision to change shat and dPdpsi locally.

Usually called from eikcoefs_final_nperiod.py. It can be called from terminal too as
long as there is a dictionary containing all the local equilibrium information.
"""

import os
import sys
import pickle

import numpy as np
import netCDF4 as nc

from utils import (
    nperiod_data_extend,
    find_optim_theta_arr,
    reflect_n_append,
    symmetrize,
)


bishop_dict = sys.argv[1]
parnt_dir_nam = os.path.dirname(os.getcwd())


dict_file = open(bishop_dict, "rb")
bishop_dict = pickle.load(dict_file)

mag_well = bishop_dict["mag_well"]
mag_local_peak = bishop_dict["mag_local_peak"]
B_local_max_0_idx = bishop_dict["B_local_peak"]

eqbm_type = bishop_dict["eqbm_type"]
surf_idx = bishop_dict["surf_idx"]
pres_scale = bishop_dict["pres_scale"]

qfac = bishop_dict["qfac"]
dqdpsi = bishop_dict["dqdpsi"]
shat = bishop_dict["shat"]

rho = bishop_dict["rho"]
dpsidrho = bishop_dict["dpsidrho"]
drhodpsi = 1 / dpsidrho

F = bishop_dict["F"]
dFdpsi = bishop_dict["dFdpsi"]
dPdpsi = bishop_dict["dPdpsi"]

R_ex = bishop_dict["R_ex"]
Z_ex = bishop_dict["Z_ex"]

R_c_ex = bishop_dict["R_c_ex"]
B_p_ex = bishop_dict["B_p_ex"]
B_ex = bishop_dict["B_ex"]

dtdr_st = bishop_dict["dtdr_st_ex"]
dt_st_l_ex = bishop_dict["dt_st_l_ex"]
dl_ex = bishop_dict["dl_ex"]
dBl_ex = bishop_dict["dBl_ex"]
u_ML_ex = bishop_dict["u_ML_ex"]
R_c_ex = bishop_dict["R_c_ex"]

a_N = bishop_dict["a_N"]
B_N = bishop_dict["B_N"]

theta_st_com_ex = bishop_dict["theta_st"]
nperiod = bishop_dict["nperiod"]
high_res_fac = bishop_dict["high_res_fac"]

a_s = bishop_dict["a_s"]
b_s = bishop_dict["b_s"]
c_s = bishop_dict["c_s"]


theta_st_com = theta_st_com_ex[theta_st_com_ex <= np.pi]


def bishop_save(shat_n, dPdpsi_n, pfac):

    dPdpsi_n = pfac * dPdpsi_n
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
    gradpar_n = np.abs(
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
    grho_n = 1 / dpsidrho * dpsi_dr_ex * a_N
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
    cvdrift_n = dpsidrho / np.abs(B_ex) * (-2 * (2 * dPdpsi_n / (2 * B_ex))) + gbdrift_n
    gbdrift0_n = 1 * 2 / (B_ex**3) * dpsidrho * F / R_ex * (dqdr_n * dBl_ex / dl_ex)

    Rprime_ex = nperiod_data_extend(
        np.sin(u_ML_ex[theta_st_com_ex <= np.pi]), nperiod, istheta=0, par="e"
    )
    Zprime_ex = -nperiod_data_extend(
        np.cos(u_ML_ex[theta_st_com_ex <= np.pi]), nperiod, istheta=0, par="o"
    )
    jacob_n = -dpsidrho / (a_N**2 * B_ex * gradpar_n)
    aplot_n = -qfac * theta_st_com_ex

    if nperiod == 1:
        theta_st_com_ex_uniq = theta_st_com_ex
    else:
        theta_st_com_ex_uniq = find_optim_theta_arr(
            np.vstack(
                (
                    gradpar_n,
                    cvdrift_n,
                    gbdrift_n,
                    gbdrift0_n,
                    gds2_n,
                    gds21_n,
                    gds22_n,
                    B_ex,
                )
            ),
            theta_st_com_ex,
        )

    if theta_st_com_ex_uniq[0] != 0:
        print("temp fix at 270 bishoper_save")
        theta_st_com_ex_uniq[0] = 0.0
    # The next 12 lines make sure that the larger theta array is chosen
    # between [0, np.pi] and [np.pi, 2*np.pi]. Doing so and then extending to nperiod>1
    # will make the B symmetrix about the global extrema
    # This will symmetrize the theta grid about any local extrema(e.g. negtri_ps100)
    theta1 = theta_st_com_ex_uniq[theta_st_com_ex_uniq >= 0]
    theta1 = theta1[theta1 <= np.pi]

    theta2 = theta_st_com_ex_uniq[theta_st_com_ex_uniq >= np.pi]
    theta2 = theta2[theta2 <= 2 * np.pi]

    if len(theta1) > len(theta2):
        theta_st_com_uniq_sym = theta1
    else:
        theta2 = np.abs(theta2 - 2 * np.pi)[::-1]
        theta_st_com_uniq_sym = theta2

    # if there is a magnetic well symmetrize provides a theta grid(temp1) corresponding
    # to symmetric B values which can replace the theta grid in the well from
    # theta_st_com_uniq_sym
    override1 = 0
    if mag_well == "True" and override1 == 0:
        override0 = 0
        if override0 == 0:
            # temp1, _ =  symmetrize(
            #     theta_st_com[B_local_max_0_idx:],
            #     np.interp(theta_st_com[B_local_max_0_idx:],
            #     theta_st_com_ex, B_ex),
            #     mode=2,
            #     spacing=3
            # )
            temp1, _ = symmetrize(
                theta_st_com[B_local_max_0_idx:],
                np.interp(theta_st_com[B_local_max_0_idx:], theta_st_com_ex, B_ex),
                mode=2,
                spacing=4,
            )
            theta_st_com_uniq_sym = np.unique(
                np.concatenate(
                    (
                        theta_st_com_uniq_sym[: B_local_max_0_idx - 1],
                        temp1,
                        theta_st_com_uniq_sym[theta_st_com_uniq_sym > temp1[-1]],
                    )
                )
            )
        else:
            temp1, _ = symmetrize(
                theta_st_com[:],
                np.interp(theta_st_com[:], theta_st_com_ex, B_ex),
                mode=2,
                spacing=3,
            )
            theta_st_com_uniq_sym = np.unique(
                np.concatenate(
                    (
                        theta_st_com_uniq_sym[: B_local_max_0_idx - 1],
                        temp1,
                        theta_st_com_uniq_sym[theta_st_com_uniq_sym > temp1[-1]],
                    )
                )
            )

    temp4 = np.diff(theta_st_com_uniq_sym)
    temp5 = []

    for i in range(len(temp4)):
        if temp4[i] > 1e-6:
            temp5.append(theta_st_com_uniq_sym[i])
        else:
            continue
    temp5.append(theta_st_com_uniq_sym[len(temp4)])

    theta_st_com_ex_uniq_sym = nperiod_data_extend(np.array(temp5), nperiod, istheta=1)
    R_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, R_ex)
    Rprime_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Rprime_ex)
    Z_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Z_ex)
    Zprime_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Zprime_ex)
    jacob_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, jacob_n)
    aplot_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, aplot_n)
    B_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, B_ex)
    aprime_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, aprime_n)
    gradpar_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gradpar_n)
    cvdrift_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, cvdrift_n)
    gbdrift_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gbdrift_n)
    gbdrift0_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gbdrift0_n)
    gds2_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gds2_n)
    gds21_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gds21_n)
    gds22_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, gds22_n)
    grho_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, grho_n)

    # plt.plot(theta_st_com_ex_uniq_sym, B_ex_uniq, '-or', ms=2)
    # plt.show()

    gradpar_ball = reflect_n_append(gradpar_uniq, "e")
    theta_ball = reflect_n_append(theta_st_com_ex_uniq_sym, "o")
    cvdrift_ball = reflect_n_append(cvdrift_uniq, "e")
    gbdrift_ball = reflect_n_append(gbdrift_uniq, "e")
    gbdrift0_ball = reflect_n_append(gbdrift0_uniq, "o")
    B_ball = reflect_n_append(B_ex_uniq, "e")
    B_ball = B_ball / B_N
    Rplot_ball = reflect_n_append(R_ex_uniq, "e") / a_N
    jacob_ball = reflect_n_append(jacob_ex_uniq, "e")
    Rprime_ball = reflect_n_append(Rprime_ex_uniq, "e")
    Zplot_ball = reflect_n_append(Z_ex_uniq, "o") / a_N
    Zprime_ball = reflect_n_append(Zprime_ex_uniq, "o")
    aprime_ball = reflect_n_append(aprime_ex_uniq, "o")
    aplot_ball = reflect_n_append(aplot_ex_uniq, "o")
    gds2_ball = reflect_n_append(gds2_uniq, "e")
    gds21_ball = reflect_n_append(gds21_uniq, "o")
    gds22_ball = reflect_n_append(gds22_uniq, "e")
    grho_ball = reflect_n_append(grho_uniq, "e")

    # Repetition check
    rep_idxs = np.where(np.diff(theta_ball) == 0)[0]
    if len(rep_idxs) > 0:
        print("repeated indices found...removing them now")
        del theta_ball[rep_idxs]
        del gradpar_ball[rep_idxs]
        del theta_ball[rep_idxs]
        del cvdrift_ball[rep_idxs]
        del gbdrift_ball[rep_idxs]
        del gbdrift0_ball[rep_idxs]
        del B_ball[rep_idxs]
        del B_ball[rep_idxs]
        del R_ball[rep_idxs]  # FIXME Undefined variable!
        del Rprime_ball[rep_idxs]
        del Z_ball[rep_idxs]  # FIXME Undefined variable!
        del Zprime_ball[rep_idxs]
        del aprime_ball[rep_idxs]
        del aplot_ball[rep_idxs]
        del gds2_ball[rep_idxs]
        del gds21_ball[rep_idxs]
        del gds22_ball[rep_idxs]
        del grho_ball[rep_idxs]

    # ==================================================
    # ==========----------GX_NC_SAVE----------==========
    # ==================================================

    ntheta = len(theta_ball)
    ntheta2 = ntheta - 1
    theta_ball2 = np.delete(theta_ball, int(ntheta) - 1)
    gradpar_sav = np.interp(theta_ball2, theta_ball, gradpar_ball)
    bmag_sav = np.interp(theta_ball2, theta_ball, B_ball)
    grho_sav = np.interp(theta_ball2, theta_ball, grho_ball)
    gbdrift_sav = np.interp(theta_ball2, theta_ball, gbdrift_ball)
    gbdrift0_sav = np.interp(theta_ball2, theta_ball, gbdrift0_ball)
    cvdrift_sav = np.interp(theta_ball2, theta_ball, cvdrift_ball)
    gds21_sav = np.interp(theta_ball2, theta_ball, gds21_ball)
    gds2_sav = np.interp(theta_ball2, theta_ball, gds2_ball)
    gds22_sav = np.interp(theta_ball2, theta_ball, gds22_ball)

    Rplot_sav = np.interp(theta_ball2, theta_ball, Rplot_ball)
    Zplot_sav = np.interp(theta_ball2, theta_ball, Zplot_ball)
    jacob_sav = np.interp(theta_ball2, theta_ball, jacob_ball)
    aplot_sav = np.interp(theta_ball2, theta_ball, aplot_ball)
    Rprime_sav = np.interp(theta_ball2, theta_ball, Rprime_ball)
    Zprime_sav = np.interp(theta_ball2, theta_ball, Zprime_ball)
    aprime_sav = np.interp(theta_ball2, theta_ball, aprime_ball)

    fn = (
        f"{parnt_dir_nam}/output_files/GX_nc_files/"
        f"gx_out_postri_surf_{int(surf_idx)}_nperiod_{nperiod}_nt{ntheta2}.nc"
    )

    ds = nc.Dataset(fn, "w")
    ds.createDimension("z", ntheta2)
    # scalar = ds.createDimension('scalar', 1)

    theta_nc = ds.createVariable("theta", "f8", ("z",))
    bmag_nc = ds.createVariable("bmag", "f8", ("z",))
    gradpar_nc = ds.createVariable("gradpar", "f8", ("z",))
    grho_nc = ds.createVariable("grho", "f8", ("z",))
    gds2_nc = ds.createVariable("gds2", "f8", ("z",))
    gds21_nc = ds.createVariable("gds21", "f8", ("z",))
    gds22_nc = ds.createVariable("gds22", "f8", ("z",))
    gbdrift_nc = ds.createVariable("gbdrift", "f8", ("z",))
    gbdrift0_nc = ds.createVariable("gbdrift0", "f8", ("z",))
    cvdrift_nc = ds.createVariable("cvdrift", "f8", ("z",))
    cvdrift0_nc = ds.createVariable("cvdrift0", "f8", ("z",))
    jacob_nc = ds.createVariable("jacob", "f8", ("z",))

    Rplot_nc = ds.createVariable("Rplot", "f8", ("z",))
    Zplot_nc = ds.createVariable("Zplot", "f8", ("z",))
    aplot_nc = ds.createVariable("aplot", "f8", ("z",))
    Rprime_nc = ds.createVariable("Rprime", "f8", ("z",))
    Zprime_nc = ds.createVariable("Zprime", "f8", ("z",))
    aprime_nc = ds.createVariable("aprime", "f8", ("z",))

    drhodpsi_nc = ds.createVariable(
        "drhodpsi",
        "f8",
    )
    kxfac_nc = ds.createVariable(
        "kxfac",
        "f8",
    )
    Rmaj_nc = ds.createVariable(
        "Rmaj",
        "f8",
    )
    q = ds.createVariable(
        "q",
        "f8",
    )
    shat = ds.createVariable(
        "shat",
        "f8",
    )

    theta_nc[:] = theta_ball2
    bmag_nc[:] = bmag_sav
    gradpar_nc[:] = gradpar_sav
    grho_nc[:] = grho_sav
    gds2_nc[:] = gds2_sav
    gds21_nc[:] = gds21_sav
    gds22_nc[:] = gds22_sav
    gbdrift_nc[:] = gbdrift_sav
    gbdrift0_nc[:] = gbdrift0_sav
    cvdrift_nc[:] = cvdrift_sav
    cvdrift0_nc[:] = gbdrift0_sav
    jacob_nc[:] = jacob_sav

    Rplot_nc[:] = Rplot_sav
    Zplot_nc[:] = Zplot_sav
    aplot_nc[:] = aplot_sav

    Rprime_nc[:] = Rprime_sav
    Zprime_nc[:] = Zprime_sav
    aprime_nc[:] = aprime_sav

    drhodpsi_nc[0] = a_N**2 * B_N / dpsidrho
    kxfac_nc[0] = a_N**2 * B_N * abs(qfac / rho * dpsidrho)
    Rmaj_nc[0] = (np.max(Rplot_nc) + np.min(Rplot_nc)) / 2 * 1 / (a_N)
    q[0] = qfac
    shat[0] = shat_n
    ds.close()

    print("GX file saved succesfully in the dir output_files\n")


pfac = 1.0
bishop_save(shat, dPdpsi, pfac)
