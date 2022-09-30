#!/usr/bin/env python3
"""
This script creates and saves a grid.out file for a GS2 run using local equilibrium data
from a dictionary. There is also a provision to change shat and dPdpsi locally.

Called from the script eikcoefs_final.py
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .utils import (
    nperiod_data_extend,
    find_optim_theta_arr,
    symmetrize,
    reflect_n_append,
    lambda_create,
)


def bishop_to_gs2(bishop_dict: Dict[str, Any], output_dir: Path) -> None:

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

    # TODO Function call depends on many variables outside of its own scope.
    #     Could simply merge with the top level function.
    #     Would be preferable to break into smaller functions defined at module scope.
    def bishop_save(shat_n, dPdpsi_n, pfac):

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
        gds22_n = (
            (dqdpsi_n * dpsidrho) ** 2 * np.abs(dpsi_dr_ex) ** 2 / (a_N * B_N) ** 2
        )
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
        cvdrift_n = (
            dpsidrho / np.abs(B_ex) * (-2 * (2 * dPdpsi_n / (2 * B_ex))) + gbdrift_n
        )
        gbdrift0_n = (
            1 * 2 / (B_ex**3) * dpsidrho * F / R_ex * (dqdr_n * dBl_ex / dl_ex)
        )

        Rprime_ex = nperiod_data_extend(
            np.sin(u_ML_ex[theta_st_com_ex <= np.pi]), nperiod, istheta=0, par="e"
        )
        Zprime_ex = -nperiod_data_extend(
            np.cos(u_ML_ex[theta_st_com_ex <= np.pi]), nperiod, istheta=0, par="o"
        )

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
        # between [0, np.pi] and [np.pi, 2*np.pi]. Doing so and then extending to
        # nperiod>1 will make the B symmetrix about the global extrema
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
                #     spacing=3,
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

        theta_st_com_ex_uniq_sym = nperiod_data_extend(
            np.array(temp5), nperiod, istheta=1
        )
        R_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, R_ex)
        Rprime_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Rprime_ex)
        Z_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Z_ex)
        Zprime_ex_uniq = np.interp(theta_st_com_ex_uniq_sym, theta_st_com_ex, Zprime_ex)
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

        gradpar_ball = reflect_n_append(gradpar_uniq, "e")
        theta_ball = reflect_n_append(theta_st_com_ex_uniq_sym, "o")
        cvdrift_ball = reflect_n_append(cvdrift_uniq, "e")
        gbdrift_ball = reflect_n_append(gbdrift_uniq, "e")
        gbdrift0_ball = reflect_n_append(gbdrift0_uniq, "o")
        B_ball = reflect_n_append(B_ex_uniq, "e")
        B_ball = B_ball / B_N
        R_ball = reflect_n_append(R_ex_uniq, "e") / a_N
        Rprime_ball = reflect_n_append(Rprime_ex_uniq, "e")
        Z_ball = reflect_n_append(Z_ex_uniq, "o") / a_N
        Zprime_ball = reflect_n_append(Zprime_ex_uniq, "o")
        aprime_ball = reflect_n_append(aprime_ex_uniq, "o")
        gds2_ball = reflect_n_append(gds2_uniq, "e")
        gds21_ball = reflect_n_append(gds21_uniq, "o")
        gds22_ball = reflect_n_append(gds22_uniq, "e")
        grho_ball = reflect_n_append(grho_uniq, "e")

        # Repetition check
        rep_idxs = np.where(np.diff(theta_ball) == 0)[0]
        if len(rep_idxs) > 0:
            print("repeated indices found")
            del theta_ball[rep_idxs]
            del gradpar_ball[rep_idxs]
            del theta_ball[rep_idxs]
            del cvdrift_ball[rep_idxs]
            del gbdrift_ball[rep_idxs]
            del gbdrift0_ball[rep_idxs]
            del B_ball[rep_idxs]
            del B_ball[rep_idxs]
            del R_ball[rep_idxs]
            del Rprime_ball[rep_idxs]
            del Z_ball[rep_idxs]
            del Zprime_ball[rep_idxs]
            del aprime_ball[rep_idxs]
            del gds2_ball[rep_idxs]
            del gds21_ball[rep_idxs]
            del gds22_ball[rep_idxs]
            del grho_ball[rep_idxs]

        ntheta = len(theta_ball)
        data = np.zeros((ntheta + 1, 10))
        A1 = []
        A2 = []
        A3 = []
        A4 = []
        A5 = []
        A6 = []
        A7 = []
        A8 = []

        for i in range(ntheta + 1):
            if i == 0:
                data[0, :5] = np.array([int((ntheta - 1) / 2), 0.0, shat, 1.0, qfac])
            else:
                data[i, :] = np.array(
                    [
                        theta_ball[i - 1],
                        B_ball[i - 1],
                        gradpar_ball[i - 1],
                        gds2_ball[i - 1],
                        gds21_ball[i - 1],
                        gds22_ball[i - 1],
                        cvdrift_ball[i - 1],
                        gbdrift0_ball[i - 1],
                        gbdrift_ball[i - 1],
                        gbdrift0_ball[i - 1],
                    ]
                )  # two gbdrift0's because cvdrift0=gbdrift0

                A2.append(
                    "    %.9f    %.9f    %.9f    %.9f\n"
                    % (
                        gbdrift_ball[i - 1],
                        gradpar_ball[i - 1],
                        grho_ball[i - 1],
                        theta_ball[i - 1],
                    )
                )
                A3.append(
                    "    %.9f    %.9f    %.12f    %.9f\n"
                    % (
                        cvdrift_ball[i - 1],
                        gds2_ball[i - 1],
                        B_ball[i - 1],
                        theta_ball[i - 1],
                    )
                )
                A4.append(
                    "    %.9f    %.9f    %.9f\n"
                    % (gds21_ball[i - 1], gds22_ball[i - 1], theta_ball[i - 1])
                )
                A5.append(
                    "    %.9f    %.9f    %.9f\n"
                    % (gbdrift0_ball[i - 1], gbdrift0_ball[i - 1], theta_ball[i - 1])
                )
                A6.append(
                    "    %.9f    %.9f    %.9f\n"
                    % (R_ball[i - 1], Rprime_ball[i - 1], theta_ball[i - 1])
                )
                A7.append(
                    "    %.9f    %.9f    %.9f\n"
                    % (Z_ball[i - 1], Zprime_ball[i - 1], theta_ball[i - 1])
                )
                A8.append(
                    "    %.9f    %.9f    %.9f\n"
                    % (
                        -qfac * reflect_n_append(theta_st_com_ex_uniq_sym, "o")[i - 1],
                        aprime_ball[i - 1],
                        theta_ball[i - 1],
                    )
                )

        A1.append([A2, A3, A4, A5, A6, A7, A8])
        A1 = A1[0]
        if mag_well == "True":
            temp1 = find_peaks(-B_ex_uniq[theta_st_com_ex_uniq_sym <= np.pi])[0]
            assert len(temp1) == 1, "something wrong with the mag_well(bishoper_save)"
            nlambda = len(
                lambda_create(
                    B_ex_uniq[theta_st_com_ex_uniq_sym <= np.pi][temp1.item() :]
                )
            )
            lambda_arr = lambda_create(
                B_ex_uniq[theta_st_com_ex_uniq_sym <= np.pi][temp1.item() :] / B_N
            )

        else:
            nlambda = len(lambda_create(B_ball))
            lambda_arr = lambda_create(B_ball)

        lambda_look = 0
        if lambda_look == 1:
            from matplotlib import pyplot as plt

            plt.plot(theta_ball, B_ball, "-sg", ms=3)
            plt.hlines(1 / lambda_arr, xmin=-10, xmax=10)
            plt.show()

        char = (
            f"grid.out_D3D_{eqbm_type}_pres_scale_{pres_scale}_surf_{surf_idx}"
            f"_nperiod_{nperiod}_nl{nlambda}_nt{len(theta_ball)}"
        )
        if isinstance(pfac, int) != 1:
            before_dec = str(pfac).split(".")[0]
            after_dec = str(pfac).split(".")[1]
            name_suffix = f"{before_dec}p{after_dec}"
            fname_in_txt_rescaled = output_dir / f"{char}_eikcoefs_{name_suffix}"
        else:
            fname_in_txt_rescaled = output_dir / f"{char}_eikcoefs_{pfac}_dPspsi"

        with open(fname_in_txt_rescaled, "w") as g:
            headings = [
                "nlambda\n",
                "lambda\n",
                "ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q\n",
                "gbdrift gradpar grho tgrid\n",
                "cvdrift gds2 bmag tgrid\n",
                "gds21 gds22 tgrid\n",
                "cvdrift0 gbdrift0 tgrid\n",
                "Rplot Rprime tgrid\n",
                "Zplot Zprime tgrid\n",
                "aplot aprime tgrid\n",
            ]
            g.write(headings[0])
            g.write("%d\n" % (nlambda))
            g.writelines(headings[1])

            for i in range(nlambda):
                g.writelines("%.19f\n" % (lambda_arr[i]))

            Rmaj = (np.max(R_ex) + np.min(R_ex)) / (2 * a_N)

            g.writelines(headings[2])
            g.writelines(
                "  %d    %d    %d   %0.1f   %0.1f    %.9f   %.1f   %.2f\n"
                % (
                    (ntheta - 1) / 2,
                    1,
                    (ntheta - 1),
                    a_N**2 * B_N * np.abs(1 / dpsidrho),
                    Rmaj,
                    shat,
                    a_N**2 * B_N * abs(qfac / rho * dpsidrho),
                    qfac,
                )
            )

            for i in np.arange(3, len(headings)):
                g.writelines(headings[i])
                for j in range(ntheta):
                    g.write(A1[i - 3][j])

        return

    pfac = [1.0]
    for i in range(len(pfac)):
        bishop_save(shat, pfac[i] * dPdpsi, pfac[i])

    print(f"GS2 file saved succesfully in the dir {output_dir}")
