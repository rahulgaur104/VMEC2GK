#!/usr/bin/env python3

"""
All the routines required to do the main calculation.
"""

import warnings
from configparser import ConfigParser
from textwrap import dedent
from inspect import currentframe, getframeinfo
from ast import literal_eval

import numpy as np
from scipy.signal import find_peaks


def parse_input_file(filename):
    """
    Read a VMEC2GK input file. See input_files/eikcoefs_final_input.txt for an example.
    Returns a dict mapping variable names as strings to their associated values.
    Uses ast.literal_eval to convert the strings obtained from ConfigParser to
    ints/floats/tuples etc. The config file should have a single header 'VMEC2GK'.
    """
    parser = ConfigParser()
    parser.read(filename)
    try:
        return {k: literal_eval(v) for k, v in parser["VMEC2GK"].items()}
    except KeyError:
        raise ValueError("VMEC2GK input files should have a single header 'VMEC2GK'")


def extract_essence(arr, extract_len, mode=0):
    """
    Reducing data from a 2 pi range to a pi range because of up-down symmetry.
    Depending on the value of mode the returned array is reversed (mode = False) or not
    reversed.
    """
    # TODO mode->reverse, update instances of its use
    # TODO return arr[:, :extract_len] if mode else arr[:, extract_len - 1::-1]
    brr = np.zeros((np.shape(arr)[0], extract_len))
    for i in range(np.shape(arr)[0]):
        if mode == 0:
            brr[i] = arr[i][0:extract_len][::-1]
        else:
            brr[i] = arr[i][0:extract_len]

    return brr


def ifft_routine(arr, xm, char, fixdlen, fac):
    """Must be replaced with the inbuilt IFFT"""
    if np.shape(np.shape(arr))[0] > 1:
        colms = np.shape(arr)[0]
        rows = np.shape(arr)[1]
        N = fac * rows + 1  # > 2*(rows)-1
        arr_ifftd = np.zeros((colms, N))
        theta = np.linspace(-np.pi, np.pi, N)
        # TODO
        # f = np.cos if char == "e" else np.sin
        # angles = np.outer(xm, theta)
        # arr_ifftd = np.matmul(arr, f(angles))
        if char == "e":  # even array
            for i in range(colms):
                for j in range(N):
                    for k in range(rows):
                        angle = xm[k] * theta[j]
                        arr_ifftd[i, j] = arr_ifftd[i, j] + np.cos(angle) * arr[i][k]
        else:  # odd array
            for i in range(colms):
                for j in range(N):
                    for k in range(rows):
                        angle = xm[k] * theta[j]
                        arr_ifftd[i, j] = arr_ifftd[i, j] + np.sin(angle) * arr[i][k]

        arr_final = np.zeros((colms, fac * fixdlen))
        if N > fac * fixdlen + 1:  # if longer arrays, interpolate
            theta_n = np.linspace(-np.pi, np.pi, fac * fixdlen)
            for i in range(colms):
                arr_final[i] = np.interp(theta_n, theta, arr_ifftd[i])
        else:
            arr_final = arr_ifftd

    else:
        rows = len(arr)
        N = fac * rows + 1
        arr_ifftd = np.zeros((N,))
        theta = np.linspace(-np.pi, np.pi, N)
        if char == "e":  # even array
            for j in range(N):
                for k in range(rows):
                    angle = xm[k] * theta[j]
                    arr_ifftd[j] = arr_ifftd[j] + np.cos(angle) * arr[k]
        else:  # odd array
            for j in range(N):
                for k in range(rows):
                    angle = xm[k] * theta[j]
                    arr_ifftd[j] = arr_ifftd[j] + np.sin(angle) * arr[k]

        arr_final = np.zeros((fac * fixdlen,))
        if N > fac * fixdlen + 1:
            theta_n = np.linspace(-np.pi, np.pi, fixdlen)
            arr_final = np.interp(theta_n, theta, arr_ifftd)
        else:
            arr_final = arr_ifftd

    return arr_final


def derm(arr, ch, par="e"):
    """
    Finite difference subroutine
    ch = 'l' means difference along the flux surface
    ch = 'r' mean difference across the flux surfaces
    par = 'e' means even parity of the arr. PARITY OF THE INPUT ARRAY
    par = 'o' means odd parity
    """
    temp = np.shape(arr)
    # finite diff along the flux surface for a single array
    if len(temp) == 1 and ch == "l":
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            diff_arr[0, 1:-1] = np.diff(arr[0, :-1], axis=0) + np.diff(
                arr[0, 1:], axis=0
            )
        else:
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 2 * (arr[0, 1] - arr[0, 0])
            diff_arr[0, -1] = 2 * (arr[0, -1] - arr[0, -2])
            diff_arr[0, 1:-1] = np.diff(arr[0, :-1], axis=0) + np.diff(
                arr[0, 1:], axis=0
            )
    # finite diff across surfaces for a single array
    elif len(temp) == 1 and ch == "r":
        d1, d2 = np.shape(arr)[0], 1
        diff_arr = np.zeros((d1, d2))
        arr = np.reshape(arr, (d1, d2))
        diff_arr[0, 0] = 2 * (
            arr[1, 0] - arr[0, 0]
        )  # single dimension arrays like psi, F and q don't have parity
        diff_arr[-1, 0] = 2 * (arr[-1, 0] - arr[-2, 0])
        diff_arr[1:-1, 0] = np.diff(arr[:-1, 0], axis=0) + np.diff(arr[1:, 0], axis=0)

    # Multi-dimensional arrays
    else:
        d1, d2 = np.shape(arr)[0], np.shape(arr)[1]

        diff_arr = np.zeros((d1, d2))
        if ch == "r":  # across surfaces for multi-dim array
            diff_arr[0, :] = 2 * (arr[1, :] - arr[0, :])
            diff_arr[-1, :] = 2 * (arr[-1, :] - arr[-2, :])
            diff_arr[1:-1, :] = np.diff(arr[:-1, :], axis=0) + np.diff(
                arr[1:, :], axis=0
            )

        else:  # along a surface for a multi-dim array
            if par == "e":
                diff_arr[:, 0] = np.zeros((d1,))
                diff_arr[:, -1] = np.zeros((d1,))
                diff_arr[:, 1:-1] = np.diff(arr[:, :-1], axis=1) + np.diff(
                    arr[:, 1:], axis=1
                )
            else:
                diff_arr[:, 0] = 2 * (arr[:, 1] - arr[:, 0])
                diff_arr[:, -1] = 2 * (arr[:, -1] - arr[:, -2])
                diff_arr[:, 1:-1] = np.diff(arr[:, :-1], axis=1) + np.diff(
                    arr[:, 1:], axis=1
                )

    arr = np.reshape(diff_arr, temp)
    return diff_arr


def dermv(arr, brr, ch, par="e"):
    """
    Finite difference subroutine
    brr is the independent variable arr. Needed for weighted finite-difference
    ch = 'l' means difference along the flux surface
    ch = 'r' mean difference across the flux surfaces
    par = 'e' means even parity of the arr. PARITY OF THE INPUT ARRAY
    par = 'o' means odd parity
    """
    temp = np.shape(arr)
    # finite diff along the flux surface for a single array
    if len(temp) == 1 and ch == "l":
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            # diff_arr[0, 1:-1] = (
            #     np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            # )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))

            h1 = np.abs(brr[0, 1]) - np.abs(brr[0, 0])
            h0 = np.abs(brr[0, -1]) - np.abs(brr[0, -2])
            diff_arr[0, 0] = (4 * arr[0, 1] - 3 * arr[0, 0] - arr[0, 2]) / (
                2 * (brr[0, 1] - brr[0, 0])
            )

            # diff_arr[0, -1] =  (-4*arr[0,-1]+3*arr[0, -2]+arr[0, -3])/(
            #     2*(brr[0, -1]-brr[0, -2]))
            # )
            diff_arr[0, -1] = (-4 * arr[0, -2] + 3 * arr[0, -1] + arr[0, -3]) / (
                2 * (brr[0, -1] - brr[0, -2])
            )
            # diff_arr[0, -1] = 2*(arr[0, -1] - arr[0, -2])/(
            #    2*(brr[0, -1] - brr[0, -2])
            # )
            # diff_arr[0, 1:-1] = (
            #     np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            # )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)

    # finite diff across surfaces for a single array
    elif len(temp) == 1 and ch == "r":
        d1, d2 = np.shape(arr)[0], 1
        diff_arr = np.zeros((d1, d2))
        arr = np.reshape(arr, (d1, d2))
        diff_arr[0, 0] = (
            2 * (arr[1, 0] - arr[0, 0]) / (2 * (brr[1, 0] - brr[0, 0]))
        )  # single dimension arrays like psi, F and q don't have parity
        diff_arr[-1, 0] = (
            2 * (arr[-1, 0] - arr[-2, 0]) / (2 * (brr[-1, 0] - brr[-2, 0]))
        )
        # diff_arr[1:-1, 0] = np.diff(arr[:-1,0], axis=0) + np.diff(arr[1:,0], axis=0)
        for i in range(1, d1 - 1):
            h1 = brr[i + 1, 0] - brr[i, 0]
            h0 = brr[i, 0] - brr[i - 1, 0]
            diff_arr[i, 0] = (
                arr[i + 1, 0] / h1**2
                - arr[i, 0] * (1 / h0**2 - 1 / h1**2)
                - arr[i - 1, 0] / h0**2
            ) / (1 / h1 + 1 / h0)

    else:
        d1, d2 = np.shape(arr)[0], np.shape(arr)[1]

        diff_arr = np.zeros((d1, d2))
        if ch == "r":  # across surfaces for multi-dim array
            diff_arr[0, :] = 2 * (arr[1, :] - arr[0, :]) / (2 * (brr[1, :] - brr[0, :]))
            diff_arr[-1, :] = (
                2 * (arr[-1, :] - arr[-2, :]) / (2 * (brr[-1, :] - brr[-2, :]))
            )
            # diff_arr[1:-1, :] = (
            #     np.diff(arr[:-1,:], axis=0) + np.diff(arr[1:,:], axis=0)
            # )
            for i in range(1, d1 - 1):
                h1 = brr[i + 1, :] - brr[i, :]
                h0 = brr[i, :] - brr[i - 1, :]
                diff_arr[i, :] = (
                    arr[i + 1, :] / h1**2
                    + arr[i, :] * (1 / h0**2 - 1 / h1**2)
                    - arr[i - 1, :] / h0**2
                ) / (1 / h1 + 1 / h0)

        else:  # along a surface for a multi-dim array
            if par == "e":
                diff_arr[:, 0] = np.zeros((d1,))
                diff_arr[:, -1] = np.zeros((d1,))
                # diff_arr[:, 1:-1] = (
                #     np.diff(arr[:,:-1], axis=1) + np.diff(arr[:,1:], axis=1)
                # )
                for i in range(1, d2 - 1):
                    h1 = brr[:, i + 1] - brr[:, i]
                    h0 = brr[:, i] - brr[:, i - 1]
                    diff_arr[:, i] = (
                        arr[:, i + 1] / h1**2
                        + arr[:, i] * (1 / h0**2 - 1 / h1**2)
                        - arr[:, i - 1] / h0**2
                    ) / (1 / h1 + 1 / h0)
            else:
                diff_arr[:, 0] = (
                    2 * (arr[:, 1] - arr[:, 0]) / (2 * (brr[:, 1] - brr[:, 0]))
                )
                diff_arr[:, -1] = (
                    2 * (arr[:, -1] - arr[:, -2]) / (2 * (brr[:, -1] - brr[:, -2]))
                )
                # diff_arr[:, 1:-1] = (
                #     np.diff(arr[:,:-1], axis=1) + np.diff(arr[:,1:], axis=1)
                # )
                for i in range(1, d2 - 1):
                    h1 = brr[:, i + 1] - brr[:, i]
                    h0 = brr[:, i] - brr[:, i - 1]
                    diff_arr[:, i] = (
                        arr[:, i + 1] / h1**2
                        + arr[:, i] * (1 / h0**2 - 1 / h1**2)
                        - arr[:, i - 1] / h0**2
                    ) / (1 / h1 + 1 / h0)

    arr = np.reshape(diff_arr, temp)

    return diff_arr


def half_full_combine(arrh, arrf):
    """Function to combine data fronm both the half and the full radial meshes"""
    len0 = len(arrh)
    arr = np.zeros((2 * len0 - 1,))
    # Take the first element from array f as the first element of the final array. The
    # first element in arrh is useless when dpsids < 0
    if arrh[0] < 0 and arrh[0] != -np.inf:
        raise ValueError(
            "half_full_combine: One or more of the dpsids values is of the wrong sign"
        )

    arr[0] = arrf[0]

    for i in np.arange(1, len0):
        arr[2 * i - 2] = arrf[i - 1]
        arr[2 * i - 1] = arrh[i]

    arr[2 * len0 - 2] = arrf[len0 - 1]

    return arr


def nperiod_data_extend(arr, nperiod, istheta=0, par="e"):
    if nperiod > 1:
        if istheta:  # for istheta par='o'
            arr_dum = arr
            for i in range(nperiod - 1):
                arr_app = np.concatenate(
                    (
                        2 * np.pi * (i + 1) - arr_dum[::-1][1:],
                        2 * np.pi * (i + 1) + arr_dum[1:],
                    )
                )
                arr = np.concatenate((arr, arr_app))
        else:
            if par == "e":
                arr_app = np.concatenate((arr[::-1][1:], arr[1:]))
                for i in range(nperiod - 1):
                    arr = np.concatenate((arr, arr_app))
            else:
                arr_app = np.concatenate((-arr[::-1][1:], arr[1:]))
                for i in range(nperiod - 1):
                    arr = np.concatenate((arr, arr_app))
    return arr


def reflect_n_append(arr, ch):
    """
    The purpose of this function is to extend the span of an array from [0, np.pi] to
    [-np.pi,np.pi). ch can either be 'e'(even) or 'o'(odd) depending upon the parity
    of the input array.
    """
    rows = 1
    brr = np.zeros((2 * len(arr) - 1,))
    if ch == "e":
        for i in range(rows):
            brr = np.concatenate((arr[::-1][:-1], arr[0:]))
    else:
        for i in range(rows):
            brr = np.concatenate((-arr[::-1][:-1], np.array([0.0]), arr[1:]))
    return brr


def find_optim_theta_arr(arr, theta_arr, res_par=-2):
    """
    minimizes the size of the input theta array while preserving the local maxima
    and minima of the eikcoefs. Right now it strictly preserves the maximas and
    minimas of B. Provision to preserve maximas and minimas of all the eikcoefs.
    surf 256
    """

    res_par = 3  # 100_negtri

    rows, colms = np.shape(arr)
    idx = []
    idx2 = []
    for i in range(rows):
        peaks, _ = find_peaks(arr[i], height=-1e10)
        peaks = peaks.astype(np.int)
        idx.append(np.ndarray.tolist(peaks))
        peaks2, _ = find_peaks(-arr[i], height=-1e10)
        idx.append(np.ndarray.tolist(peaks2))

    idx.append([0, len(theta_arr) - 1])
    idx = np.sum(idx)
    idx = list(set(idx))
    idx.sort()
    comb_peaks = np.array(idx)
    diff_peaks = np.sort(np.unique(np.diff(np.sort(comb_peaks))))
    # if len(diff_peaks[diff_peaks > 8]) > 0:
    diff_peaks = diff_peaks[diff_peaks >= np.median(diff_peaks)]
    # else:
    #    diff_peaks = diff_peaks[diff_peaks>4]
    diff = int(diff_peaks[0] / 3)
    comb_peaks = np.sort(
        np.abs(
            np.concatenate(
                (
                    peaks - diff,
                    peaks,
                    peaks + diff,
                    peaks2 - diff,
                    peaks2,
                    peaks2 + diff,
                    np.array([0, len(theta_arr) - 1 - diff]),
                )
            )
        )
    )
    # comb_peaks = np.sort(
    #     np.abs(
    #         np.concatenate(
    #             (
    #                 peaks-3*diff,
    #                 peaks-2*diff,
    #                 peaks,
    #                 peaks+2*diff,
    #                 peaks+3*diff,
    #                 peaks2-3*diff,
    #                 peaks2,
    #                 peaks2+3*diff,
    #                 np.array([0, len(theta_arr)-1-diff])
    #             )
    #         )
    #     )
    # )

    diff2 = int(np.mean(np.diff(comb_peaks)) - res_par) + 0
    if diff2 <= 0:
        diff2 = 1
        line_number = getframeinfo(currentframe()).lineno()
        warn_msg = dedent(
            f"""\
            This is not an error, but you should look at the theta optimizer function in
            utils.py script line {line_number}. Try changing the variable res_par in
            this routine to get rid of this warning.
            """
        )
        warnings.warn(warn_msg.replace("\n", " "))

    # 1 - ps1, nperiod =10
    # 2 - ps1 nperiod 5
    comb_peaks_diff = np.diff(comb_peaks)
    idx_gt_diff2 = np.where(comb_peaks_diff > diff2)[0][:]
    for i in idx_gt_diff2:
        j = comb_peaks[i]
        while j < comb_peaks[i + 1]:
            idx2.append(j + diff2)
            j = j + diff2
    comb_peaks = np.concatenate((comb_peaks, np.array(idx2)))
    comb_peaks = np.concatenate((comb_peaks, np.array([len(theta_arr) - 1])))
    comb_peaks = comb_peaks[comb_peaks < len(theta_arr)]
    comb_peaks = np.sort(np.unique(comb_peaks))

    return theta_arr[comb_peaks]


def lambda_create(arr):
    arr1 = np.sort(np.unique(1 / arr))
    diff_arr1 = np.diff(arr1)
    req_diff = np.mean(diff_arr1) / 1.5
    # postri
    # 4 - ps1 nperiod 10
    # 1.5 - ps1 nperiod 5

    # negtri
    # 4 -ps10 nperiod 7
    idx = [arr1[0], arr1[-1]]

    diff_arr_sum = 0
    i = 1
    while i < len(arr1) - 1:
        if diff_arr1[i] <= req_diff:
            diff_arr_sum = diff_arr_sum + req_diff
        else:
            idx.append(arr1[i])
            diff_arr_sum = 0
        i = i + 1

    return np.unique(np.array(idx))


def equalize(arr, b_arr, extremum, len1, spacing=1):
    """len1 is the required len"""
    # theta_arr1 = np.interp(
    #     np.linspace(b_arr[extremum.item()+1], b_arr[-1], int((len1-1)/2)),
    #     b_arr[extremum.item():],
    #     arr[extremum.item():],
    # )
    theta_arr1 = np.interp(
        b_arr[extremum.item() + 1 :][::spacing],
        b_arr[extremum.item() :],
        arr[extremum.item() :],
    )
    # theta_arr2 = np.interp(
    #     np.linspace(b_arr[0], b_arr[extremum.item()-1], int((len1-1)/2)),
    #     b_arr[:extremum.item()][::-1],
    #     arr[:extremum.item()][::-1],
    # )
    theta_arr2 = np.interp(
        b_arr[0 : extremum.item()][::spacing],
        b_arr[: extremum.item() + 1][::-1],
        arr[: extremum.item() + 1][::-1],
    )
    return np.sort(
        np.concatenate((theta_arr2, np.array([arr[extremum.item()]]), theta_arr1))
    )


def symmetrize(theta_arr, b_arr, mode=1, spacing=1):
    """
    this function symmetrizes the theta grid about a local extremum in B
    such a symmetrization makes it easy to specify lambda = 1/b_arr
    This routine is specifically written for the case where B mag has <= 1 trough in
    the range (-np.pi, np.pi)
    Not tested for any other case
    b_arr and theta_arr should only have one extremum right now
    mode = 1 gives only the new theta array
    """
    all_peaks = np.concatenate((find_peaks(b_arr)[0], find_peaks(-b_arr)[0]))
    if len(all_peaks) == 0:
        return theta_arr
    else:
        override_peak_find = 1
        if override_peak_find == 1:
            # relevant peaks is a len 1 array
            relevant_peaks = np.array([np.max(np.sort(all_peaks))])
        else:
            relevant_peaks = all_peaks

        split_b_arr = np.split(b_arr, relevant_peaks)
        split_theta_arr = np.split(theta_arr, relevant_peaks)
        num_arrays = np.shape(split_b_arr)[0]
        theta_new_arr = split_theta_arr[0]

        for i in range(num_arrays):
            if i == 0:
                # creating the theta array containing theta points from theta = 0 up to
                # theta1 s.t B(theta1) = B[0]
                theta_new_arr1 = np.concatenate(
                    (
                        theta_new_arr,
                        theta_arr[relevant_peaks],
                        np.interp(
                            split_b_arr[i], split_b_arr[i + 1], split_theta_arr[i + 1]
                        ),
                    )
                )  # array containing the theta points around the extremum
            elif i == num_arrays - 1:
                # array containing the theta points other than the ones around the
                # extremum
                theta_new_arr2 = split_theta_arr[i][
                    split_b_arr[i] > np.max(split_b_arr[i - 1])
                ]

                theta_new_arr = np.concatenate((theta_new_arr1, theta_new_arr2))

            else:
                print("symmetrize only desgned for 1 extremum! something's wrong")
        if mode == 1:
            return np.concatenate((np.sort(np.unique(theta_new_arr1)), theta_new_arr2))
        else:
            # theta_new_arr1 = equalize(
            #     np.sort(np.unique(theta_new_arr1)),
            #     all_peaks,
            #     len(theta_new_arr)-len(theta_arr),
            # )
            theta_new_arr1 = equalize(
                np.sort(np.unique(theta_new_arr1)),
                np.concatenate(
                    (split_b_arr[0], b_arr[relevant_peaks], split_b_arr[0][::-1])
                ),
                relevant_peaks,
                len(theta_arr) - len(theta_new_arr2),
                spacing,
            )
            return [theta_new_arr1, theta_new_arr2]
