#!/usr/bin/env python3
"""
For plotting flux surfaces and P and q profiles.
"""

from pathlib import Path

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from .utils import (
    parse_input_file,
    ifft_routine,
    extract_essence,
)


def plot_profiles(vmec_filename: Path, output_dir: Path = Path(".")) -> None:

    rtg = nc.Dataset(vmec_filename, "r")

    surf_min = 0
    surf_max = len(rtg.variables["phi"][:].data)

    # =====================================
    # Get R and Z from Fourier coefficients

    # fac = 0.5*(no of poloidal points in real space)/(number of modes in Fourier space)
    xm = rtg.variables["xm"][:].data
    fixdlen = len(xm)
    fac = 4

    rmnc = rtg.variables["rmnc"][surf_min:surf_max].data
    R = ifft_routine(rmnc, xm, "e", fixdlen, fac)

    zmns = rtg.variables["zmns"][surf_min:surf_max].data
    Z = ifft_routine(zmns, xm, "o", fixdlen, fac)

    rmnc_LCFS = rtg.variables["rmnc"][-1].data
    R_LCFS = ifft_routine(rmnc_LCFS, xm, "e", fixdlen, fac)

    zmns_LCFS = rtg.variables["zmns"][-1].data
    Z_LCFS = ifft_routine(zmns_LCFS, xm, "o", fixdlen, fac)

    # Reduce data from 2*pi range to pi using up-down symettry
    idx0 = int((xm[-1] + 1) * fac / 2)
    mode = R[0][0] >= R[0][idx0]
    R = extract_essence(R, idx0 + 1, mode)
    Z = np.abs(extract_essence(Z, idx0 + 1, mode))

    # Get all the relevant quantities from a full-grid onto a half grid by interpolating
    # in the radial direction
    Phi_f = rtg.variables["phi"][surf_min:surf_max].data
    dPhids = rtg.variables["phipf"][surf_min:surf_max].data
    Phi_half = Phi_f + 0.5 / (surf_max - 1) * dPhids
    for i in np.arange(0, idx0 + 1):
        R[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), R[:, i])
        Z[:, i] = np.interp(np.sqrt(Phi_half[:]), np.sqrt(Phi_f[:]), Z[:, i])

    # =====================
    # Get rhoc

    rhoc = (np.max(R[:-1], axis=1) - np.min(R[:-1], axis=1)) / (
        np.max(R[-1]) - np.min(R[-1])
    )

    # =====================
    # Get P and q

    # Convert MPa to T^2 by multiplying by \mu  = 4*np.pi*1E-7
    mu = 4e-7 * np.pi
    P = mu * rtg.variables["pres"][surf_min + 1 : surf_max + 1].data
    q_vmec_half = -1 / rtg.variables["iotas"][surf_min + 1 : surf_max + 1].data

    # =====================
    # Create plots

    # P vs rhoc
    plt.plot(rhoc, P, "-b", linewidth=2)
    plt.xlabel("rhoc", fontsize=20)
    plt.ylabel("P", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "P_vs_rhoc.png")
    plt.close()

    # q vs rhoc
    plt.plot(rhoc, q_vmec_half, "-b", linewidth=2)
    plt.xlabel("rhoc", fontsize=20)
    plt.ylabel("q", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "q_vs_rhoc.png")
    plt.close()

    # flux surfaces
    # magnetic axis
    R_mag_ax = rtg.variables["raxis_cc"][:].data.item()
    plt.plot(R_mag_ax, 0, "xk", ms=12, mew=3)
    # last closed flux surface
    plt.plot(R_LCFS[Z_LCFS >= 0], Z_LCFS[Z_LCFS >= 0], "-k", linewidth=2.5)
    #
    for i in range(0, surf_max - surf_min, 80):
        plt.plot(R[i], Z[i], "-b", linewidth=1, alpha=0.2)
    plt.xlim([np.min(R_LCFS) - 0.02, np.max(R_LCFS) + 0.02])
    plt.ylim([0 - 0.02, np.max(Z_LCFS) + 0.02])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_dir / "flux_surfs.png")
    plt.close()


if __name__ == "__main__":

    project_root_dir = Path(__file__).absolute().parents[1]
    input_dir = project_root_dir / "input_files"
    default_input_file = input_dir / "eikcoefs_final_input.txt"
    default_output_dir = project_root_dir / "output_files_vmec" / "profiles"

    # Get vmec filename from config file
    config = parse_input_file(default_input_file)
    vmec_filename = input_dir / f"{config['vmec_fname']}.nc"

    # Make output dir if it doesn't already exist
    default_output_dir.mkdir(parents=True, exist_ok=True)

    # Run main function
    plot_profiles(vmec_filename, default_output_dir)
