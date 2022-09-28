#!/usr/bin/env python3
"""
"""

import os
import pickle
from pathlib import Path

from eikcoefs_final import vmec_to_bishop
from utils import parse_input_file


def main(
    vmec_filename: Path,
    surf_idx: int,
    norm_scheme: int,
    nperiod: int,
    save_GX: bool,
    save_GS2: bool,
    ball_scan: bool,
    foms: bool,
    output_dir: Path = Path("."),
):
    if not any([save_GX, save_GS2, ball_scan]):
        raise ValueError(
            "No output selected. "
            "Set at least one of 'save_GX', 'save_GS2', 'ball_scan'."
        )

    bishop_dict = vmec_to_bishop(
        vmec_filename,
        surf_idx=surf_idx,
        norm_scheme=norm_scheme,
        nperiod=nperiod,
    )

    # TODO replace with function call, don't save pickle
    # saving bishop dict only once
    if ball_scan == 1 or save_GS2 == 1 or foms == 1:
        dict_file = open("bishop_dict.pkl", "wb")
        pickle.dump(bishop_dict, dict_file)
        dict_file.close()

    # TODO replace with function call, don't save pickle
    if save_GS2 == 1:
        os.system("python3 bishoper_save_GS2.py bishop_dict.pkl")

    # TODO replace with function call, don't save pickle
    if save_GX == 1:
        os.system("python3 bishoper_save_GX.py bishop_dict.pkl")

    # TODO replace with function call, don't save pickle
    if ball_scan == 1:
        os.system("python3 bishoper_ball.py bishop_dict.pkl")


# ===========================
# main script
#
# As this is the __main__ file, we don't need 'if __name__ == "__main__"' here.
#
# Called with either:
# $ python3 -m VMEC2GK
# $ python3 __main__.py

# TODO use argparse to control behaviour

project_root_dir = Path(__file__).absolute().parents[1]
input_dir = project_root_dir / "input_files"
default_input_file = input_dir / "eikcoefs_final_input.txt"
default_output_dir = project_root_dir / "output_files"

# Get vmec filename from config file
config = parse_input_file(default_input_file)
vmec_filename = input_dir / f"{config['vmec_fname']}.nc"
surf_idx = config["surf_idx"]
norm_scheme = config["norm_scheme"]
nperiod = config["nperiod"]
save_GX = bool(config["want_to_save_gx"])
save_GS2 = bool(config["want_to_save_gs2"])
ball_scan = bool(config["want_to_ball_scan"])
foms = bool(config["want_foms"])

# Make output dir if it doesn't already exist
default_output_dir.mkdir(parents=True, exist_ok=True)

# Run main function
main(
    vmec_filename,
    surf_idx=surf_idx,
    norm_scheme=norm_scheme,
    nperiod=nperiod,
    save_GX=save_GX,
    save_GS2=save_GS2,
    ball_scan=ball_scan,
    foms=foms,
    output_dir=default_output_dir,
)
