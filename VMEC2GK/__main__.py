#!/usr/bin/env python3
"""
The main script, used to coordinate other tasks in the project.
"""

import pickle
from pathlib import Path

from .eikcoefs_final import vmec_to_bishop
from .bishoper_save_GX import save_gx
from .bishoper_save_GS2 import save_gs2
from .bishoper_ball import plot_ballooning_scan
from .utils import parse_input_file


def main(
    vmec_filename: Path,
    surf_idx: int,
    norm_scheme: int,
    nperiod: int,
    save_GX: bool,
    save_GS2: bool,
    ball_scan: bool,
    pickle_bishop_dict: bool,
    foms: bool,
    output_dir: Path = Path("."),
):
    if not any([save_GX, save_GS2, ball_scan]):
        raise RuntimeError(
            "No output selected. "
            "Set at least one of 'save_GX', 'save_GS2', 'ball_scan'."
        )

    # Ensure output_dir is of type Path, and ensure it exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bishop_dict = vmec_to_bishop(
        vmec_filename,
        surf_idx=surf_idx,
        norm_scheme=norm_scheme,
        nperiod=nperiod,
    )

    if pickle_bishop_dict:
        with open("bishop_dict.pkl", "wb") as dict_file:
            pickle.dump(bishop_dict, dict_file)

    if save_GS2 == 1:
        gs2_output_dir = output_dir / "GS2_grid_files"
        gs2_output_dir.mkdir(exist_ok=True)
        save_gs2(bishop_dict, gs2_output_dir)

    if save_GX == 1:
        gx_output_dir = output_dir / "GX_nc_files"
        gx_output_dir.mkdir(exist_ok=True)
        save_gx(bishop_dict, gx_output_dir)

    if ball_scan == 1:
        ball_scan_output_dir = output_dir / "s-alpha-plots"
        ball_scan_output_dir.mkdir(exist_ok=True)
        plot_ballooning_scan(bishop_dict, ball_scan_output_dir)


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
save_GX = bool(config.get("want_to_save_gx", False))
save_GS2 = bool(config.get("want_to_save_gs2", False))
ball_scan = bool(config.get("want_to_ball_scan", False))
foms = bool(config.get("want_foms", False))
pickle_bishop_dict = bool(config.get("want_to_pickle", False))

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
    pickle_bishop_dict=pickle_bishop_dict,
    output_dir=default_output_dir,
)
