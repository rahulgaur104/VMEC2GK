#!/usr/bin/env python3
"""
The main script, used to coordinate other tasks in the project.
"""

import pickle
from pathlib import Path
from argparse import ArgumentParser
from textwrap import dedent

from VMEC2GK.eikcoefs_final import vmec_to_bishop
from VMEC2GK.bishoper_save_GX import bishop_to_gx
from VMEC2GK.bishoper_save_GS2 import bishop_to_gs2
from VMEC2GK.bishoper_ball import plot_ballooning_scan
from VMEC2GK.utils import parse_input_file


def main(
    vmec_filename: Path,
    surf_idx: int,
    norm_scheme: int,
    nperiod: int,
    save_gx: bool,
    save_gs2: bool,
    ball_scan: bool,
    pickle_bishop_dict: bool,
    foms: bool,
    output_dir: Path = Path("."),
):
    if not any([save_gx, save_gs2, ball_scan]):
        raise RuntimeError(
            "No output selected. "
            "Set at least one of 'save_gx', 'save_gs2', 'ball_scan'."
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

    if save_gs2 == 1:
        gs2_output_dir = output_dir / "GS2_grid_files"
        gs2_output_dir.mkdir(exist_ok=True)
        bishop_to_gs2(bishop_dict, gs2_output_dir)

    if save_gx == 1:
        gx_output_dir = output_dir / "GX_nc_files"
        gx_output_dir.mkdir(exist_ok=True)
        bishop_to_gx(bishop_dict, gx_output_dir)

    if ball_scan == 1:
        ball_scan_output_dir = output_dir / "s-alpha-plots"
        ball_scan_output_dir.mkdir(exist_ok=True)
        plot_ballooning_scan(bishop_dict, ball_scan_output_dir)


def parse_cmd_line_args():
    """
    Reads command line arguments and controls the output of the program.

    Users can provide an input file using '-i' or '--input_file', which defines all
    other arguments. Arguments given on the command line take precedence.

    For info on how to run the program, try:

    $ python3 -m VMEC2GK --help
    """

    # TODO Need to rethink the interaction between:
    # - default cmd line args (all None or False currently)
    # - default input file (should this even feature?)
    # - User provided cmd line args
    # - User provided input file
    # One issue is that the output dir and vmec file and defined relative to the
    # input file, meaning any users will find their files being written somewhere
    # inside the Python project. Not good if pip installed from a remote server!

    project_description = dedent(
        """\
        VMEC equilibrium to gyrokinetics.
         
        This program takes a VMEC equilibrium file for an axisymmetric equilibrium and
        creates the geometric coefficient files required for a gyrokinetics run with
        GS2 or GX.
        """
    )

    # Define command line args
    parser = ArgumentParser(description=project_description)

    # Input file
    parser.add_argument(
        "-i",
        "--input_file",
        type=Path,
        help="VMEC2GK input file, see 'input_files/eikcoefs_final_input.txt'.",
    )

    # VMEC file
    parser.add_argument(
        "-v", "--vmec_file", type=Path, help="VMEC equilibrium file to process."
    )

    # Eq options
    parser.add_argument(
        "-s",
        "--surf_idx",
        type=int,
        help=dedent(
            """\
            Which flux surface to process. This is set via an integer value, with 0
            being at the magnetic axis and the maximum (detemined by the VMEC file)
            being the 'Last Closed Flux Surface' (LCFS).
            """
        ),
    )

    parser.add_argument(
        "-n",
        "--norm_scheme",
        type=int,
        help=dedent(
            """\
            Sets length normalisation. If set to 1, a_N is effective minor radius of the
            LCFS, such that pi*a_N**2 = area enclosed by the LCFS, and 
            B_N = Phi_LCFS/(pi*a_N**2). If set to 2, a_N is the minor radius of the
            LCFS, such that B_N = F/a_N where F is poloidal current on the local
            equilibrium surface.
            """
        ),
    )

    parser.add_argument(
        "-p",
        "--nperiod",
        type=int,
        help="The number of times the flux tube goes around is given by 2*nperiod - 1.",
    )

    # Output options
    parser.add_argument("--save_gs2", action="store_true", help="Save GS2 grid file")

    parser.add_argument("--save_gx", action="store_true", help="Save GX netCDF4 file")

    parser.add_argument(
        "--ball_scan",
        action="store_true",
        help="Perform s-alpha ballooning stability analysis.",
    )

    parser.add_argument("--foms", action="store_true", help="Currently not in use")

    parser.add_argument(
        "--pickle", action="store_true", help="Save intermediate data as a pickle file"
    )

    # Get command line args
    cmd_line_args = parser.parse_args()

    # If the user provided an input file, read that into results. If not provided, fall
    # back on defaults
    input_file = cmd_line_args.input_file
    if input_file is None:
        project_root_dir = Path(__file__).absolute().parents[1]
        input_file = project_root_dir / "input_files" / "eikcoefs_final_input.txt"

    args = parse_input_file(input_file)

    # Overwrite input file with cmd line args
    for key, value in vars(cmd_line_args).items():
        if value:
            args[key] = value

    return args


# ===========================
# main script
#
# As this is the __main__ file, we don't need 'if __name__ == "__main__"' here.
#
# Called with either:
# $ python3 -m VMEC2GK
# $ python3 __main__.py

args = parse_cmd_line_args()
main(
    args["vmec_file"],
    surf_idx=args["surf_idx"],
    norm_scheme=args["norm_scheme"],
    nperiod=args["nperiod"],
    save_gx=args["save_gx"],
    save_gs2=args["save_gs2"],
    ball_scan=args["ball_scan"],
    foms=args["foms"],
    pickle_bishop_dict=args["pickle"],
    output_dir=args["output_dir"],
)
