"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import train_gender, train_car
from src.config import Configuration


def cmd_gender(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START GENDER TRAINING", sep_type="START")
    train_gender(CONFIG)
    print_separator("END GENDER TRAINING", sep_type="START")
    

def cmd_car(args: argparse.Namespace):
    """Call train_car with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START CAR TRAINING", sep_type="START")
    train_car(CONFIG)
    print_separator("END CAR TRAINING", sep_type="START")

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       read_extract
    # ======================================================================================
    p_gender = subparsers.add_parser("gender", help="Gender classification script")
    p_gender.set_defaults(func=cmd_gender)

    # ======================================================================================
    #                                       car
    # ======================================================================================
    p_car = subparsers.add_parser("car", help="Car classification script")
    p_car.set_defaults(func=cmd_car)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)