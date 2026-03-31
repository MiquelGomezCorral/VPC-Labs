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
    parser.add_argument("-n", "--exp_name", type=str, default="base_name", help="Experiment name (default: 'base_name')")
    parser.add_argument("-d", "--exp_description", type=str, default="Base experiment description", help="Experiment description (default: 'Base experiment description')")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       read_extract
    # ======================================================================================
    p_gender = subparsers.add_parser("gender", help="Gender classification script")
    p_gender.add_argument("-mt", "--model_type", choices=["small", "large"], default="small", help="Model type (default: 'small', options: 'small', 'large')")
    p_gender.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size (default: 512)")
    p_gender.add_argument("-ep", "--num_epochs", type=int, default=100, help="Number of epochs (default: 100)")
    p_gender.add_argument("-dr", "--dropout_rate", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    p_gender.add_argument("-ls", "--label_smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")
    p_gender.add_argument("-lr", "--learning_rate", type=float, default=5e-3, help="Learning rate (default: 0.01)")
    p_gender.add_argument("-lp", "--lr_patience", type=int, default=3, help="Learning rate scheduler patience (default: 3)")
    p_gender.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    p_gender.add_argument("-em", "--eta_min", type=float, default=1e-6, help="Minimum learning rate for scheduler (default: 1e-6)")
    p_gender.add_argument("-pa", "--patience", type=int, default=20, help="Early stopping patience (default: 5)")
    p_gender.add_argument("-mo", "--momentum", type=float, default=0.9, help="Momentum (default: 0.9)")
    p_gender.add_argument("-lrf", "--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for scheduler (default: 0.5)")

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