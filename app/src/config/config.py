"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

import torch
from maikol_utils.print_utils import print_separator
from maikol_utils.file_utils import make_dirs

@dataclass 
class Configuration:
    """Configuration class for the project."""
    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    
    gender_data: str = os.path.join(DATA_PATH, "gender")
    gender_x_train: str = os.path.join(gender_data, "x_train.npy")
    gender_x_test: str = os.path.join(gender_data, "x_test.npy")
    gender_y_train: str = os.path.join(gender_data, "y_train.npy")
    gender_y_test: str = os.path.join(gender_data, "y_test.npy")

    car_data: str = os.path.join(DATA_PATH, "car")
    car_x_train: str = os.path.join(car_data, "x_train.npy")
    car_x_test: str = os.path.join(car_data, "x_test.npy")
    car_y_train: str = os.path.join(car_data, "y_train.npy")
    car_y_test: str = os.path.join(car_data, "y_test.npy")


    exp_name: str = "base_name"
    exp_description: str = "Base experiment description"
    seed:     int = 42
    image_size: int = 100
    num_classes: int = 2

    batch_size: int = 128
    epochs: int = 100

    dropout_rate: float = 0.5
    label_smoothing: float = 0.1

    learning_rate: float = 0.01
    weight_decay: float = 1e-4

    eta_min: float = 1e-6

    momentum: float = 0.9
    
    lr_reduce_factor: float = 0.5
    lr_patience: int = 3

    patience: int = 10

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        make_dirs([self.MODELS_PATH, self.LOGS_PATH])

        print_separator("CONFIGURATION", sep_type="LONG")

        print(
            f"Experiment description: {self.exp_description}\n"
            f"Experiment name: {self.exp_name}\n"
            f"seed: {self.seed}\n"
            f"batch_size: {self.batch_size}\n"
            f"epochs: {self.epochs}\n"
            f"dropout_rate: {self.dropout_rate}\n"
            f"label_smoothing: {self.label_smoothing}\n"
            f"learning_rate: {self.learning_rate}\n"
            f"weight_decay: {self.weight_decay}\n"
            f"eta_min: {self.eta_min}\n"
            f"momentum: {self.momentum}\n"
            f"lr_reduce_factor: {self.lr_reduce_factor}\n"
            f"lr_patience: {self.lr_patience}\n"
            f"patience: {self.patience}\n"
        )
