"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

@dataclass 
class Configuration:
    """Configuration class for the project."""
    DATA_PATH: str = os.path.join("..", "data")
    
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
    seed:     int = 42




    def __post_init__(self):
        ...
