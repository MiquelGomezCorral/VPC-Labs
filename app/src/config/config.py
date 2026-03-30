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
    seed:     int = 42
    batch_size: int = 128
    bach_size: int = 128
    num_epochs: int = 100
    image_size: int = 128

    dropout_rate: float = 0.5
    num_classes: int = 2
    label_smoothing: float = 0.1

    learning_rate: float = 0.01
    momentum: float = 0.9
    lr_reduce_factor: float = 0.5
    lr_patience: int = 2
    weight_decay: float = 1e-4

    early_stopping_patience: int = 6



    def __post_init__(self):
        # Backward compatibility for typoed field name.
        self.batch_size = int(getattr(self, "batch_size", self.bach_size))
