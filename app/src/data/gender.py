import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Configuration

def load_gender_data(CONFIG: Configuration):
    """
    Load and preprocess the gender classification dataset based on the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing data loading parameters.
    Returns:
        Tuple: A tuple containing the training and testing data (x_train, x_test, y_train, y_test).
    """
    # ============== Load data ============== 
    x_train = np.load(CONFIG.gender_x_train)
    x_test = np.load(CONFIG.gender_x_test)

    y_train = np.load(CONFIG.gender_y_train)
    y_test = np.load(CONFIG.gender_y_test)
    
    # ============== Define transforms ==============
    train_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG.image_size, CONFIG.image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((CONFIG.image_size, CONFIG.image_size), antialias=True),
        ]
    )

    # ============== Create datasets dataloaders ==============
    train_dataset = GenderDataset(x_train, y_train, transform=train_transform)
    test_dataset  = GenderDataset(x_test,  y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG.batch_size, shuffle=False)
    
    return train_loader, test_loader


# ========================================================
#                     Dataset
# ========================================================

class GenderDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        if self.x.max() > 1.0: # si están en el rango [0, 255], normalizamos a [0, 1]
            self.x = self.x / 255.0
        self.x = self.x.clamp(0.0, 1.0)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]

        # Handle both HWC and CHW inputs.
        if image.ndim == 3 and image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        if self.transform is not None:
            image = self.transform(image)


        return image, self.y[idx]