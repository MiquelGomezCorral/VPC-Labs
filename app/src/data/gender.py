import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Configuration



def load_gender_data(CONFIG: Configuration, use_transforms=True):
    """
    Load and preprocess the gender classification dataset based on the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing data loading parameters.
        def load_gender_data(CONFIG: Configuration, use_transforms=True):
 (bool): Whether to apply data augmentations (default: True).
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
            # transforms.RandomResizedCrop(
            #     (CONFIG.image_size, CONFIG.image_size),
            #     scale=(0.95, 1.0),
            #     ratio=(0.95, 1.05),
            #     antialias=True,
            # ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(
            #     brightness=0.1,
            #     saturation=0.1,
            #     hue=0.03,
            # ),
            transforms.RandomGrayscale(p=0.1),
            # transforms.RandomErasing(
            #     p=0.25,
            #     scale=(0.02, 0.12),
            #     ratio=(0.3, 3.3),
            #     value='random',
            #     inplace=False,
            # ),
            TopBiasedRandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value='random',
                inplace=False,
                top_bias_power=3.0,
            ),
            AddGaussianNoise(std=0.02),
        ]
    ) if use_transforms else transforms.Compose(
        [transforms.Resize((CONFIG.image_size, CONFIG.image_size), antialias=True)]
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
    

class AddGaussianNoise:
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp(0.0, 1.0)


class TopBiasedRandomErasing:
    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.12),
        ratio=(0.3, 3.3),
        value="random",
        inplace=False,
        top_bias_power=3.0,
        max_attempts=10,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        self.top_bias_power = top_bias_power
        self.max_attempts = max_attempts

    def _sample_fill(self, c, h, w, tensor):
        if self.value == "random":
            # return torch.empty((c, h, w), dtype=tensor.dtype, device=tensor.device).uniform_(0.0, 1.0)
            fill_value = torch.empty(1).uniform_(0.0, 1.0).item()
            return torch.full((c, h, w), fill_value, dtype=tensor.dtype, device=tensor.device)


        if isinstance(self.value, (tuple, list)):
            fill = torch.tensor(self.value, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
            return fill.expand(c, h, w)

        return torch.full((c, h, w), float(self.value), dtype=tensor.dtype, device=tensor.device)

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor

        if tensor.ndim != 3:
            return tensor

        c, img_h, img_w = tensor.shape
        area = img_h * img_w

        for _ in range(self.max_attempts):
            erase_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

            erase_h = int(round((erase_area * aspect_ratio) ** 0.5))
            erase_w = int(round((erase_area / aspect_ratio) ** 0.5))

            if erase_h <= 0 or erase_w <= 0 or erase_h >= img_h or erase_w >= img_w:
                continue

            max_top = img_h - erase_h
            max_left = img_w - erase_w

            # Bias the vertical position towards the top of the image.
            u = torch.rand(1).item()
            top = int((u ** self.top_bias_power) * (max_top + 1))
            left = int(torch.randint(0, max_left + 1, (1,)).item())

            out = tensor if self.inplace else tensor.clone()
            out[:, top : top + erase_h, left : left + erase_w] = self._sample_fill(c, erase_h, erase_w, out)
            return out

        return tensor
