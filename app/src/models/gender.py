import torch 
from torch import nn
import pytorch_lightning as pl

from src.config import Configuration


# Architecture SpecificationsInput Layer: 
# $32\times32$ pixel RGB images.
# Convolutional Block 1: 
#   Two consecutive convolutional layers with 32 filters each, using a $3\times3$ spatial dimension.
#   Followed by a $2\times2$ Max-Pooling layer.
# 
# Convolutional Block 2: 
#   Two consecutive convolutional layers with 64 filters each, using a $3\times3$ spatial dimension.
#   Followed by a $2\times2$ Max-Pooling layer.
# 
# Fully-Connected (FC) Layer: A single layer containing 16 neurons.
# 
# Output Layer: A Softmax layer with 2 neurons for binary gender classification.
# 
# Technical Parameters
#   Activation Function: Rectified Linear Units (ReLU) are used in all layers.
#   Regularization: Dropout is applied to both convolutional and fully-connected activations to prevent overfitting.
#   Optimization Gain: This specific architecture is 16 times more memory efficient than the initial Starting CNN while maintaining comparable accuracy.

class GenderCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=2):
        super(GenderCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)  # Output logits for binary classification
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # x = self.pool(x)
        x = self.fc(x)
        return x


    def print_number_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        model_mb = model_bytes / (1024 ** 2)

        print(f"Model parameters (trainable/total): {trainable_params:,}/{total_params:,}")
        print(f"Approx. model size (parameters only): {model_mb:.3f} MB")


class GenderModule(pl.LightningModule):
    def __init__(self, CONFIG: Configuration):
        super().__init__()
        # self.save_hyperparameters(ignore=["CONFIG"])
        self.CONFIG = CONFIG
        self.model = GenderCNN(dropout_rate=CONFIG.dropout_rate, num_classes=CONFIG.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.CONFIG.learning_rate,
        #     momentum=self.CONFIG.momentum,
        #     weight_decay=self.CONFIG.weight_decay,
        #     nesterov=True,
        # )
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.CONFIG.learning_rate,
            weight_decay=self.CONFIG.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.CONFIG.epochs,      # restart every N epochs
            eta_min=self.CONFIG.eta_min, # minimum lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }