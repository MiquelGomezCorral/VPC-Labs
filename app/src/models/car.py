import torch 
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models

from src.config import Configuration
from maikol_utils.print_utils import print_separator

class BilinearCarCNN(nn.Module):
    def __init__(self, CONFIG: Configuration):
        super().__init__()
        self.num_classes = CONFIG.num_classes
        
        if CONFIG.model_type == "vg-res":
            self.symmetric = False
            self.model1 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model2 = nn.Sequential(*list(resnet.children())[:-2])
        elif CONFIG.model_type == "vg-vg":
            self.symmetric = True
            self.model1 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            self.model2 = None
        elif CONFIG.model_type == "res-res":
            self.symmetric = True
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model1 = nn.Sequential(*list(resnet.children())[:-2])
            self.model2 = None
        else:
            raise ValueError(f"Invalid model type: {CONFIG.model_type}. Must be one of ['vg-res', 'vg-vg', 'res-res']")
        # resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # self.model2 = nn.Sequential(*list(resnet.children())[:-2])

        # Dynamically determine the number of features after the convolutional layers
        c1 = [m for m in self.model1.modules() if isinstance(m, nn.Conv2d)][-1].out_channels
        c2 = [m for m in self.model2.modules() if isinstance(m, nn.Conv2d)][-1].out_channels if self.model2 is not None else c1
        self.num_features = c1 * c2 

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.num_features, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(CONFIG.dropout_rate),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(CONFIG.dropout_rate),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Dropout(CONFIG.dropout_rate),
        #     nn.Linear(512, CONFIG.num_classes)  
        # )

    def forward(self, x):
        out1 = self.model1(x) 
        out2 = self.model2(x) if not self.symmetric else out1

        features = self.outer_product([out1, out2])
        return self.classifier(features)

    def outer_product(self, x):
        # 'bchw,bdhw->bcd' handles the outer product across spatial dimensions
        phi_I = torch.einsum('bchw,bdhw->bcd', x[0], x[1])
        phi_I = phi_I.view(phi_I.size(0), -1)
        
        # Divide by spatial area (H * W) dynamically
        phi_I = phi_I / (x[0].size(2) * x[0].size(3))
        
        y_ssqrt = torch.sign(phi_I) * torch.sqrt(torch.abs(phi_I) + 1e-12)
        return F.normalize(y_ssqrt, p=2, dim=1)
    
    def print_number_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_mb = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)

        # Breakdowns to verify freezing/unfreezing stages
        m1_trainable = sum(p.numel() for p in self.model1.parameters() if p.requires_grad)
        cls_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)

        print_separator("Parameter Breakdown")
        print(f" - Total Parameters: {total_params:,}")
        print(f" - Total Trainable: {trainable_params:,}")
        
        if self.symmetric:
            print(f"   - Backbone Trainable (Symmetric): {m1_trainable:,}")
        else:
            m2_trainable = sum(p.numel() for p in self.model2.parameters() if p.requires_grad)
            print(f"   - Backbone 1 (VGG) Trainable: {m1_trainable:,}")
            print(f"   - Backbone 2 (ResNet) Trainable: {m2_trainable:,}")
            
        print(f"   - Classifier Trainable: {cls_trainable:,}")
        print(f" - Approx Size: {model_mb:.2f} MB")
        print_separator()



class CarModule(pl.LightningModule):
    def __init__(self, CONFIG: Configuration, weights=None):
        super().__init__()
        self.CONFIG = CONFIG
        self.model = BilinearCarCNN(CONFIG)

        if weights is None:
            self.criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=CONFIG.label_smoothing 
            )
        else: 
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=weights, 
                label_smoothing=CONFIG.label_smoothing 
            )

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
    def is_symmetric(self):
        return self.model.symmetric