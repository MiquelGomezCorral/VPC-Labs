
from src.config import Configuration
from src.data import load_gender_data
from src.models import GENDER_CNN
import pytorch_lightning as pl
import torch


class GenderLightningModule(pl.LightningModule):
    def __init__(self, config: Configuration):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = GENDER_CNN(dropout_rate=config.dropout_rate, num_classes=config.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_reduce_factor,
            patience=self.config.lr_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def train_gender(CONFIG: Configuration):
    """
    Train a gender classification model using the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing training parameters.
    """
    pl.seed_everything(CONFIG.seed, workers=True)

    train_loader, test_loader = load_gender_data(CONFIG)
    model = GenderLightningModule(CONFIG)

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=CONFIG.early_stopping_patience,
            verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            filename="gender-best-{epoch:02d}-{val_loss:.4f}",
            dirpath=CONFIG.MODELS_PATH,
        ),
    ]

    logger = pl.loggers.CSVLogger(
        save_dir=CONFIG.LOGS_PATH,
        name=CONFIG.exp_name,
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG.num_epochs,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    