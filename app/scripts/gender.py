
import torch

from src.config import Configuration
from src.data import load_gender_data
from src.models import GenderModule
import pytorch_lightning as pl


def train_gender(CONFIG: Configuration):
    """
    Train a gender classification model using the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing training parameters.
    """
    # Ensure full determinism across all sources of randomness
    pl.seed_everything(CONFIG.seed, workers=True)

    train_loader, test_loader = load_gender_data(CONFIG)
    model = GenderModule(CONFIG, weights=torch.tensor([0.4, 0.6]))
    model.model.print_number_parameters()


    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=CONFIG.patience,
            verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            filename=f"gender-{CONFIG.model_type}-{{epoch:02d}}-{{val_acc:.4f}}",
            dirpath=CONFIG.MODELS_PATH,
        ),
    ]

    logger = pl.loggers.CSVLogger(
        save_dir=CONFIG.LOGS_PATH,
        name=CONFIG.exp_name,
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        deterministic="warn",  # Use "warn" instead of True to allow non-deterministic algorithms when necessary
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


