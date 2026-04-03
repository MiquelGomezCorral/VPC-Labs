import torch
import pytorch_lightning as pl

from maikol_utils.print_utils import print_separator

from src.data import load_car_data
from src.models import CarModule
from src.config import Configuration


def train_car(CONFIG: Configuration):
    """
    Train a car classification model using the provided configuration.
    Args:
        CONFIG (Configuration): The configuration object containing training parameters.
    """
    pl.seed_everything(CONFIG.seed, workers=True)

    train_loader, test_loader = load_car_data(CONFIG)

    model = CarModule(CONFIG)#, weights=torch.tensor([0.4, 0.6]))
    model.model.print_number_parameters()


    # ========================= STAGE 1: Train Classifier Only =========================
    print_separator("STAGE 1: Train Classifier Only", sep_type="SUPER")
    # Freeze backbones
    for param in model.model.model1.parameters():
        param.requires_grad = False
    if not model.is_symmetric():
        for param in model.model.model2.parameters():
            param.requires_grad = False

    callbacks_s1 = [
        pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=CONFIG.patience, verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max", save_top_k=1, save_weights_only=True, 
            filename=f"stage1-{CONFIG.model_type}-{{epoch:02d}}-{{val_acc:.4f}}", 
            dirpath=CONFIG.car_models
        ),
    ]

    logger = pl.loggers.CSVLogger(save_dir=CONFIG.LOGS_PATH, name=CONFIG.exp_name)

    trainer_s1 = pl.Trainer(
        max_epochs=CONFIG.epochs, callbacks=callbacks_s1, logger=logger,
        check_val_every_n_epoch=1, log_every_n_steps=1, deterministic="warn"
    )
    trainer_s1.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # ========================= STAGE 2: Fine-tune Everything =========================
    print_separator("STAGE 2: Fine-tune Everything", sep_type="SUPER")
    # Unfreeze backbones
    for param in model.model.model1.parameters():
        param.requires_grad = True
    if not model.is_symmetric():
        for param in model.model.model2.parameters():
            param.requires_grad = True

    # Drop LR by 2 orders of magnitude to prevent big updates
    CONFIG.learning_rate *= 0.01  

    
    callbacks_s2 = [
        pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=CONFIG.patience, verbose=True),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc", mode="max", save_top_k=1, save_weights_only=True, 
            filename=f"stage2-{CONFIG.model_type}-{{epoch:02d}}-{{val_acc:.4f}}", 
            dirpath=CONFIG.car_models
        ),
    ]

    trainer_s2 = pl.Trainer(
        max_epochs=CONFIG.epochs, callbacks=callbacks_s2, logger=logger,
        check_val_every_n_epoch=1, log_every_n_steps=1, deterministic="warn"
    )
    trainer_s2.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)