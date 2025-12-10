import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Optional

class FactorizationMachine(pl.LightningModule):
    """
    Factorization Machine implemented as a PyTorch Lightning Module.
    Supports both Binary Classification and Regression.
    """
    def __init__(self, 
                 n_numeric_features: int, 
                 categorical_field_dims: List[int], 
                 embed_dim: int, 
                 item_field_idx: int, 
                 target_type: str = "binary", # 'binary' or 'continuous'
                 dropout_rate: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 initial_bias: float = 0.0,
                 pos_weight: float = 1.0  # For imbalanced binary classification
                 ):
        super().__init__()
        self.save_hyperparameters() # Auto-logs params to MLflow/Tensorboard

        self.n_numeric_features = n_numeric_features
        self.categorical_field_dims = categorical_field_dims
        self.target_type = target_type.lower()
        self.item_field_idx = item_field_idx
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # --- Model Architecture ---
        # 1. Linear Part
        self.embeddings = nn.ModuleList([
            nn.Embedding(num, 1) for num in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.linear_numeric = nn.Linear(self.n_numeric_features, 1)

        # 2. Interaction Part
        self.interaction_embeddings = nn.ModuleList([
            nn.Embedding(num, embed_dim) for num in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.interaction_numeric_vectors = nn.Parameter(torch.randn(n_numeric_features, embed_dim))

        self.bias = nn.Parameter(torch.tensor([initial_bias]))
        self.dropout = nn.Dropout(dropout_rate)

        # --- Loss Function Setup ---
        if self.target_type == "binary":
            # Handle class imbalance if pos_weight != 1.0
            p_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
            self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=p_weight)
        elif self.target_type == "continuous":
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def forward(self, x_numeric, x_categorical):
        # --- Linear Terms ---
        linear_terms = self.bias
        cat_linear_terms = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        linear_terms = linear_terms + torch.sum(torch.cat(cat_linear_terms, dim=1), dim=1, keepdim=True)
        
        if self.n_numeric_features > 0:
            linear_terms = linear_terms + self.linear_numeric(x_numeric)

        # --- Interaction Terms ---
        cat_interaction_vectors = [emb(x_categorical[:, i]) for i, emb in enumerate(self.interaction_embeddings)]
        stacked_cat_vector = torch.stack(cat_interaction_vectors, dim=1)
        
        if self.n_numeric_features > 0:
            numeric_interaction_vectors = x_numeric.unsqueeze(2) * self.interaction_numeric_vectors.unsqueeze(0)
            all_vectors = torch.cat([stacked_cat_vector, numeric_interaction_vectors], dim=1)
        else:
            all_vectors = stacked_cat_vector
            
        all_vectors = self.dropout(all_vectors)
        
        # FM Interaction equation: 0.5 * (sum(v)^2 - sum(v^2))
        sum_of_squares = torch.sum(all_vectors, dim=1).pow(2)
        square_of_sums = torch.sum(all_vectors.pow(2), dim=1)
        interaction_terms = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        logits = linear_terms + interaction_terms
        return logits.squeeze(1)

    def _shared_step(self, batch, stage):
        x_num, x_cat, y, weights = batch
        y_hat = self(x_num, x_cat)
        
        # Calculate raw loss
        # Note: BCEWithLogitsLoss expects raw logits, not sigmoid output
        if self.target_type == "binary":
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y)

        # Apply Inverse Propensity Weights (IPW)
        weighted_loss = (loss * weights).mean()
        
        self.log(f"{stage}_loss", weighted_loss, prog_bar=True, on_step=False, on_epoch=True)
        return weighted_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        # Optional: Add LR Scheduler here if needed
        return optimizer

    # Keep utility methods on the class
    def handle_new_items(self, new_item_idx, strategy='cold', source_item_id=None):
        """Utility to handle cold/warm starts for new items"""
        idx = self.item_field_idx
        with torch.no_grad():
            if strategy == 'cold':
                nn.init.normal_(self.embeddings[idx].weight[new_item_idx], 0, 0.01)
                nn.init.normal_(self.interaction_embeddings[idx].weight[new_item_idx], 0, 0.01)
            elif strategy == 'warm_copy' and source_item_id is not None:
                self.embeddings[idx].weight[new_item_idx] = self.embeddings[idx].weight[source_item_id].clone()
                self.interaction_embeddings[idx].weight[new_item_idx] = self.interaction_embeddings[idx].weight[source_item_id].clone()

# ---------

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from model import FactorizationMachine

def objective(trial: optuna.trial.Trial, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              config: dict):
    
    # 1. Suggest Hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # 2. Initialize Model
    model = FactorizationMachine(
        n_numeric_features=config['n_numeric'],
        categorical_field_dims=config['cat_dims'],
        embed_dim=embed_dim,
        item_field_idx=config['item_idx'],
        target_type=config['target_type'],
        dropout_rate=dropout_rate,
        learning_rate=lr,
        weight_decay=weight_decay,
        initial_bias=config['initial_bias'],
        pos_weight=config.get('pos_weight', 1.0)
    )

    # 3. Setup Early Stopping via Pruning
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # 4. Initialize Trainer with Speed Constraints
    # We disable checkpointing and logging during tuning to save I/O time
    trainer = pl.Trainer(
        logger=False, 
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="auto", # Automatically detects GPU/CPU
        devices="auto",
        callbacks=[pruning_callback],
        
        # --- Speed Up Logic ---
        # Only use 20% of training data and 20% of validation data per epoch
        # to get a rough estimate of performance quickly.
        limit_train_batches=0.2, 
        limit_val_batches=0.2,
        enable_progress_bar=False 
    )

    # 5. Train
    trainer.fit(model, train_loader, val_loader)

    # 6. Retrieve metric
    return trainer.callback_metrics["val_loss"].item()

def tune_hyperparameters(n_trials, train_loader, val_loader, config):
    print(f"--- Starting Tuning ({n_trials} trials) ---")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, config), 
        n_trials=n_trials
    )

    print(f"Best params: {study.best_params}")
    return study.best_params

# ---------------

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl

# Imports from your refactored files
from factorization_datasets import PandasDataset
from model import FactorizationMachine
from tuning import tune_hyperparameters

# ... Load your data into 'df' here ...

# 1. Setup Configuration
# Assuming you have calculated categorical dims and such from the df
config = {
    'n_numeric': len(numerical_cols),
    'cat_dims': [df[c].nunique() for c in categorical_cols],
    'item_idx': 2, # example index
    'target_type': 'binary',
    'initial_bias': -2.5, # Calculate using your logit helper
    'pos_weight': 1.0
}

# 2. Create DataLoaders
train_dataset = PandasDataset(train_df, categorical_cols, numerical_cols, 'clicked', 'banner_id', 'ipw')
val_dataset = PandasDataset(val_df, categorical_cols, numerical_cols, 'clicked', 'banner_id', 'ipw')

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)

# 3. Hyperparameter Tuning (Fast mode)
best_params = tune_hyperparameters(n_trials=20, train_loader=train_loader, val_loader=val_loader, config=config)

# 4. Train Final Model with Best Params on Full Data
final_model = FactorizationMachine(
    n_numeric_features=config['n_numeric'],
    categorical_field_dims=config['cat_dims'],
    embed_dim=best_params['embed_dim'],
    item_field_idx=config['item_idx'],
    target_type=config['target_type'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['lr'],
    weight_decay=best_params['weight_decay'],
    initial_bias=config['initial_bias']
)

# 5. Define Callbacks and Logger for Production Training
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints',
    filename='fm-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

# Connect to MLflow
mlf_logger = MLFlowLogger(experiment_name="FM_Recommendation_Model", tracking_uri="databricks")

# 6. Final Trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu", # Use "auto" or "gpu"
    devices=1,         # Set to >1 for multi-GPU
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=mlf_logger,
    # Remove limit_batches for full training
)

trainer.fit(final_model, train_loader, val_loader)

print(f"Best model saved at: {checkpoint_callback.best_model_path}")
