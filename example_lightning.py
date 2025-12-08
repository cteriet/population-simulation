# schema.py

from dataclasses import dataclass, field
from typing import List, Optional

class FeatureType:
    CUSTOMER = "customer"
    ITEM = "item"
    CONTEXT = "context"

@dataclass
class FeatureConfig:
    name: str
    feature_type: str  # 'customer', 'item', 'context'
    data_type: str     # 'categorical', 'numerical'
    cardinality: int = 0  # For categorical features
    
@dataclass
class ModelConfig:
    features: List[FeatureConfig]
    embed_dim: int
    target_col: str
    weight_col: Optional[str] = None
    learning_rate: float = 1e-3
    dropout_rate: float = 0.1
    
    @property
    def numerical_features(self) -> List[str]:
        return [f.name for f in self.features if f.data_type == 'numerical']
    
    @property
    def categorical_features(self) -> List[str]:
        return [f.name for f in self.features if f.data_type == 'categorical']
        
    @property
    def customer_feature_indices(self) -> List[int]:
        """Returns indices of categorical features belonging to customer."""
        return [i for i, f in enumerate(self.features) 
                if f.data_type == 'categorical' and f.feature_type == FeatureType.CUSTOMER]

    @property
    def item_feature_indices(self) -> List[int]:
        """Returns indices of categorical features belonging to items."""
        return [i for i, f in enumerate(self.features) 
                if f.data_type == 'categorical' and f.feature_type == FeatureType.ITEM]
                
                
# data.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from streaming import StreamingDataset as MosaicStreamingDataset
from typing import Union, Optional
from .schema import ModelConfig

class PandasDataset(Dataset):
    """Memory-based dataset for testing/sampling."""
    def __init__(self, df: pd.DataFrame, config: ModelConfig):
        self.config = config
        self.df = df
        
        # Pre-convert to numpy for speed
        self.x_cat = df[config.categorical_features].values.astype(np.int64)
        self.x_num = df[config.numerical_features].values.astype(np.float32) if config.numerical_features else np.zeros((len(df), 0))
        self.y = df[config.target_col].values.astype(np.float32)
        
        if config.weight_col:
            self.weights = df[config.weight_col].values.astype(np.float32)
        else:
            self.weights = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32)
        }

class StreamingSparkDataset(MosaicStreamingDataset):
    """
    MosaicML Streaming Dataset. 
    Expects data to be present in MDS format at `remote` location.
    """
    def __init__(self, remote: str, local: str, config: ModelConfig, split: str = None):
        super().__init__(local=local, remote=remote, split=split, shuffle=True)
        self.config = config

    def __getitem__(self, idx):
        # streaming returns a dict sample
        sample = super().__getitem__(idx)
        
        # Extract and format based on config names
        x_cat = [int(sample[col]) for col in self.config.categorical_features]
        x_num = [float(sample[col]) for col in self.config.numerical_features]
        y = float(sample[self.config.target_col])
        w = float(sample[self.config.weight_col]) if self.config.weight_col else 1.0

        return {
            "x_num": torch.tensor(x_num, dtype=torch.float32),
            "x_cat": torch.tensor(x_cat, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
            "weight": torch.tensor(w, dtype=torch.float32)
        }

class DataLoaderFactory:
    """Factory (Requirement 4) to generate loaders regardless of source."""
    
    @staticmethod
    def get_dataloader(
        source_type: str, 
        config: ModelConfig, 
        batch_size: int = 1024,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        
        dataset = None
        
        if source_type == "pandas":
            df = kwargs.get("df")
            if df is None: raise ValueError("df argument required for pandas source")
            dataset = PandasDataset(df, config)
            
        elif source_type == "spark_streaming":
            # Requires 'remote' (S3/ADLS path) and 'local' (temp path on cluster)
            remote = kwargs.get("remote")
            local = kwargs.get("local")
            dataset = StreamingSparkDataset(remote=remote, local=local, config=config)
            
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from .schema import ModelConfig

class FactorizationMachinePL(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters() # Logs params to MLflow automatically
        self.config = config
        
        # --- Metrics ---
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")

        # --- Embeddings (Linear & Interaction) ---
        # 1. Linear part (Bias is handled separately)
        self.embeddings = nn.ModuleList([
            nn.Embedding(f.cardinality, 1) for f in config.features if f.data_type == 'categorical'
        ])
        
        # 2. Interaction part
        self.interaction_embeddings = nn.ModuleList([
            nn.Embedding(f.cardinality, config.embed_dim) 
            for f in config.features if f.data_type == 'categorical'
        ])

        # 3. Numeric handling
        n_num = len(config.numerical_features)
        if n_num > 0:
            self.linear_numeric = nn.Linear(n_num, 1)
            self.interaction_numeric = nn.Parameter(torch.randn(n_num, config.embed_dim))
        
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Loss function with weight support
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x_num, x_cat):
        """
        Standard FM Forward pass.
        Returns: logits
        """
        # 1. Linear Terms
        # Categorical linear: sum(embedding(x))
        linear_terms = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        linear_sum = torch.sum(torch.cat(linear_terms, dim=1), dim=1, keepdim=True)
        
        # Numeric linear
        if len(self.config.numerical_features) > 0:
            linear_sum += self.linear_numeric(x_num)
            
        linear_total = self.bias + linear_sum

        # 2. Interaction Terms
        # Categorical interactions
        cat_interactions = [emb(x_cat[:, i]) for i, emb in enumerate(self.interaction_embeddings)]
        stacked_interactions = torch.stack(cat_interactions, dim=1) # (Batch, Fields, Embed)

        # Numeric interactions
        if len(self.config.numerical_features) > 0:
            # Broadcast numeric values to interaction vectors
            # x_num: (Batch, Num_Feats) -> (Batch, Num_Feats, 1)
            # param: (Num_Feats, Embed) -> (1, Num_Feats, Embed)
            num_interactions = x_num.unsqueeze(2) * self.interaction_numeric.unsqueeze(0)
            all_interactions = torch.cat([stacked_interactions, num_interactions], dim=1)
        else:
            all_interactions = stacked_interactions

        all_interactions = self.dropout(all_interactions)

        # FM Equation: 0.5 * ( sum(v)^2 - sum(v^2) )
        sum_of_squares = torch.sum(all_interactions, dim=1).pow(2)
        square_of_sums = torch.sum(all_interactions.pow(2), dim=1)
        interaction_total = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        return (linear_total + interaction_total).squeeze(1)

    def training_step(self, batch, batch_idx):
        x_num, x_cat, y, w = batch['x_num'], batch['x_cat'], batch['y'], batch['weight']
        logits = self(x_num, x_cat)
        
        # Apply sample weights
        loss = (self.criterion(logits, y) * w).mean()
        
        # Logging
        preds = torch.sigmoid(logits)
        self.train_auc(preds, y.int())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y, w = batch['x_num'], batch['x_cat'], batch['y'], batch['weight']
        logits = self(x_num, x_cat)
        loss = (self.criterion(logits, y) * w).mean()
        
        preds = torch.sigmoid(logits)
        self.val_auc(preds, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", self.val_auc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)

    # --- Encoding Methods (Requirement 5 & 6) ---
    # We construct vectors such that <V_cust, V_item> approximates the FM score.
    # Approach: 
    # V_cust = [ Sum(Cust_Embeddings), 1 ]
    # V_item = [ Sum(Item_Embeddings), Bias ]
    # Note: This ignores the "Interaction between customer features themselves" 
    # but captures "Customer <-> Item" interaction which is usually dominant for recommendation.

    def encode_customer(self, x_cat_customer):
        """
        Args:
            x_cat_customer: Tensor of shape (Batch, Num_Customer_Features) containing indices.
                            Must correspond to the indices in self.config.customer_feature_indices.
        Returns:
            Tensor of shape (Batch, Embed_Dim + 1)
        """
        indices = self.config.customer_feature_indices
        
        # Get interaction embeddings for customer fields
        vectors = []
        for local_idx, global_idx in enumerate(indices):
            emb_layer = self.interaction_embeddings[global_idx]
            vectors.append(emb_layer(x_cat_customer[:, local_idx]))
            
        # Sum them up
        v_sum = torch.sum(torch.stack(vectors, dim=1), dim=1)
        
        # Append a '1' to facilitate bias addition via dot product
        batch_size = x_cat_customer.size(0)
        ones = torch.ones((batch_size, 1), device=self.device)
        
        return torch.cat([v_sum, ones], dim=1)

    def encode_item(self, x_cat_item):
        """
        Args:
            x_cat_item: Tensor of shape (Batch, Num_Item_Features)
        Returns:
            Tensor of shape (Batch, Embed_Dim + 1)
        """
        indices = self.config.item_feature_indices
        
        vectors = []
        for local_idx, global_idx in enumerate(indices):
            emb_layer = self.interaction_embeddings[global_idx]
            vectors.append(emb_layer(x_cat_item[:, local_idx]))
            
        v_sum = torch.sum(torch.stack(vectors, dim=1), dim=1)
        
        # Append the global Bias to the item vector
        # Note: If you want Item Linear terms included, you would add them here too.
        # Ideally: Item_Vec = [v_sum, bias + item_linear_terms]
        # For simplicity, we just append Global Bias.
        batch_size = x_cat_item.size(0)
        bias_expanded = self.bias.expand(batch_size, 1)
        
        return torch.cat([v_sum, bias_expanded], dim=1)
        
# example

import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 1. Define Configuration
features = [
    FeatureConfig("customer_id", FeatureType.CUSTOMER, "categorical", cardinality=10000),
    FeatureConfig("customer_age_bucket", FeatureType.CUSTOMER, "categorical", cardinality=10),
    FeatureConfig("banner_id", FeatureType.ITEM, "categorical", cardinality=50),
    FeatureConfig("banner_size", FeatureType.ITEM, "categorical", cardinality=5),
    FeatureConfig("device_type", FeatureType.CONTEXT, "categorical", cardinality=3),
    # ... numerical features ...
]

config = ModelConfig(
    features=features,
    embed_dim=32,
    target_col="is_clicked",
    weight_col="ips_weight"
)

# 2. Create Model
model = FactorizationMachinePL(config)

# 3. Create Dataloaders (Factory Pattern)
# Option A: Pandas (Small scale / Local dev)
# train_loader = DataLoaderFactory.get_dataloader("pandas", config, df=train_df)

# Option B: Streaming (Databricks / Full Scale)
# Ensure you have run the Spark -> MDS Writer job first!
train_loader = DataLoaderFactory.get_dataloader(
    "spark_streaming", 
    config, 
    remote="s3://my-bucket/data/train_mds/", 
    local="/tmp/cache/train"
)

val_loader = DataLoaderFactory.get_dataloader(
    "spark_streaming", 
    config, 
    remote="s3://my-bucket/data/val_mds/", 
    local="/tmp/cache/val"
)

# 4. Train with Lightning & MLflow
mlflow.pytorch.autolog() # Auto-logs metrics, params, and models

checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
early_stop_cb = EarlyStopping(monitor="val_loss", patience=3)

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices="auto",
    callbacks=[checkpoint_cb, early_stop_cb],
    default_root_dir="/dbfs/mlflow_checkpoints", # Save checkpoints to DBFS
    enable_progress_bar=True
)

with mlflow.start_run() as run:
    trainer.fit(model, train_loader, val_loader)

# 5. Inference / Vector Generation
model.eval()
# Create dummy customer input (indices for customer_id, customer_age)
cust_indices = torch.tensor([[50, 2], [100, 5]], dtype=torch.long) # Batch of 2
cust_vectors = model.encode_customer(cust_indices) 
# cust_vectors is now compatible with a vector database

#tuning.py

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
import mlflow

def objective(trial: optuna.trial.Trial, base_config: ModelConfig, train_loader, val_loader):
    # 1. Suggest Hyperparameters
    # We create a copy of the config to avoid mutating the original object across trials
    trial_config = ModelConfig(
        features=base_config.features, # Keep schema same
        target_col=base_config.target_col,
        weight_col=base_config.weight_col,
        
        # Tunable parameters
        embed_dim=trial.suggest_categorical("embed_dim", [16, 32, 64, 128]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5)
    )

    # 2. Initialize Model
    model = FactorizationMachinePL(trial_config)

    # 3. Initialize Trainer
    # We use the PruningCallback to stop bad trials early
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    trainer = Trainer(
        max_epochs=5, # Keep epochs low for tuning
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False, # Don't save checkpoints for every trial to save disk space
        logger=False, # Optional: Don't log every trial to MLflow unless you want to
        callbacks=[pruning_callback],
        
        # CRITICAL for Streaming Data:
        # Don't iterate the whole infinite stream. 
        # Cap it to a reasonable amount of steps to determine convergence speed.
        limit_train_batches=5000, 
        limit_val_batches=500
    )

    # 4. Train
    trainer.fit(model, train_loader, val_loader)

    # 5. Return the metric to minimize
    # callback_metrics is a dict populated by self.log inside the LightningModule
    return trainer.callback_metrics["val_loss"].item()

# --- Execution ---
def run_tuning():
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    # We can reuse the same loaders if they are iterable/streaming
    # However, ensure your DataloaderFactory allows re-initialization if needed
    train_dl = get_dataloader("spark_streaming", config, ...)
    val_dl = get_dataloader("spark_streaming", config, ...)

    study.optimize(
        lambda trial: objective(trial, config, train_dl, val_dl), 
        n_trials=20
    )
    
    print("Best Params:", study.best_params)
    return study.best_params
    
# data_utils.py

def get_dataloader(
    source_type: str, 
    config: ModelConfig, 
    batch_size: int = 1024,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create dataloaders.
    
    Args:
        source_type: 'pandas' or 'spark_streaming'
        config: Model schema configuration
        **kwargs: Arguments specific to the source (e.g., 'df', 'remote', 'local')
    """
    dataset = None
    
    if source_type == "pandas":
        df = kwargs.get("df")
        if df is None: 
            raise ValueError("Argument 'df' is required for pandas source")
        dataset = PandasDataset(df, config)
        
    elif source_type == "spark_streaming":
        remote = kwargs.get("remote")
        local = kwargs.get("local")
        if not remote or not local:
             raise ValueError("Arguments 'remote' and 'local' required for streaming")
        
        dataset = StreamingSparkDataset(remote=remote, local=local, config=config)
        
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
# Spark side conversion to MDS

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml import Pipeline
from streaming.spark import convert_dataframe_to_mds
import shutil

# Assume 'config' is your ModelConfig object from the previous step
# Assume 'df' is your raw loaded Spark DataFrame

def prepare_and_write_data(spark: SparkSession, 
                           df, 
                           config, 
                           output_path: str, 
                           indexer_path: str):
    """
    1. Indexes categorical strings to integers.
    2. Casts numericals to float.
    3. Writes to MDS format for MosaicML Streaming.
    """
    
    # --- 1. Preprocessing: String Indexing ---
    # We must convert string identifiers to integers (0..N) for nn.Embedding
    # We use handleInvalid='keep' to map unseen items to a specific 'unknown' index
    
    indexers = []
    output_cols = []
    
    for feature in config.features:
        if feature.data_type == 'categorical':
            # Input: "customer_id", Output: "customer_id_idx"
            out_col = f"{feature.name}_idx"
            indexer = StringIndexer(
                inputCol=feature.name, 
                outputCol=out_col, 
                handleInvalid='keep' # Crucial for cold-start safety
            )
            indexers.append(indexer)
            output_cols.append(out_col)
        else:
            output_cols.append(feature.name)

    # Add target and weight to output
    output_cols.append(config.target_col)
    if config.weight_col:
        output_cols.append(config.weight_col)

    # Fit the pipeline (This effectively calculates the vocabulary for every column)
    pipeline = Pipeline(stages=indexers)
    pipeline_model = pipeline.fit(df)
    
    # SAVE the indexers! You need these later for inference/encoding new items
    # e.g. to know that "banner_abc" maps to index 54
    pipeline_model.write().overwrite().save(indexer_path)
    print(f"Indexer mappings saved to {indexer_path}")

    # Transform the data
    processed_df = pipeline_model.transform(df)

    # --- 2. Type Casting & Selection ---
    # Mosaic MDS needs explicit types. 
    # PyTorch typically wants float32 for dense, int64 for indices.
    
    final_selects = []
    mds_columns = {} # Dictionary mapping {col_name: mds_type_string}
    
    for feature in config.features:
        if feature.data_type == 'categorical':
            col_name = f"{feature.name}_idx"
            # PyTorch Embedding requires Long (Int64)
            final_selects.append(F.col(col_name).cast("long").alias(feature.name))
            mds_columns[feature.name] = 'int64'
        else:
            # Numerical features
            final_selects.append(F.col(feature.name).cast("float"))
            mds_columns[feature.name] = 'float32'
            
    # Target
    final_selects.append(F.col(config.target_col).cast("float"))
    mds_columns[config.target_col] = 'float32'
    
    # Weight
    if config.weight_col:
        final_selects.append(F.col(config.weight_col).cast("float"))
        mds_columns[config.weight_col] = 'float32'

    export_df = processed_df.select(final_selects)

    # --- 3. Write to MDS ---
    print(f"Writing MDS shards to {output_path}...")
    
    # Check if directory exists and cleanup (optional, be careful in prod)
    # dbutils.fs.rm(output_path, recurse=True) 
    
    # Use MosaicML's Spark helper
    # mds_kwargs can adjust compression (zstd, etc.)
    convert_dataframe_to_mds(
        dataframe=export_df,
        out=output_path,
        columns=mds_columns,
        partition_size=2048, # Adjust based on data size (target ~64-128MB per shard)
        keep_local=False,    # Delete local temp files after upload
        compression='zstd'
    )
    
    print("Write complete.")

# --- Usage Example ---
# Assuming you are in Databricks and have your ModelConfig defined
# output_s3 = "s3://my-datalake/training-data/v1/mds/"
# indexer_s3 = "s3://my-datalake/models/v1/indexers/"

# prepare_and_write_data(spark, raw_df, config, output_s3, indexer_s3)


# loading on node

# In your DataLoaderFactory / data_utils.py

def get_dataloader(source_type, config, ...):
    
    if source_type == "spark_streaming":
        # Remote: The permanent 400GB home (Databricks Volume)
        remote_path = "/Volumes/prod/ml_data/banner_recommendations/v1/"
        
        # Local: The temporary cache on the GPU driver node
        local_path = "/local_disk0/mds_cache/"
        
        dataset = StreamingSparkDataset(
            remote=remote_path,
            local=local_path,
            config=config,
            batch_size=1024,
            # THIS IS THE KEY:
            # Only keep 20GB of data on the local training node at any time.
            # It will auto-delete old chunks as it streams new ones from the Volume.
            cache_limit="20gb" 
        )
