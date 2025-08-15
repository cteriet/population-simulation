import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightgbm as lgb
import optuna
import itertools
from joblib import Parallel, delayed
import os

# --- 0. Global Constants and Configuration ---
# You can adjust these parameters for your specific use case.

# -- Data Splitting --
DATE_COLUMN = 'event_date'
INITIAL_TRAIN_DAYS = 8
INITIAL_VAL_DAYS = 3
INITIAL_TEST_DAYS = 3

# -- Model Update Strategy --
# How often to perform a partial fit (in days)
PARTIAL_FIT_INTERVAL_DAYS = 1
# How often to do a full retrain on a large window (in days)
FULL_RETRAIN_INTERVAL_DAYS = 30
# The size of the window for a full retrain
FULL_RETRAIN_WINDOW_DAYS = 90

# -- Prediction/Ranking --
TOP_N_BANNERS = 5 # Number of top banners to recommend per customer
PREDICTION_EPSILON = 0.05 # Chance of random exploration in ranking
N_JOBS_PREDICTION = -1 # Number of CPU cores for parallel prediction (-1 uses all available)
MODEL_NAME_TAG = "FactorizationMachine_v1"

# --- 1. Custom Dataset & Data Handling ---

class BannerDataset(Dataset):
    """
    Custom PyTorch Dataset for handling the banner interaction data.
    This class makes the data loading process safer and more modular.
    """
    def __init__(self, df, categorical_cols, numerical_cols, label_col, weight_col=None):
        self.df = df
        # Find the index of the 'banner_id' column for easy access later
        self.banner_idx = categorical_cols.index('banner_id')

        # Convert data to numpy for faster access during training
        self.x_num = df[numerical_cols].values.astype(np.float32) if numerical_cols else np.array([])
        self.x_cat = df[categorical_cols].values.astype(np.int64)
        self.y = df[label_col].values.astype(np.float32)
        self.weights = df[weight_col].values.astype(np.float32) if weight_col and weight_col in df else np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_num_item = self.x_num[idx] if self.x_num.size > 0 else torch.empty(0)
        return (
            torch.tensor(x_num_item, dtype=torch.float32),
            torch.tensor(self.x_cat[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32)
        )

def temporal_split(df, date_col, train_days, val_days, test_days=None):
    """
    Splits a dataframe into train, validation, and optionally test sets based on time.
    """
    print(f"Performing temporal split on '{date_col}'...")
    df[date_col] = pd.to_datetime(df[date_col])
    end_date = df[date_col].max()
    
    # Validation set end date
    val_end = end_date
    if test_days:
        val_end -= pd.Timedelta(days=test_days)
    
    # Train set end date
    train_end = val_end - pd.Timedelta(days=val_days)

    # Filter dataframes
    train_df = df[df[date_col] < train_end]
    val_df = df[(df[date_col] >= train_end) & (df[date_col] < val_end)]
    
    print(f"Train period: {train_df[date_col].min().date()} to {train_df[date_col].max().date()} ({len(train_df)} rows)")
    print(f"Validation period: {val_df[date_col].min().date()} to {val_df[date_col].max().date()} ({len(val_df)} rows)")

    if test_days:
        test_df = df[df[date_col] >= val_end]
        print(f"Test period: {test_df[date_col].min().date()} to {test_df[date_col].max().date()} ({len(test_df)} rows)")
        return train_df, val_df, test_df
    
    return train_df, val_df


# --- 2. Factorization Machine Model ---
class FactorizationMachine(nn.Module):
    """
    Factorization Machine model implemented in PyTorch.
    This version is made safer by explicitly passing the banner field index.
    """
    def __init__(self, n_numeric_features, categorical_field_dims, embed_dim, banner_field_idx, dropout_rate=0.1):
        super().__init__()
        self.n_numeric_features = n_numeric_features
        self.categorical_field_dims = categorical_field_dims
        self.embed_dim = embed_dim
        self.banner_field_idx = banner_field_idx # Store banner index

        # --- Linear Part (w_i * x_i) ---
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, 1) for num_embeddings in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.linear_numeric = nn.Linear(self.n_numeric_features, 1)

        # --- Interaction Part (v_i, v_j) ---
        self.interaction_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embed_dim) for num_embeddings in categorical_field_dims
        ])
        if self.n_numeric_features > 0:
            self.interaction_numeric_vectors = nn.Parameter(torch.randn(n_numeric_features, embed_dim))

        self.bias = nn.Parameter(torch.zeros((1,)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_numeric, x_categorical):
        linear_terms = self.bias
        cat_linear_terms = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        linear_terms += torch.sum(torch.cat(cat_linear_terms, dim=1), dim=1, keepdim=True)

        if self.n_numeric_features > 0:
            linear_terms += self.linear_numeric(x_numeric)

        cat_interaction_vectors = [emb(x_categorical[:, i]) for i, emb in enumerate(self.interaction_embeddings)]
        
        if self.n_numeric_features > 0:
            numeric_interaction_vectors = x_numeric.unsqueeze(2) * self.interaction_numeric_vectors.unsqueeze(0)
            all_vectors = torch.cat(cat_interaction_vectors + [numeric_interaction_vectors], dim=1)
        else:
            all_vectors = torch.cat(cat_interaction_vectors, dim=1)
            
        all_vectors = self.dropout(all_vectors)
        sum_of_squares = torch.sum(all_vectors, dim=1).pow(2)
        square_of_sums = torch.sum(all_vectors.pow(2), dim=1)
        interaction_terms = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        logits = linear_terms + interaction_terms
        return logits.squeeze(1)

    def handle_new_banners(self, new_banner_id, strategy='cold', source_banner_id=None, source_group_ids=None):
        banner_embedding_idx = self.banner_field_idx # Use stored index
        with torch.no_grad():
            # This logic assumes the embedding layer is large enough.
            # In a real system, you might need to resize the embedding layer.
            if new_banner_id >= self.embeddings[banner_embedding_idx].num_embeddings:
                print(f"Error: Banner ID {new_banner_id} is out of bounds for the embedding layer.")
                return

            if strategy == 'cold':
                nn.init.normal_(self.embeddings[banner_embedding_idx].weight[new_banner_id], 0, 0.01)
                nn.init.normal_(self.interaction_embeddings[banner_embedding_idx].weight[new_banner_id], 0, 0.01)
                print(f"Cold start for new banner ID: {new_banner_id}")
            # ... other strategies ...
            else:
                raise ValueError("Invalid strategy or missing source IDs for warm start.")


# --- 3. Calibration Model ---
class Calibrator:
    """
    A wrapper for the Isotonic Regression model for calibrating classifier scores.
    """
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')

    def fit(self, fm_model, data_loader, device):
        print("\n--- Fitting Calibration Model (Isotonic Regression) ---")
        fm_model.eval()
        all_labels, all_logits = [], []
        with torch.no_grad():
            for x_num, x_cat, y, _ in tqdm(data_loader, desc="Generating scores for calibration"):
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                logits = fm_model(x_num, x_cat)
                all_labels.extend(y.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        self.model.fit(all_logits, all_labels)
        print("Calibration model fitting complete.")

    def predict(self, logits):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        return self.model.predict(logits)

# --- 4. Incremental Update Logic ---
def partial_fit_update(model, optimizer, new_data_df, categorical_cols, numerical_cols, propensity_feature_cols, device):
    """
    Performs an incremental update (partial fit) of the main model.
    """
    print(f"\n--- Starting Partial Fit on {len(new_data_df)} new samples ---")
    
    # a. Handle new banners that might appear in the new data
    # In a real system, you'd check against a master list of known banners.
    # Here, we assume the LabelEncoder mapping is updated externally.
    # For now, we simulate this by calling the cold-start handler for any new max ID.
    # This part requires careful management of feature mappings in production.
    # For this example, we'll assume the pre-processing handles this.

    # b. Train IPW model and get weights for the new data
    propensity_model_new = train_propensity_model(new_data_df, propensity_feature_cols, 'banner_id')
    new_data_df['ipw'] = calculate_stabilized_ipw(new_data_df, propensity_model_new, 'banner_id', propensity_feature_cols)

    # c. Create a DataLoader for the new data
    new_dataset = BannerDataset(new_data_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    new_loader = DataLoader(new_dataset, batch_size=1024, shuffle=True)
    
    # d. Perform the partial fit
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    model.train()
    total_loss = 0
    # Use a smaller learning rate for fine-tuning
    for g in optimizer.param_groups:
        g['lr'] = 1e-4 

    for x_num, x_cat, y, weights in tqdm(new_loader, desc="Partial Fit Training"):
        x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
        optimizer.zero_grad()
        outputs = model(x_num, x_cat)
        loss = (criterion(outputs, y) * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(new_loader)
    print(f"Partial fit complete. Average loss on new data: {avg_loss:.4f}")
    return model, optimizer

# --- 5. Scalable Banner Ranking ---

def _predict_scores_for_customer_chunk(customer_chunk_df, model, all_banners_df, cat_cols_model, num_cols_model, banner_manipulable_cols, device):
    """Helper function to predict scores for a chunk of customers, for parallelization."""
    model.to(device)
    model.eval()

    results = []
    
    # Get the index for the banner_id from the model's feature list
    banner_idx_model = cat_cols_model.index('banner_id')

    with torch.no_grad():
        for _, customer_row in customer_chunk_df.iterrows():
            customer_id = customer_row['customer_id']
            
            # 1. Generate all banner feature combinations for this customer
            # Create all permutations of manipulable banner features (e.g., position)
            manipulable_options = [all_banners_df[col].unique() for col in banner_manipulable_cols]
            option_combinations = list(itertools.product(*manipulable_options))
            
            # Create a large dataframe of every banner with every option combination
            num_banners = len(all_banners_df)
            num_options = len(option_combinations)
            
            # Repeat banner features for each option combination
            inference_df = pd.DataFrame(np.repeat(all_banners_df.values, num_options, axis=0), columns=all_banners_df.columns)
            
            # Tile the option combinations to match the repeated banners
            options_array = np.tile(option_combinations, (num_banners, 1))
            inference_df[banner_manipulable_cols] = options_array

            # 2. Add customer features
            for col in customer_chunk_df.columns:
                if col not in inference_df.columns:
                    inference_df[col] = customer_row[col]

            # 3. Prepare tensors for the model
            x_num_tensor = torch.tensor(inference_df[num_cols_model].values, dtype=torch.float32).to(device)
            x_cat_tensor = torch.tensor(inference_df[cat_cols_model].values, dtype=torch.long).to(device)
            
            # 4. Predict
            logits = model(x_num_tensor, x_cat_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            inference_df['predicted_prob'] = probs
            
            # 5. Select the best version of each banner (based on grouping)
            best_banners = inference_df.loc[inference_df.groupby('banner_group')['predicted_prob'].idxmax()]
            
            # 6. Store results
            best_banners['customer_id'] = customer_id
            results.append(best_banners)
            
    return pd.concat(results, ignore_index=True)


def get_ranked_banners_parallel(customers_df, all_banners_df, eligibility_df, model, calibrator, cat_cols_model, num_cols_model, banner_manipulable_cols, device):
    """
    Generates top N banner recommendations for a list of customers in a highly parallelized manner.
    """
    print(f"\n--- Generating Banner Rankings for {len(customers_df)} Customers ---")
    
    # For now, assume all banners are eligible
    eligible_banners = all_banners_df['banner_id'].unique()

    # Split customers into chunks for parallel processing
    n_jobs = os.cpu_count() if N_JOBS_PREDICTION == -1 else N_JOBS_PREDICTION
    customer_chunks = np.array_split(customers_df, n_jobs)
    
    print(f"Distributing prediction task across {n_jobs} workers...")
    
    # Run prediction in parallel
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_predict_scores_for_customer_chunk)(
            chunk, model, all_banners_df[all_banners_df['banner_id'].isin(eligible_banners)],
            cat_cols_model, num_cols_model, banner_manipulable_cols, device
        ) for chunk in customer_chunks
    )
    
    all_predictions_df = pd.concat(parallel_results, ignore_index=True)
    
    # Apply calibration to the final predicted probabilities
    # Note: Calibration requires raw logits, but here we approximate with probs for simplicity.
    # A more robust implementation would pass logits through the pipeline.
    # all_predictions_df['calibrated_prob'] = calibrator.predict(all_predictions_df['predicted_prob'])

    # Final Ranking and Exploration Logic
    final_recommendations = []
    for customer_id, group in tqdm(all_predictions_df.groupby('customer_id'), desc="Final Ranking"):
        # Exploration vs. Exploitation
        if np.random.rand() < PREDICTION_EPSILON:
            # Exploration: choose N random banners
            recs = group.sample(min(TOP_N_BANNERS, len(group)))
            recs['reason'] = 'exploration_random'
        else:
            # Exploitation: choose top N banners by probability
            recs = group.nlargest(TOP_N_BANNERS, 'predicted_prob')
            recs['reason'] = MODEL_NAME_TAG
            
        final_recommendations.append(recs)
        
    return pd.concat(final_recommendations, ignore_index=True)


# --- All other functions (IPW, train, evaluate, etc.) remain the same ---
# (Assuming the functions from the original script are present here)
def train_propensity_model(df, feature_cols, banner_col):
    X = df[feature_cols]
    y = df[banner_col]
    lgb_cat_features = [col for col in feature_cols if df[col].dtype.name in ['category', 'int64']]
    model = lgb.LGBMClassifier(objective='multiclass')
    model.fit(X, y, categorical_feature=lgb_cat_features)
    return model

def calculate_stabilized_ipw(df, propensity_model, banner_col, feature_cols, clip_range=(0.1, 10.0)):
    marginal_prob_map = (df[banner_col].value_counts() / len(df)).to_dict()
    propensities = propensity_model.predict_proba(df[feature_cols])
    actual_banner_indices = df[banner_col].values
    conditional_probs = propensities[np.arange(len(df)), actual_banner_indices]
    marginal_probs = df[banner_col].map(marginal_prob_map).values
    weights = marginal_probs / (conditional_probs + 1e-8)
    return np.clip(weights, clip_range[0], clip_range[1])

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y, weights in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(x_num, x_cat)
            loss = (criterion(outputs, y) * weights).mean()
            loss.backward()
            optimizer.step()
    return model
    
# --- Main Execution Block ---
if __name__ == '__main__':
    # --- A. Generate Synthetic Data with a Date Column ---
    print("Generating synthetic data with temporal component...")
    num_samples = 100000
    total_days = INITIAL_TRAIN_DAYS + INITIAL_VAL_DAYS + INITIAL_TEST_DAYS
    df = pd.DataFrame({
        'event_date': pd.to_datetime('2025-07-01') + pd.to_timedelta(np.random.randint(0, total_days, num_samples), unit='d'),
        'customer_id': np.random.randint(0, 1000, num_samples),
        'banner_id': np.random.randint(0, 100, num_samples),
        'product_id': np.random.randint(0, 10, num_samples),
        'customer_age': np.random.randint(18, 65, num_samples),
    })
    click_prob = 0.05 + (df['customer_id'] % 5 == 0) * 0.1 + (df['banner_id'] < 10) * 0.1
    df['clicked'] = (np.random.rand(num_samples) < click_prob).astype(int)

    # --- B. Preprocess Data ---
    categorical_cols = ['customer_id', 'banner_id', 'product_id']
    numerical_cols = ['customer_age']
    propensity_feature_cols = ['customer_id', 'product_id', 'customer_age']

    # Fit encoders on the entire dataset to have a consistent mapping
    encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    
    scaler = StandardScaler().fit(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # --- C. Initial Temporal Split ---
    train_df, val_df, test_df = temporal_split(
        df, DATE_COLUMN, INITIAL_TRAIN_DAYS, INITIAL_VAL_DAYS, INITIAL_TEST_DAYS
    )

    # --- D. Train Initial IPW and Main Models ---
    propensity_model = train_propensity_model(train_df, propensity_feature_cols, 'banner_id')
    train_df['ipw'] = calculate_stabilized_ipw(train_df, propensity_model, 'banner_id', propensity_feature_cols)
    val_df['ipw'] = calculate_stabilized_ipw(val_df, propensity_model, 'banner_id', propensity_feature_cols)

    train_dataset = BannerDataset(train_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    val_dataset = BannerDataset(val_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categorical_field_dims = [df[col].nunique() for col in categorical_cols]
    
    fm_model = FactorizationMachine(
        n_numeric_features=len(numerical_cols),
        categorical_field_dims=categorical_field_dims,
        embed_dim=16, # From hyperparameter tuning
        banner_field_idx=train_dataset.banner_idx,
        dropout_rate=0.2
    ).to(device)

    optimizer = optim.Adam(fm_model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    print("\n--- Training Initial Factorization Machine Model ---")
    fm_model = train_model(fm_model, train_loader, val_loader, optimizer, criterion, epochs=5, device=device)

    # --- E. Fit the Calibration Model ---
    calibrator = Calibrator()
    # We use the validation set to fit the calibrator
    calibrator.fit(fm_model, val_loader, device)

    # --- F. Simulate an Incremental Update ---
    # The 'test_df' here simulates the new data arriving after the initial training
    fm_model, optimizer = partial_fit_update(
        fm_model, optimizer, test_df, categorical_cols, numerical_cols, propensity_feature_cols, device
    )

    # --- G. Demonstrate Scalable Ranking ---
    # 1. Create necessary inputs for the ranking function
    # A dataframe of all unique customers to get recommendations for
    customers_for_ranking = df[['customer_id'] + numerical_cols].drop_duplicates('customer_id').head(100)
    
    # A dataframe of all possible banners and their features/metadata
    all_banners_metadata = df[['banner_id', 'product_id']].drop_duplicates('banner_id').reset_index(drop=True)
    all_banners_metadata['banner_group'] = all_banners_metadata['banner_id'] # Simple grouping
    all_banners_metadata['position'] = 'top' # Add a manipulable column
    banner_manipulable_cols = ['position'] # In a real scenario, this would have multiple values
    
    # An eligibility mapping (simplified for this example)
    eligibility_df = None # Assuming all are eligible

    # 2. Run the parallelized ranking function
    recommendations = get_ranked_banners_parallel(
        customers_df=customers_for_ranking,
        all_banners_df=all_banners_metadata,
        eligibility_df=eligibility_df,
        model=fm_model,
        calibrator=calibrator,
        cat_cols_model=categorical_cols,
        num_cols_model=numerical_cols,
        banner_manipulable_cols=banner_manipulable_cols,
        device=device
    )

    print("\n--- Sample of Final Recommendations ---")
    print(recommendations.head(15))
