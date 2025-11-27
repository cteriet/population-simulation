import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lightgbm as lgb
from torch.utils.data import Dataset, DataLoader
import optuna
import optuna.visualization as vis

# --- 1. Custom Dataset for Banner Data ---
class BannerDataset(Dataset):
    """
    Custom PyTorch Dataset for handling the banner interaction data.
    This class makes the data loading process safer and more modular.
    """
    def __init__(self, df, categorical_cols, numerical_cols, label_col, weight_col=None):
        self.df = df
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.label_col = label_col
        self.weight_col = weight_col

        # Create mappings for safer access
        self.categorical_map = {col: i for i, col in enumerate(categorical_cols)}
        self.numerical_map = {col: i for i, col in enumerate(numerical_cols)}
        
        # Store the index for the banner_id field
        self.banner_idx = self.categorical_map.get('banner_id', -1)
        if self.banner_idx == -1:
            raise ValueError("'banner_id' must be in the categorical_cols list.")

        # Convert data to numpy for faster access
        self.x_num = df[numerical_cols].values.astype(np.float32) if numerical_cols else np.array([])
        self.x_cat = df[categorical_cols].values.astype(np.int64)
        self.y = df[label_col].values.astype(np.float32)
        self.weights = df[weight_col].values.astype(np.float32) if weight_col else np.ones(len(df), dtype=np.float32)

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

# --- 2. Factorization Machine Model ---
class FactorizationMachine(nn.Module):
    """
    Factorization Machine model implemented in PyTorch.
    This version is made safer by explicitly passing the banner field index.
    """
    def __init__(self, 
                 n_numeric_features: int, 
                 categorical_field_dims: List[int], 
                 embed_dim: int, 
                 banner_field_idx: int, 
                 dropout_rate: float = 0.1,
                 initial_bias: torch.tensor = torch.tensor((0.,))
                ):
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

        self.bias = nn.Parameter(initial_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_numeric, x_categorical):
        linear_terms = self.bias
        cat_linear_terms = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        linear_terms = linear_terms + torch.sum(torch.cat(cat_linear_terms, dim=1), dim=1, keepdim=True)

        cat_interaction_vectors = [emb(x_categorical[:, i]) for i, emb in enumerate(self.interaction_embeddings)]

        stacked_cat_vector = torch.stack(cat_interaction_vectors, dim=1)
        
        if self.n_numeric_features > 0:
            numeric_interaction_vectors = x_numeric.unsqueeze(2) * self.interaction_numeric_vectors.unsqueeze(0)
            all_vectors = torch.cat([stacked_cat_vector, numeric_interaction_vectors], dim=1)
        else:
            all_vectors = torch.cat(stacked_cat_vector, dim=1)
            
        all_vectors = self.dropout(all_vectors)
        
        sum_of_squares = torch.sum(all_vectors, dim=1).pow(2)
        square_of_sums = torch.sum(all_vectors.pow(2), dim=1)
        interaction_terms = 0.5 * torch.sum(sum_of_squares - square_of_sums, dim=1, keepdim=True)

        logits = linear_terms + interaction_terms
        return logits.squeeze(1)

    def handle_new_banners(self, new_banner_id, strategy='cold', source_banner_id=None, source_group_ids=None):
        banner_embedding_idx = self.banner_field_idx # Use stored index
        with torch.no_grad():
            if strategy == 'cold':
                nn.init.normal_(self.embeddings[banner_embedding_idx].weight[new_banner_id], 0, 0.01)
                nn.init.normal_(self.interaction_embeddings[banner_embedding_idx].weight[new_banner_id], 0, 0.01)
                print(f"Cold start for new banner ID: {new_banner_id}")
            elif strategy == 'warm_copy' and source_banner_id is not None:
                self.embeddings[banner_embedding_idx].weight[new_banner_id] = self.embeddings[banner_embedding_idx].weight[source_banner_id].clone()
                self.interaction_embeddings[banner_embedding_idx].weight[new_banner_id] = self.interaction_embeddings[banner_embedding_idx].weight[source_banner_id].clone()
                print(f"Warm start for banner {new_banner_id}, copying from {source_banner_id}")
            elif strategy == 'warm_average' and source_group_ids is not None:
                avg_linear_weight = self.embeddings[banner_embedding_idx].weight[source_group_ids].mean(dim=0)
                avg_interaction_weight = self.interaction_embeddings[banner_embedding_idx].weight[source_group_ids].mean(dim=0)
                self.embeddings[banner_embedding_idx].weight[new_banner_id] = avg_linear_weight
                self.interaction_embeddings[banner_embedding_idx].weight[new_banner_id] = avg_interaction_weight
                print(f"Warm start for banner {new_banner_id}, averaging from {len(source_group_ids)} banners.")
            else:
                raise ValueError("Invalid strategy or missing source IDs for warm start.")


# --- 3. Robust Inverse Propensity Weighting (IPW) ---
def train_propensity_model(df, feature_cols, banner_col):
    """
    Trains a calibrated LightGBM model to predict P(banner | features).
    Using a calibrated model makes the resulting IPW weights more stable.
    """
    print("Training calibrated propensity score model...")
    X = df[feature_cols]
    y = df[banner_col]
    
    # Identify categorical features for LightGBM
    lgb_cat_features = [col for col in feature_cols if df[col].dtype == 'int64' or df[col].dtype.name == 'category']

    # 1. Define the base model
    base_model = lgb.LGBMClassifier(objective='multiclass', n_estimators=100, learning_rate=0.05, num_leaves=31)
    
    # 2. Wrap the base model with CalibratedClassifierCV
    # This will train the base model and then a calibrator on cross-validated folds.
    # 'isotonic' is a non-parametric and powerful calibration method.
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

    # 3. Fit the calibrated model
    # We need to pass fit parameters for the base model via the `fit_params` argument.
    fit_params = {'categorical_feature': lgb_cat_features}
    calibrated_model.fit(X, y, fit_params=fit_params)
    
    print("Calibrated propensity model training complete.")
    return calibrated_model

def calculate_stabilized_ipw(df, propensity_model, banner_col='banner_id', feature_cols=None, clip_range=(0.1, 10.0)):
    """
    Calculates stabilized inverse propensity weights using a pre-trained ML model.
    """
    # 1. Calculate marginal probability P(banner)
    banner_counts = df[banner_col].value_counts()
    marginal_prob_map = (banner_counts / len(df)).to_dict()
    
    # 2. Get conditional probability P(banner | features) from the model
    print("Calculating propensities from model...")
    propensities = propensity_model.predict_proba(df[feature_cols])
    
    # Extract the probability for the actual banner that was shown
    actual_banner_indices = df[banner_col].values
    # Ensure all banner indices are within the model's class range
    valid_indices_mask = actual_banner_indices < propensities.shape[1]
    
    conditional_probs = np.ones(len(df))
    # Use advanced indexing to get the probability for the observed class
    conditional_probs[valid_indices_mask] = propensities[valid_indices_mask, actual_banner_indices[valid_indices_mask]]

    # 3. Calculate stabilized weights
    marginal_probs = df[banner_col].map(marginal_prob_map).values
    epsilon = 1e-8
    weights = marginal_probs / (conditional_probs + epsilon)

    # 4. Clip weights to avoid extreme values
    weights = np.clip(weights, clip_range[0], clip_range[1])
    
    print(f"IPW weights calculated. Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")
    return weights

# --- 4. Training and Evaluation (Largely unchanged) ---
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_num, x_cat, y, weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(x_num, x_cat)
            loss = criterion(outputs, y)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_num, x_cat, y, weights in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
                outputs = model(x_num, x_cat)
                loss = criterion(outputs, y)
                loss = (loss * weights).mean()
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return train_losses, val_losses

def objective(trial, train_loader, val_loader, n_numeric_features, categorical_field_dims, banner_field_idx, device):
    """
    The objective function for Optuna to minimize.
    """
    # 1. Suggest hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    epochs = 10 # A fixed number of epochs for each trial

    # 2. Build model and optimizer with suggested params
    model = FactorizationMachine(
        n_numeric_features=n_numeric_features,
        categorical_field_dims=categorical_field_dims,
        embed_dim=embed_dim,
        banner_field_idx=banner_field_idx,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # 3. Training and Validation Loop
    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y, weights in train_loader:
            x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(x_num, x_cat)
            loss = (criterion(outputs, y) * weights).mean()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_num, x_cat, y, weights in val_loader:
                x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
                outputs = model(x_num, x_cat)
                loss = (criterion(outputs, y) * weights).mean()
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 4. Report intermediate results for pruning
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss # Return the final validation loss for this trial

def evaluate_model(model, data_loader, device, title="Evaluation"):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x_num, x_cat, y, _ in data_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            outputs = model(x_num, x_cat)
            preds = torch.sigmoid(outputs)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    
    roc_auc = roc_auc_score(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    print(f"{title} ROC AUC: {roc_auc:.4f}")
    print(f"{title} PR AUC (Average Precision): {pr_auc:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{title} Metrics', fontsize=16)
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_title('ROC Curve'); axes[0, 0].set_xlabel('FPR'); axes[0, 0].set_ylabel('TPR'); axes[0, 0].legend(); axes[0, 0].grid(True)

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    prc_auc_val = auc(recall, precision)
    axes[0, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {prc_auc_val:.2f})')
    axes[0, 1].set_title('Precision-Recall Curve'); axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision'); axes[0, 1].legend(); axes[0, 1].grid(True)

    sorted_indices = np.argsort(all_preds)[::-1]
    lift = (np.cumsum(all_labels[sorted_indices]) / np.arange(1, len(all_labels) + 1)) / (np.sum(all_labels) / len(all_labels))
    axes[1, 0].plot(np.arange(1, len(all_labels) + 1) / len(all_labels), lift, lw=2)
    axes[1, 0].plot([0, 1], [1, 1], 'k--')
    axes[1, 0].set_title('Lift Curve'); axes[1, 0].set_xlabel('% Population'); axes[1, 0].set_ylabel('Lift'); axes[1, 0].grid(True)
    
    prob_true, prob_pred = calibration_curve(all_labels, all_preds, n_bins=10, strategy='uniform')
    axes[1, 1].plot(prob_pred, prob_true, "s-", label='Model')
    axes[1, 1].plot([0, 1], [0, 1], "k:", label='Perfectly calibrated')
    axes[1, 1].set_title('Calibration Curve'); axes[1, 1].set_xlabel('Mean Predicted Prob.'); axes[1, 1].set_ylabel('Fraction of Positives'); axes[1, 1].legend(); axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- 5. NEW: Plotting and Hyperparameter Tuning ---

def plot_training_history(train_losses, val_losses):
    """
    Visualizes the training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def objective(trial, train_loader, val_loader, n_numeric_features, categorical_field_dims, banner_field_idx, device):
    """
    The objective function for Optuna to minimize.
    """
    # 1. Suggest hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [8, 16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    epochs = 10 # A fixed number of epochs for each trial

    # 2. Build model and optimizer with suggested params
    model = FactorizationMachine(
        n_numeric_features=n_numeric_features,
        categorical_field_dims=categorical_field_dims,
        embed_dim=embed_dim,
        banner_field_idx=banner_field_idx,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # 3. Training and Validation Loop
    for epoch in range(epochs):
        model.train()
        for x_num, x_cat, y, weights in train_loader:
            x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(x_num, x_cat)
            loss = (criterion(outputs, y) * weights).mean()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_num, x_cat, y, weights in val_loader:
                x_num, x_cat, y, weights = x_num.to(device), x_cat.to(device), y.to(device), weights.to(device)
                outputs = model(x_num, x_cat)
                loss = (criterion(outputs, y) * weights).mean()
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 4. Report intermediate results for pruning
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss # Return the final validation loss for this trial

def tune_hyperparameters(n_trials, train_loader, val_loader, n_numeric, cat_dims, banner_idx, device):
    """
    Performs hyperparameter tuning using Optuna and visualizes the results.
    """
    print(f"\n--- Starting Hyperparameter Tuning for {n_trials} trials ---")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Use a lambda to pass the extra arguments to the objective function
    func = lambda trial: objective(trial, train_loader, val_loader, n_numeric, cat_dims, banner_idx, device)
    
    study.optimize(func, n_trials=n_trials)

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # --- NEW: Visualize Tuning Results ---
    print("\n--- Visualizing Tuning Study ---")
    # Check if the study has completed trials before plotting
    if len(study.trials) > 0:
        # Plot 1: Shows the objective value for each trial
        fig1 = vis.plot_optimization_history(study)
        fig1.show()
        
        # Plot 2: Shows the importance of each hyperparameter
        fig2 = vis.plot_param_importances(study)
        fig2.show()
    else:
        print("No completed trials to visualize.")
        
    return study

# --- 5. Prediction with Exploration (Largely unchanged) ---
def predict_banner_ranking(model, customer_features_num, customer_features_cat, eligible_banners, device, alpha=0.05):
    if np.random.rand() < alpha:
        np.random.shuffle(eligible_banners)
        print("Exploration triggered: returning random banner order.")
        return eligible_banners
    
    model.eval()
    with torch.no_grad():
        n_banners = len(eligible_banners)
        x_num_batch = torch.tensor(np.tile(customer_features_num, (n_banners, 1)), dtype=torch.float32).to(device)
        customer_part = np.tile(customer_features_cat, (n_banners, 1))
        banner_part = np.array(eligible_banners).reshape(-1, 1)
        x_cat_batch = torch.tensor(np.hstack([customer_part, banner_part]), dtype=torch.long).to(device)
        outputs = model(x_num_batch, x_cat_batch)
        scores = torch.sigmoid(outputs).cpu().numpy()
        ranked_indices = np.argsort(scores)[::-1]
        ranked_banners = [eligible_banners[i] for i in ranked_indices]
        print(f"Exploitation: ranked {len(ranked_banners)} banners.")
        return ranked_banners

# --- 6. Example Usage ---
if __name__ == '__main__':
    # --- A. Generate & Preprocess Data ---
    print("Generating synthetic data...")
    num_samples, num_customers, num_banners, num_products = 50000, 1000, 100, 10
    df = pd.DataFrame({
        'customer_id': np.random.randint(0, num_customers, num_samples),
        'banner_id': np.random.randint(0, num_banners, num_samples),
        'product_id': np.random.randint(0, num_products, num_samples),
        'customer_age': np.random.randint(18, 65, num_samples),
        'customer_segment': np.random.choice(['A', 'B', 'C'], num_samples, p=[0.5, 0.3, 0.2]),
    })
    df.loc[df['customer_age'] > 50, 'banner_id'] = np.random.choice([0, 1, 2], len(df[df['customer_age'] > 50]), p=[0.6, 0.2, 0.2])
    click_prob = 0.05 + (df['customer_segment'] == 'A') * 0.1 + (df['banner_id'] < 5) * 0.15 + (df['product_id'] == 0) * 0.2
    df['clicked'] = (np.random.rand(num_samples) < click_prob).astype(int)
    print(f"Data generated. Click rate: {df['clicked'].mean():.2%}")

    categorical_cols = ['customer_id', 'banner_id', 'product_id', 'customer_segment']
    numerical_cols = ['customer_age']
    for col in categorical_cols: df[col] = LabelEncoder().fit_transform(df[col])
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])
    
    # --- B. Calculate Robust IPW ---
    propensity_feature_cols = ['customer_id', 'product_id', 'customer_age', 'customer_segment']
    propensity_model = train_propensity_model(df, propensity_feature_cols, 'banner_id')
    df['ipw'] = calculate_stabilized_ipw(df, propensity_model, 'banner_id', propensity_feature_cols)
    
    # --- C. Create Datasets and DataLoaders ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['clicked'])
    
    train_dataset = BannerDataset(train_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    test_dataset = BannerDataset(test_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    # --- D. Initialize and Train Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    categorical_field_dims=[df[col].nunique() for col in categorical_cols]
    
    # Run the tuning process, which will now also plot the results
    study = tune_hyperparameters(
        n_trials=20, # Increase for a more thorough search
        train_loader=train_loader,
        val_loader=test_loader,
        n_numeric=len(numerical_cols),
        cat_dims=categorical_field_dims,
        banner_idx=train_dataset.banner_idx,
        device=device
    )

    print("\n--- Training Final Model with Best Hyperparameters ---")
    best_params = study.best_params
    
    model = FactorizationMachine(
        n_numeric_features=len(numerical_cols),
        categorical_field_dims=categorical_field_dims,
        embed_dim=best_params['embed_dim'],
        banner_field_idx=train_dataset.banner_idx,
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_model(model, train_loader, test_loader, optimizer, criterion, epochs=5, device=device)
    
    # --- E. Evaluate and Demonstrate ---
    evaluate_model(model, test_loader, device, title="Test Set")
    
    print("\nDemonstrating cold-start and prediction...")
    new_banner_id = num_banners
    model.handle_new_banners(new_banner_id, strategy='warm_average', source_group_ids=[3, 4, 6])

    sample_row = test_df.iloc[0]
    customer_num_feats = sample_row[numerical_cols].values
    customer_cat_feats_full = sample_row[categorical_cols].values
    customer_cat_feats_nobanner = np.delete(customer_cat_feats_full, train_dataset.banner_idx)
    
    eligible_banners = [2, 7, 15, 45, new_banner_id]
    ranked_list = predict_banner_ranking(model, customer_num_feats, customer_cat_feats_nobanner, eligible_banners, device, alpha=0.0)
    print(f"Ranked banners for customer (exploitation only): {ranked_list}")

# --- 7. Alternative Main Execution Block with Hyper Parameter Optimalization ---
if __name__ == 'ksjdhfkjdsdfshkj':
    # --- A. Generate & Preprocess Data (as before) ---
    print("Generating synthetic data...")
    num_samples, num_customers, num_banners, num_products = 50000, 1000, 100, 10
    df = pd.DataFrame({
        'customer_id': np.random.randint(0, num_customers, num_samples),
        'banner_id': np.random.randint(0, num_banners, num_samples),
        'product_id': np.random.randint(0, num_products, num_samples),
        'customer_age': np.random.randint(18, 65, num_samples),
        'customer_segment': np.random.choice(['A', 'B', 'C'], num_samples, p=[0.5, 0.3, 0.2]),
    })
    df.loc[df['customer_age'] > 50, 'banner_id'] = np.random.choice([0, 1, 2], len(df[df['customer_age'] > 50]), p=[0.6, 0.2, 0.2])
    click_prob = 0.05 + (df['customer_segment'] == 'A') * 0.1 + (df['banner_id'] < 5) * 0.15 + (df['product_id'] == 0) * 0.2
    df['clicked'] = (np.random.rand(num_samples) < click_prob).astype(int)
    
    categorical_cols = ['customer_id', 'banner_id', 'product_id', 'customer_segment']
    numerical_cols = ['customer_age']
    for col in categorical_cols: df[col] = LabelEncoder().fit_transform(df[col])
    df[numerical_cols] = StandardScaler().fit_transform(df[numerical_cols])
    
    # --- B. Calculate Robust IPW (as before) ---
    propensity_feature_cols = ['customer_id', 'product_id', 'customer_age', 'customer_segment']
    propensity_model = train_propensity_model(df, propensity_feature_cols, 'banner_id')
    df['ipw'] = calculate_stabilized_ipw(df, propensity_model, 'banner_id', propensity_feature_cols)
    
    # --- C. Create Datasets and DataLoaders (as before) ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['clicked'])
    train_dataset = BannerDataset(train_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    test_dataset = BannerDataset(test_df, categorical_cols, numerical_cols, 'clicked', 'ipw')
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    
    # --- D. Run Hyperparameter Tuning ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting hyperparameter tuning with Optuna on device: {device}")
    
    # Create a study object and specify the direction is to minimize the objective.
    # A pruner is used to stop unpromising trials early.
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Start the optimization. Optuna will call the 'objective' function n_trials times.
    study.optimize(lambda trial: objective(
        trial,
        train_loader,
        test_loader,
        n_numeric_features=len(numerical_cols),
        categorical_field_dims=[df[col].nunique() for col in categorical_cols],
        banner_field_idx=train_dataset.banner_idx,
        device=device
    ), n_trials=30) # Run 30 trials. Increase for a more thorough search.

    # --- E. Print Tuning Results ---
    print("\nHyperparameter tuning finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Validation Loss): {best_trial.value:.5f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\n--- All Trials Summary ---")
    results_df = study.trials_dataframe()
    print(results_df[['number', 'value', 'params_embed_dim', 'params_lr', 'params_dropout_rate', 'params_weight_decay', 'state']])

    # --- F. Train Final Model with Best Hyperparameters ---
    print("\nTraining final model with the best hyperparameters...")
    best_params = study.best_params
    final_model = FactorizationMachine(
        n_numeric_features=len(numerical_cols),
        categorical_field_dims=[df[col].nunique() for col in categorical_cols],
        embed_dim=best_params['embed_dim'],
        banner_field_idx=train_dataset.banner_idx,
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # You would typically train this final model for more epochs on the full training set
    # For this example, we'll just show the setup.
    print("Final model is ready for full training and evaluation.")
