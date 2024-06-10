import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from pickle import dump
import os
import torch
from transformers import AutoModel
from padelpy import from_smiles
import requests
from tqdm import tqdm
import safe
from safe.tokenizer import SAFETokenizer


def download_file(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def check_and_download_tokenizer(tokenizer_path):
    tokenizer_url = (
        "https://huggingface.co/datamol-io/safe-gpt/resolve/main/tokenizer.json"
    )
    config_url = "https://huggingface.co/datamol-io/safe-gpt/resolve/main/config.json"
    if not os.path.exists(tokenizer_path):
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        print(f"Downloading {tokenizer_url} to {tokenizer_path}...")
        download_file(tokenizer_url, tokenizer_path)
        print("Download complete.")
    config_path = os.path.join(os.path.dirname(tokenizer_path), "config.json")
    if not os.path.exists(config_path):
        print(f"Downloading {config_url} to {config_path}...")
        download_file(config_url, config_path)
        print("Download complete.")


def setup_safe_tokenizer(tokenizer_path):
    check_and_download_tokenizer(tokenizer_path)
    tokenizer = SAFETokenizer().load(tokenizer_path)
    tokenizer = tokenizer.get_pretrained()
    return tokenizer


# Convert SMILES to Fingerprints using RDKit
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)  # Return a zero vector if the molecule is invalid


# Function to generate token embeddings from SMILES using Hugging Face model
def generate_smiles_embedding(smiles, where="tokenemb"):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the SAFETokenizer
    tokenizer_path = (
        "/home/thomas/Documents/scratch_thomas/GitHub/takehome-valence/tokenizer.json"
    )
    safe_tokenizer = setup_safe_tokenizer(tokenizer_path)

    # Load the model and move it to the appropriate device
    hf_model = AutoModel.from_pretrained("datamol-io/safe-gpt")
    hf_model.to(device)
    hf_model.eval()

    try:
        # Attempt to encode the SMILES string
        safe_str = safe.encode(smiles, slicer="mmpa")
    except Exception as e:
        print(f"Encoding failed for SMILES: {smiles} Error: {e}")
        # Use the tokenizer's end-of-sequence token as the fallback string
        safe_str = (
            safe_tokenizer.eos_token
        )  # This should be the EOS token from the tokenizer

    # Encode the SMILES or the fallback string using the tokenizer
    tokens = safe_tokenizer.encode(safe_str)
    tokens_tensor = torch.tensor([tokens]).to(
        device
    )  # Convert to tensor, add batch dimension, and move to device

    # return embeddings
    if where == "tokenemb":
        # Extract the word token embeddings (wte) and positional embeddings (wpe)
        with torch.no_grad():
            # Access the transformer component of the model
            transformer = hf_model
            # Get the word token embeddings
            word_token_embeddings = transformer.wte(tokens_tensor)
            # Get the positional embeddings
            positional_embeddings = transformer.wpe(
                torch.arange(tokens_tensor.size(1), device=device).unsqueeze(0)
            )
            # Add the word token embeddings and positional embeddings
            combined_embeddings = word_token_embeddings + positional_embeddings
        # Move embeddings to CPU and convert to NumPy arrays
        combined_embeddings = (
            combined_embeddings.mean(dim=1).squeeze().cpu().numpy()
        )  # (sequence_length, embedding_size)
        return combined_embeddings
    else:
        with torch.no_grad():
            outputs = hf_model(tokens_tensor)
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings


def get_config():
    config = {
        "use_morgan": False,
        "use_descriptors": True,
        "use_padel": False,
        "use_hf_embeddings": False,
        "hf_model_name": "datamol-io/safe-gpt",
        "scaler": MinMaxScaler(),
        "use_feature_selection": False,
        "num_top_features": 100,
        "cv_folds": 10,
        "random_state": 42,
        "scoring": {
            "rmse": make_scorer(
                lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
                greater_is_better=False,
            ),
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        },
        "train_voting_regressor": False,  # Flag to train Voting Regressor
        "train_stacking_regressor": True,  # Flag to train Stacking Regressor
    }
    return config


# Helper functions to generate features
def generate_morgan_features(df):
    return df["smiles"].apply(smiles_to_fingerprint).tolist()


def generate_rdkit_descriptors(df):
    descriptor_names = [name for name, _ in Descriptors.descList]
    descriptors_list = []
    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = [getattr(Descriptors, name)(mol) for name in descriptor_names]
        else:
            descriptors = [None] * len(descriptor_names)
        descriptors_list.append(descriptors)
    return pd.DataFrame(descriptors_list).fillna(0).values


def generate_padel_descriptors(df):
    padel_descriptors_list = []
    for smiles in tqdm(df["smiles"], desc="Generating PaDEL descriptors"):
        padel_descriptors = from_smiles(smiles, threads=-1)
        padel_descriptors_list.append(padel_descriptors)
    return (
        pd.DataFrame(padel_descriptors_list)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values
    )


def generate_hf_embeddings(df):
    embeddings_list = [generate_smiles_embedding(smiles) for smiles in df["smiles"]]
    return np.array(embeddings_list)


# Simplified get_dataset function with feature check
def get_dataset(config):
    df = dm.data.freesolv()  # Replace with your data loading method
    features = []

    # Generate and collect features based on configuration
    if config["use_morgan"]:
        print("Generating Morgan Fingerprints...")
        morgan_features = generate_morgan_features(df)
        features.append(morgan_features)

    if config["use_descriptors"]:
        print("Generating RDKit Descriptors...")
        rdkit_descriptors = generate_rdkit_descriptors(df)
        features.append(rdkit_descriptors)

    if config["use_padel"]:
        print("Generating PaDEL Descriptors...")
        padel_descriptors = generate_padel_descriptors(df)
        features.append(padel_descriptors)

    if config["use_hf_embeddings"]:
        print("Generating Hugging Face Embeddings...")
        hf_embeddings = generate_hf_embeddings(df)
        features.append(hf_embeddings)

    # Concatenate all features along the feature axis
    X = np.hstack(features) if features else np.array([])

    y = df["expt"].values

    # Check for features that are highly correlated with the target
    if X.size > 0:
        df_features = pd.DataFrame(X)
        df_features["target"] = y
        correlation_matrix = df_features.corr()
        correlation_with_target = correlation_matrix["target"].drop("target")

        # Define a high correlation threshold (e.g., 0.9 or higher)
        high_correlation_threshold = 0.9
        highly_correlated_features = correlation_with_target[
            correlation_with_target.abs() > high_correlation_threshold
        ]

        if not highly_correlated_features.empty:
            print(
                "Warning: The following features are highly correlated with the target:"
            )
            print(highly_correlated_features)

    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    return X, y


# Feature selection function
def select_features(X, y, num_top_features, random_state):
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=random_state
    )
    model.fit(X, y)
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[-num_top_features:]
    return X[:, top_feature_indices], top_feature_indices


# Helper function to log cross-validation results
def log_cv_results(log_file, cv_results, metric_name):
    scores = -cv_results[f"test_{metric_name}"]
    train_scores = -cv_results[f"train_{metric_name}"]

    log_file.write(f"{metric_name.upper()} across folds: {scores}\n")
    log_file.write(
        f"Mean {metric_name.upper()}: {scores.mean():.4f}, Standard Deviation: {scores.std():.4f}\n"
    )
    log_file.write(f"Training {metric_name.upper()} across folds: {train_scores}\n")
    log_file.write(
        f"Mean Training {metric_name.upper()}: {train_scores.mean():.4f}, Standard Deviation: {train_scores.std():.4f}\n"
    )


# Function to train and evaluate a model and log the results
def train_and_evaluate_model(X, y, config, model, model_name, log_file):
    log_file.write(f"\nTesting {model_name}:\n")
    log_file.write("-" * 30 + "\n")
    print(f"Training and evaluating {model_name}...")

    feature_indices = None
    if config["use_feature_selection"]:
        X, feature_indices = select_features(
            X, y, config["num_top_features"], config["random_state"]
        )
        log_file.write(f"Selected Feature Indices: {feature_indices}\n")

    X = config["scaler"].fit_transform(X)
    kf = KFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]
    )

    cv_results = cross_validate(
        model, X, y, cv=kf, scoring=config["scoring"], return_train_score=True
    )

    log_file.write("K-Fold Cross-Validation Results:\n")
    log_cv_results(log_file, cv_results, "rmse")
    log_cv_results(log_file, cv_results, "mae")

    mean_train_rmse = -cv_results["train_rmse"].mean()
    mean_test_rmse = -cv_results["test_rmse"].mean()

    print(
        f"\n{model_name} - Mean Train RMSE: {mean_train_rmse:.4f}, Mean Test RMSE: {mean_test_rmse:.4f}"
    )

    # Fit the model on the entire dataset
    model.fit(X, y)

    return model, mean_test_rmse, feature_indices


if __name__ == "__main__":
    config = get_config()
    # Base models for the ensemble
    base_models = {
        "rf": RandomForestRegressor(
            n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
        ),
        "lgbm": lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=10,
            min_data_in_leaf=25,
            feature_fraction=0.3,
            random_state=42,
            verbose=-1,
        ),
        "svr": SVR(C=10.0, epsilon=0.1),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(256, 256, 256),
            alpha=1.0,
            early_stopping=True,
            max_iter=200,
            random_state=42,
        ),
    }

    # Option 1: Using Voting Regressor
    voting_ensemble_model = VotingRegressor(
        estimators=[(name, model) for name, model in base_models.items()]
    )

    # Option 2: Using Stacking Regressor with Ridge Regression as the meta-learner
    stacking_ensemble_model = StackingRegressor(
        estimators=[(name, model) for name, model in base_models.items()],
        final_estimator=Ridge(),
    )

    X, y = get_dataset(config)
    best_model = None
    best_rmse = float("inf")
    best_model_name = ""

    with open("log_metrics.txt", "w") as log_file:
        # Test each base model individually
        for model_name, model in base_models.items():
            trained_model, mean_rmse, feature_indices = train_and_evaluate_model(
                X, y, config, model, model_name, log_file
            )
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = trained_model
                best_model_name = model_name
                best_feature_indices = feature_indices

        # Test Voting Regressor if the flag is set
        if config["train_voting_regressor"]:
            trained_model, mean_rmse, feature_indices = train_and_evaluate_model(
                X, y, config, voting_ensemble_model, "Voting Regressor", log_file
            )
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = trained_model
                best_model_name = "Voting Regressor"
                best_feature_indices = feature_indices

        # Test Stacking Regressor if the flag is set
        if config["train_stacking_regressor"]:
            trained_model, mean_rmse, feature_indices = train_and_evaluate_model(
                X, y, config, stacking_ensemble_model, "Stacking Regressor", log_file
            )
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = trained_model
                best_model_name = "Stacking Regressor"
                best_feature_indices = feature_indices

        log_file.write("\nBest Model Summary:\n")
        log_file.write(f"Best Model: {best_model_name}\n")
        log_file.write(f"Best RMSE: {best_rmse}\n")

    print(f"Best model is {best_model_name} with RMSE {best_rmse}")

    # Save the best model, scaler, and feature_indices if used
    folder_path = ".saved_models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, "final_model.pkl"), "wb") as f:
        dump(best_model, f, protocol=5)
    with open(os.path.join(folder_path, "scaler.pkl"), "wb") as f:
        dump(config["scaler"], f, protocol=5)

    if config["use_feature_selection"]:
        with open(os.path.join(folder_path, "feature_indices.pkl"), "wb") as f:
            dump(best_feature_indices, f, protocol=5)
