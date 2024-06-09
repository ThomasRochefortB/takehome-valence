import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from pickle import dump
import os


def get_config():

    # Configuration dictionary (without model definition)
    config = {
        'use_morgan': True,
        'use_descriptors': True,
        'scaler': MinMaxScaler(),
        'use_feature_selection': False,
        'num_top_features': 15,
        'cv_folds': 10,
        'random_state': 42,
        'scoring': {
            'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)), greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False)
        },
    }

    return config

config = get_config()

# Base models for the ensemble
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=42)),
    ('lgbm', lgb.LGBMRegressor(n_estimators=100, max_depth=10, min_data_in_leaf=10, random_state=42, verbose=-1)),
    ('svr', SVR(C=1.0, epsilon=0.1)),
]

# Option 1: Using Voting Regressor
voting_ensemble_model = VotingRegressor(estimators=base_models)

# Option 2: Using Stacking Regressor with Linear Regression as the meta-learner
stacking_ensemble_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
# Choose the model to use
config['model'] = stacking_ensemble_model  # or stacking_ensemble_model

# Convert SMILES to Fingerprints using RDKit
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)  # Return a zero vector if the molecule is invalid

# Function to get dataset
def get_dataset(config):
    df = dm.data.freesolv()  # Replace this with the actual method to load your data

    if config['use_morgan']:
        df["morgan"] = df["smiles"].apply(smiles_to_fingerprint)

    if config['use_descriptors']:
        descriptor_names = [name for name, _ in Descriptors.descList]
        descriptors_list = []
        for smiles in df["smiles"]:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                descriptors = {name: getattr(Descriptors, name)(mol) for name in descriptor_names}
            else:
                descriptors = {name: None for name in descriptor_names}
            descriptors_list.append(descriptors)
        descriptors_df = pd.DataFrame(descriptors_list)
        df = pd.concat([df, descriptors_df], axis=1)

    if config['use_morgan'] and config['use_descriptors']:
        X = df.apply(lambda row: np.concatenate([row["morgan"], row[descriptor_names].values]), axis=1)
    elif config['use_morgan']:
        X = df["morgan"].apply(pd.Series)
    elif config['use_descriptors']:
        X = df[descriptor_names].values
    else:
        raise ValueError("No features selected. Please include at least Morgan fingerprints or descriptors.")

    y = df["expt"]

    X = np.array(list(X))
    y = np.array(y)

    return X, y

# Feature selection function
def select_features(X, y, num_top_features, random_state):
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    feature_importances = model.feature_importances_
    top_feature_indices = np.argsort(feature_importances)[-num_top_features:]
    return X[:, top_feature_indices], top_feature_indices

# Training and evaluation function for the ensemble model
def train_and_evaluate_model(X, y, config):
    feature_indices = None
    if config['use_feature_selection']:
        X, feature_indices = select_features(X, y, config['num_top_features'], config['random_state'])
        print(f"Selected Feature Indices: {feature_indices}")

    X = config['scaler'].fit_transform(X)

    kf = KFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])

    model = config['model']

    cv_results = cross_validate(model, X, y, cv=kf, scoring=config['scoring'], return_train_score=True)

    rmse_scores = -cv_results['test_rmse']
    mae_scores = -cv_results['test_mae']

    print("K-Fold Cross-Validation Results for Ensemble Model:")
    print(f"Root Mean Squared Error (RMSE) across folds: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean()}, Standard Deviation: {rmse_scores.std()}")
    print(f"Mean Absolute Error (MAE) across folds: {mae_scores}")
    print(f"Mean MAE: {mae_scores.mean()}, Standard Deviation: {mae_scores.std()}")

    train_rmse_scores = -cv_results['train_rmse']
    train_mae_scores = -cv_results['train_mae']

    print("Training Scores (for reference):")
    print(f"Training Root Mean Squared Error (RMSE) across folds: {train_rmse_scores}")
    print(f"Mean Training RMSE: {train_rmse_scores.mean()}, Standard Deviation: {train_rmse_scores.std()}")
    print(f"Training Mean Absolute Error (MAE) across folds: {train_mae_scores}")
    print(f"Mean Training MAE: {train_mae_scores.mean()}, Standard Deviation: {train_mae_scores.std()}")

    # Fit the ensemble model on the entire dataset
    model.fit(X, y)
    
    # Create the folder if it does not exist
    folder_path = ".saved_models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the trained model, scaler, and feature_indices
    with open(os.path.join(folder_path, "final_model.pkl"), "wb") as f:
        dump(model, f, protocol=5)
    with open(os.path.join(folder_path, "scaler.pkl"), "wb") as f:
        dump(config['scaler'], f, protocol=5)
    if feature_indices is not None:
        with open(os.path.join(folder_path, "feature_indices.pkl"), "wb") as f:
            dump(feature_indices, f, protocol=5)

    return model

if __name__ == "__main__":
    X, y = get_dataset(config)
    final_model = train_and_evaluate_model(X, y, config)