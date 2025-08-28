# =============================================================================
# SCRIPT 1: TRAINING KPCA MODELS
#
# Description:
# This script loads the TRAINING data, collects all covariance matrices,
# and then trains a separate KPCA model for each combination of
# hyperparameters in the defined grid. Each trained model is saved to disk.
# =============================================================================

import os
import numpy as np
import joblib # Using joblib is often more efficient for scikit-learn models
from tqdm.auto import tqdm

import espargos_0007
import cluster_utils
import CRAP
from KPCA import KPCADenoiser

# --- CONFIGURATION BLOCK ---
PARAM_GRID = {
    'n_components': [2, 4, 8, 16, 24, 26, 28, 30, 32],
    'gamma': [0.001, 0.01, 0.1, 1.0/32.0]
}
MODELS_OUTPUT_DIR = "kpca_trained_models"
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
# ---------------------------

def main():
    # --- 1. Load TRAINING Data Only ---
    print("Loading TRAINING dataset...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)

    print("Loading clutter signatures for training data...")
    for dataset in training_set_robot:
        dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

    print("Clustering training data...")
    for dataset in training_set_robot:
        cluster_utils.cluster_dataset(dataset)

    # --- 2. Collect ALL Covariance Matrices from the Training Set ---
    print("Collecting all covariance matrices from training data...")
    all_covariance_matrices = []
    for dataset in tqdm(training_set_robot, desc="Collecting R matrices"):
        for cluster in dataset['clusters']:
            csi_by_transmitter_noclutter = []
            for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
                csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))
            
            R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
            for tx_csi in csi_by_transmitter_noclutter:
                R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]
            
            # Add the 4 covariance matrices from this cluster to our collection
            for i in range(R.shape[0]):
                all_covariance_matrices.append(R[i])
    
    # Convert list to a large NumPy array
    R_collection = np.array(all_covariance_matrices)
    print(f"Collected a total of {R_collection.shape[0]} covariance matrices.")

    SAMPLE_SIZE = 32048 # Define o novo tamanho do dataset
    if R_collection.shape[0] > SAMPLE_SIZE:
        R_collection = R_collection[:SAMPLE_SIZE]
        print(f"==> REDUCING DATASET to the first {SAMPLE_SIZE} samples for feasibility.")

    # --- 3. Prepare the massive training data matrix for KPCA ---
    print("Preparing data for KPCA (flattening and complex-to-real)...")
    n_samples, n_antennas, _ = R_collection.shape
    n_flat_features = n_antennas * n_antennas
    R_flat = R_collection.reshape(n_samples, n_flat_features)
    X_train_real = np.concatenate([R_flat.real, R_flat.imag], axis=1)
    print(f"Final training matrix shape: {X_train_real.shape}")

    # --- 4. Train and Save a Model for Each Hyperparameter Combination ---
    print("\n--- Training models for each hyperparameter combination ---")
    for n_comp in PARAM_GRID['n_components']:
        for gam in PARAM_GRID['gamma']:
            run_name = f"kpca_model_nc{n_comp}_g{gam:.4f}".replace('.', 'p')
            model_path = os.path.join(MODELS_OUTPUT_DIR, run_name + ".joblib")

            print(f"Training model: {run_name}")
            
            # a. Initialize the KPCA denoiser
            kpca_denoiser = KPCADenoiser(n_components=n_comp, kernel='rbf', gamma=gam)
            
            # b. Fit the model on the ENTIRE training dataset
            kpca_denoiser.fit(X_train_real)
            
            # c. Save the TRAINED model object to a file
            joblib.dump(kpca_denoiser, model_path)
            print(f" ==> Model saved to {model_path}")

    print("\n--- All KPCA models have been trained and saved successfully! ---")


if __name__ == "__main__":
    main()