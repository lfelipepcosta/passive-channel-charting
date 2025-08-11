# =============================================================================
# SCRIPT 1 (V4): TRAINING KPCA MODELS WITH SIGMOID KERNEL & ERROR HANDLING
#
# Description:
# Includes the 'sigmoid' kernel in the hyperparameter search and adds a
# try-except block to gracefully handle and skip any models that fail
# during training due to numerical instability.
# =============================================================================

import os
import numpy as np
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import KernelPCA

import espargos_0007
import cluster_utils
import CRAP

# --- CONFIGURATION BLOCK ---
PARAM_GRID = [
    {
        'kernel': ['rbf'],
        'gamma': [0.01],
        'n_components': [32]
    },
    {
        'kernel': ['sigmoid'],
        'gamma': [0.01],
        'coef0': [0, 0.4, 0.5, 0.6, 0.7],
        'n_components': [16, 30, 32]
    }
]

MODELS_OUTPUT_DIR = "kpca_trained_models_multikernel"
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
# ---------------------------

def create_run_name(params):
    """Creates a descriptive filename from a dictionary of parameters."""
    name = f"kpca_model"
    for key, value in sorted(params.items()):
        name += f"_{key[:4]}{str(value)}"
    return name.replace('.', 'p')

def main():
    # --- 1. Load and Prepare Data (as before) ---
    print("Loading and preparing training data...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    for dataset in training_set_robot:
        dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))
        cluster_utils.cluster_dataset(dataset)
    
    all_covariance_matrices = []
    for dataset in tqdm(training_set_robot, desc="Collecting R matrices"):
        for cluster in dataset['clusters']:
            csi_by_transmitter_noclutter = [CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]) for tx_idx, csi in enumerate(cluster['csi_freq_domain'])]
            R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
            for tx_csi in csi_by_transmitter_noclutter:
                R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]
            for i in range(R.shape[0]):
                all_covariance_matrices.append(R[i])
    
    R_collection = np.array(all_covariance_matrices)
    print(f"Collected a total of {R_collection.shape[0]} covariance matrices.")
    
    SAMPLE_SIZE = 32048
    if R_collection.shape[0] > SAMPLE_SIZE:
        R_collection = R_collection[:SAMPLE_SIZE]
        print(f"==> REDUCING DATASET to the first {SAMPLE_SIZE} samples for feasibility.")

    n_samples, n_antennas, _ = R_collection.shape
    n_flat_features = n_antennas * n_antennas
    R_flat = R_collection.reshape(n_samples, n_flat_features)
    X_train_real = np.concatenate([R_flat.real, R_flat.imag], axis=1)
    print(f"Final training matrix shape: {X_train_real.shape}")

    # --- 2. Iterate, Train, and Save Models with Error Handling ---
    print("\n--- Training models for each hyperparameter combination ---")
    
    full_param_list = list(ParameterGrid(PARAM_GRID))
    
    for params in tqdm(full_param_list, desc="Training All Models"):
        run_name = create_run_name(params)
        model_path = os.path.join(MODELS_OUTPUT_DIR, run_name + ".joblib")
        print(f"Training model: {run_name}")

        # =================================================================== #
        # ===          NEW TRY-EXCEPT BLOCK TO HANDLE FAILURES            === #
        # =================================================================== #
        try:
            params_with_inverse = params.copy()
            params_with_inverse['fit_inverse_transform'] = True

            kpca_model = KernelPCA(**params_with_inverse)
            
            # This is the step that can fail for unstable kernels like sigmoid
            kpca_model.fit(X_train_real)
            
            # This part only runs if .fit() succeeds
            joblib.dump(kpca_model, model_path)
            print(f" ==> Model saved to {model_path}")

        except Exception as e:
            # If any error occurs during training, print a message and continue
            print(f" ==> FAILED to train model {run_name}. Error: {e}. Skipping.")
        # =================================================================== #

    print("\n--- All multi-kernel KPCA models have been trained successfully (or skipped on failure)! ---")

if __name__ == "__main__":
    main()