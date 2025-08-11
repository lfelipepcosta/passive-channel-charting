# =============================================================================
# SCRIPT 2 (V4): EVALUATING PRE-TRAINED MULTI-KERNEL KPCA MODELS
#
# Description:
# The PARAM_GRID is updated to include the sigmoid kernel. The script will
# automatically skip any models that failed to train and were not saved.
# =============================================================================

import os
import numpy as np
import joblib
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import KernelPCA

# --- Matplotlib & Custom Modules (as before) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import espargos_0007
import cluster_utils
import CRAP

# --- CONFIGURATION BLOCK (must match the training script) ---
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
MODELS_INPUT_DIR = "kpca_trained_models_multikernel"
EVALUATION_BASE_DIR = "kpca_evaluation_results_multikernel"
os.makedirs(EVALUATION_BASE_DIR, exist_ok=True)
# ---------------------------

# Helper function to create filenames (must be identical to training script)
def create_run_name(params):
    name = f"run"
    for key, value in sorted(params.items()):
        name += f"_{key[:4]}{str(value)}"
    return name.replace('.', 'p')

# get_unitary_rootmusic_estimator function remains the same
def get_unitary_rootmusic_estimator(chunksize=4, shed_coeff_ratio=0):
    I = np.eye(chunksize // 2); J = np.flip(np.eye(chunksize // 2), axis=-1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))
    def unitary_rootmusic(R):
        assert(len(R) == chunksize); C = np.real(Q.H @ R @ Q)
        eig_val, eig_vec = np.linalg.eigh(C); eig_val = eig_val[::-1]; eig_vec = eig_vec[:, ::-1]
        source_count = 1; En = eig_vec[:, source_count:]
        ENSQ = Q @ En @ En.T @ Q.H
        coeffs = np.asarray([np.trace(ENSQ, offset=diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        roots = np.roots(coeffs); roots = roots[abs(roots) < 1.0]
        if not roots.size > 0: return np.nan, np.nan
        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])
    return unitary_rootmusic


def main():
    # --- 1. Load TEST Data (as before) ---
    print("Loading and preparing TEST data...")
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
    evaluation_datasets = test_set_robot + test_set_human
    for dataset in evaluation_datasets:
        dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))
        cluster_utils.cluster_dataset(dataset)
    
    umusic = get_unitary_rootmusic_estimator(4)
    results_summary = []
    
    # --- 2. Start Evaluation Loop ---
    full_param_list = list(ParameterGrid(PARAM_GRID))
    print(f"\n--- Starting Evaluation for {len(full_param_list)} Pre-Trained Models ---")

    for params in tqdm(full_param_list, desc="Evaluating All Models"):
        run_name = create_run_name(params)
        model_name = run_name.replace("run", "kpca_model") + ".joblib"
        model_path = os.path.join(MODELS_INPUT_DIR, model_name)
        
        print(f"\n--- Evaluating with: {model_name} ---")

        if not os.path.exists(model_path):
            print(f"  ==> INFO: Model file not found at {model_path}. Skipping (likely failed during training).")
            continue
        kpca_model = joblib.load(model_path)

        run_plots_dir = os.path.join(EVALUATION_BASE_DIR, run_name, "plots")
        os.makedirs(run_plots_dir, exist_ok=True)
        run_maes = []
        
        # --- 3. AoA Estimation using the loaded model ---
        for dataset in evaluation_datasets:
            dataset['cluster_aoa_angles'] = []
            dataset['cluster_aoa_powers'] = []
            for cluster in dataset['clusters']:
                csi_by_transmitter_noclutter = [CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]) for tx_idx, csi in enumerate(cluster['csi_freq_domain'])]
                R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
                for tx_csi in csi_by_transmitter_noclutter:
                    R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]
                try:
                    n_arrays, n_antennas, _ = R.shape
                    n_flat_features = n_antennas * n_antennas
                    R_flat = R.reshape(n_arrays, n_flat_features)
                    X_features_real = np.concatenate([R_flat.real, R_flat.imag], axis=1)
                    
                    X_transformed = kpca_model.transform(X_features_real)
                    X_denoised_real_features = kpca_model.inverse_transform(X_transformed)

                    denoised_real_part = X_denoised_real_features[:, :n_flat_features]
                    denoised_imag_part = X_denoised_real_features[:, n_flat_features:]
                    R_flat_denoised = denoised_real_part + 1j * denoised_imag_part
                    R_denoised = R_flat_denoised.reshape(n_arrays, n_antennas, n_antennas)
                    music_results = [umusic(R_denoised[array]) for array in range(R_denoised.shape[0])]
                except Exception as e:
                    music_results = [(np.nan, np.nan)] * R.shape[0]
                dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) for angle_power in music_results]))
                dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))
            dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
            dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])
        
        # --- 4. Generate plots and collect MAE (identical to before) ---
        for dataset in evaluation_datasets:
            relative_pos = dataset['cluster_positions'][:, np.newaxis, :] - espargos_0007.array_positions
            normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
            right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
            ideal_aoas = np.arctan2(right, normal)
            estimation_errors = dataset['cluster_aoa_angles'] - ideal_aoas
            for b in range(estimation_errors.shape[1]):
                valid_errors = estimation_errors[:, b][~np.isnan(estimation_errors[:, b])]
                mae = np.mean(np.abs(np.rad2deg(valid_errors))) if valid_errors.size > 0 else np.nan
                run_maes.append(mae)

        # --- 5. Store summary for the completed run ---
        avg_mae_for_run = np.nanmean(run_maes)
        summary_dict = {'avg_mae': avg_mae_for_run}
        summary_dict.update(params)
        results_summary.append(summary_dict)
        print(f"Finished evaluation for {model_name}. Average MAE: {avg_mae_for_run:.2f}°")

    # --- 6. Print Final Summary Report ---
    print("\n\n" + "="*80); print("=== MULTI-KERNEL EVALUATION COMPLETE: FINAL SUMMARY ==="); print("="*80)
    
    sorted_results = sorted(results_summary, key=lambda x: x.get('avg_mae', float('inf')))
    
    all_keys = set()
    for d in sorted_results:
        all_keys.update(d.keys())
    headers = ['avg_mae'] + sorted([k for k in all_keys if k not in ['avg_mae', 'run_name']])
    
    header_str = " | ".join([f"{h:<12}" for h in headers])
    print(header_str)
    print("-" * len(header_str))

    for result in sorted_results:
        row_str = f"{result.get('avg_mae', 'N/A'):.2f}°{'':<8} | "
        for key in headers[1:]:
            row_str += f"{str(result.get(key, '-')):<12} | "
        print(row_str)

    print("-" * len(header_str))
    if sorted_results:
        best_run = sorted_results[0]
        print(f"\nBest configuration found: {best_run}")
    print("="*80)

if __name__ == "__main__":
    main()