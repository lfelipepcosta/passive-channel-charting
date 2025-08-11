# =============================================================================
# EXPERIMENTATION SCRIPT: HYPERPARAMETER SEARCH FOR KPCA+MUSIC
#
# Description:
# This script systematically evaluates the performance of the KPCA+MUSIC AoA
# estimation method across a grid of hyperparameters (n_components, gamma).
# It maintains consistency with the original analysis scripts for a fair
# comparison.
#
# Author: [Your Name]
# Based on codes from: Jeija (ESPARGOS)
# Date: 05/08/2025
# =============================================================================

import os
import sys
import time
import numpy as np
from tqdm.auto import tqdm

# --- Matplotlib Setup ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save plots to file
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# --- Custom Project Modules ---
import espargos_0007
import cluster_utils
import CRAP
from KPCA import KPCADenoiser

# =============================================================================
# ===                      CONFIGURATION BLOCK                            ===
# =============================================================================
# Define the grid of hyperparameters to search.
PARAM_GRID = {
    'n_components': [32],
    'gamma': [0.001, 0.01, 0.1, 1.0/32.0, 1.0, 10.0]
}

# Define base directory for all experiment results
SEARCH_BASE_DIR = "kpca_hyperparameter_search"
os.makedirs(SEARCH_BASE_DIR, exist_ok=True)
# =============================================================================


def get_unitary_rootmusic_estimator(chunksize=4, shed_coeff_ratio=0):
    """
    Defines the Unitary Root-MUSIC estimator function.
    This function is kept identical to the original scripts for consistency.
    """
    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis=-1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))

    def unitary_rootmusic(R):
        assert (len(R) == chunksize)
        C = np.real(Q.H @ R @ Q)
        eig_val, eig_vec = np.linalg.eigh(C)
        eig_val = eig_val[::-1]
        eig_vec = eig_vec[:, ::-1]
        source_count = 1
        En = eig_vec[:, source_count:]
        ENSQ = Q @ En @ En.T @ Q.H
        coeffs = np.asarray([np.trace(ENSQ, offset=diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        roots = np.roots(coeffs)
        roots = roots[abs(roots) < 1.0]
        if not roots.size > 0: return np.nan, np.nan
        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])
    return unitary_rootmusic


# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Data Once ---
    print("Loading datasets (this may take a moment)...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
    all_datasets = training_set_robot + test_set_robot + test_set_human

    print("Loading clutter signatures...")
    for dataset in all_datasets:
        dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

    print("Clustering datasets...")
    for dataset in all_datasets:
        cluster_utils.cluster_dataset(dataset)
    
    umusic = get_unitary_rootmusic_estimator(4)
    results_summary = []
    total_runs = len(PARAM_GRID['n_components']) * len(PARAM_GRID['gamma'])
    run_counter = 1

    # --- 2. Start Hyperparameter Search Loop ---
    print(f"\n--- Starting Hyperparameter Search for {total_runs} combinations ---")
    for n_comp in PARAM_GRID['n_components']:
        for gam in PARAM_GRID['gamma']:
            
            run_name = f"run_nc{n_comp}_g{gam:.4f}".replace('.', 'p')
            print(f"\n--- [{run_counter}/{total_runs}] Executing: {run_name} ---")

            # --- Create dedicated directories for this run ---
            run_data_dir = os.path.join(SEARCH_BASE_DIR, run_name, "aoa_estimates")
            run_plots_dir = os.path.join(SEARCH_BASE_DIR, run_name, "plots")
            os.makedirs(run_data_dir, exist_ok=True)
            os.makedirs(run_plots_dir, exist_ok=True)

            run_maes = [] # To store MAE values for this specific run
            
            # --- 3. AoA Estimation for all datasets with current parameters ---
            for dataset in tqdm(all_datasets, desc=f"Processing Datasets for {run_name}", leave=False):
                dataset['cluster_aoa_angles'] = []
                dataset['cluster_aoa_powers'] = []

                for cluster in dataset['clusters']:
                    # a. Calculate Covariance Matrix R (identical to original scripts)
                    csi_by_transmitter_noclutter = []
                    for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
                        csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))
                    
                    R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
                    for tx_csi in csi_by_transmitter_noclutter:
                        R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]

                    # b. KPCA Denoising Block
                    try:
                        n_arrays, n_antennas, _ = R.shape
                        n_flat_features = n_antennas * n_antennas
                        R_flat = R.reshape(n_arrays, n_flat_features)
                        X_features_real = np.concatenate([R_flat.real, R_flat.imag], axis=1)

                        kpca_denoiser = KPCADenoiser(n_components=n_comp, kernel='rbf', gamma=gam)
                        X_denoised_real_features = kpca_denoiser.fit_transform(X_features_real)

                        denoised_real_part = X_denoised_real_features[:, :n_flat_features]
                        denoised_imag_part = X_denoised_real_features[:, n_flat_features:]
                        R_flat_denoised = denoised_real_part + 1j * denoised_imag_part
                        R_denoised = R_flat_denoised.reshape(n_arrays, n_antennas, n_antennas)

                        # c. AoA estimation using the denoised matrix
                        music_results = [umusic(R_denoised[array]) for array in range(R_denoised.shape[0])]
                    except Exception as e:
                        music_results = [(np.nan, np.nan)] * R.shape[0]

                    # d. Store results (identical conversion to original scripts)
                    dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) for angle_power in music_results]))
                    dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))

                dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
                dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])
                
                # e. Save .npy data files for this dataset and run
                dataset_name = os.path.basename(dataset['filename'])
                np.save(os.path.join(run_data_dir, dataset_name + ".aoa_angles.npy"), dataset["cluster_aoa_angles"])
                np.save(os.path.join(run_data_dir, dataset_name + ".aoa_powers.npy"), dataset["cluster_aoa_powers"])

            # --- 4. Evaluation and Visualization for this run ---
            for dataset in tqdm(test_set_robot + test_set_human, desc=f"Generating Plots for {run_name}", leave=False):
                relative_pos = dataset['cluster_positions'][:, np.newaxis, :] - espargos_0007.array_positions
                normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
                right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
                ideal_aoas = np.arctan2(right, normal)
                estimation_errors = dataset['cluster_aoa_angles'] - ideal_aoas

                norm = mcolors.Normalize(vmin=-45, vmax=45)
                for b in range(estimation_errors.shape[1]):
                    valid_errors = estimation_errors[:, b][~np.isnan(estimation_errors[:, b])]
                    mae = np.mean(np.abs(np.rad2deg(valid_errors))) if valid_errors.size > 0 else np.nan
                    run_maes.append(mae) # Collect MAE for summary

                    # Plotting logic is identical to original scripts
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].set_title(f"KPCA+MUSIC AoA Estimates from Array {b}")
                    axes[1].set_title(f"Ideal AoAs from Array {b}")
                    axes[2].set_title(f"KPCA+MUSIC AoA Estimation Error, MAE = {mae:.2f}°")
                    im1 = axes[0].scatter(dataset["cluster_positions"][:, 0], dataset["cluster_positions"][:, 1], c=np.rad2deg(dataset["cluster_aoa_angles"][:, b]), norm=norm)
                    im2 = axes[1].scatter(dataset["cluster_positions"][:, 0], dataset["cluster_positions"][:, 1], c=np.rad2deg(ideal_aoas[:, b]), norm=norm)
                    im3 = axes[2].scatter(dataset["cluster_positions"][:, 0], dataset["cluster_positions"][:, 1], c=np.rad2deg(estimation_errors[:, b]), norm=norm)
                    for ax in axes: ax.set_xlabel("x coordinate in m"); ax.set_ylabel("y coordinate in m"); ax.axis('equal')
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]); fig.colorbar(im1, cax=cbar_ax, label="Angle in Degrees")
                    plt.tight_layout(rect=[0, 0, 0.9, 1])
                    
                    safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
                    plot_filename = f"aoa_array{b}_{safe_dataset_basename}.png"
                    plt.savefig(os.path.join(run_plots_dir, plot_filename))
                    plt.close(fig)
            
            # --- 5. Store summary for the completed run ---
            avg_mae_for_run = np.nanmean(run_maes)
            results_summary.append({
                'run_name': run_name,
                'n_components': n_comp,
                'gamma': gam,
                'avg_mae': avg_mae_for_run
            })
            print(f"Finished run {run_counter}/{total_runs}. Average MAE: {avg_mae_for_run:.2f}°")
            run_counter += 1

    # --- 6. Print Final Summary Report ---
    print("\n\n" + "="*60)
    print("=== HYPERPARAMETER SEARCH COMPLETE: FINAL SUMMARY ===")
    print("="*60)
    
    # Sort results by the best average MAE
    sorted_results = sorted(results_summary, key=lambda x: x['avg_mae'])
    
    print(f"{'Run Name':<25} | {'n_components':<15} | {'gamma':<15} | {'Avg. MAE':<10}")
    print("-"*70)
    for result in sorted_results:
        print(f"{result['run_name']:<25} | {result['n_components']:<15} | {result['gamma']:.4f}{'':<10} | {result['avg_mae']:.2f}°")
    
    print("-"*70)
    best_run = sorted_results[0]
    print(f"\nBest configuration found: {best_run['run_name']}")
    print(f"==> n_components={best_run['n_components']}, gamma={best_run['gamma']:.4f} with an Average MAE of {best_run['avg_mae']:.2f}°")
    print("="*60)