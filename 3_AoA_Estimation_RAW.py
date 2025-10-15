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

PARAM_GRID = {
    'n_components': [32],
    'gamma': [0.001, 0.01, 0.1, 1.0/32.0, 1.0, 10.0]
}

SEARCH_BASE_DIR = "kpca_hyperparameter_search"
os.makedirs(SEARCH_BASE_DIR, exist_ok=True)


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

    print(f"\n--- Starting Hyperparameter Search for {total_runs} combinations ---")
    for n_comp in PARAM_GRID['n_components']:
        for gam in PARAM_GRID['gamma']:
            
            run_name = f"run_nc{n_comp}_g{gam:.4f}".replace('.', 'p')
            print(f"\n--- [{run_counter}/{total_runs}] Executing: {run_name} ---")

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
                    csi_by_transmitter_noclutter = []
                    for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
                        csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))
                    
                    R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.float32)
                    R_old = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.float32)
                    for tx_csi in csi_by_transmitter_noclutter:
                        #print("Sinal1" + str(tx_csi.shape))
                        #print("Sinal2", tx_csi[0, 0, 0, 0, 0])
                        R_old = R_old + np.float32(np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0])
                        # Snapshot, array, linha, coluna (antenas), subportadora
                        for snapshot_i in range(tx_csi.shape[0]):
                            for array_i in range(tx_csi.shape[1]):
                                # Fazer o reshape a partir daqui para esses 4 vetores (tem que considerar que tem que fazer a matriz de correlação)
                                # Com o vetor grande aplicar a PCA.transform, selecionar os componentes de maior energia, aplicar PCA.transform_inverse
                               for linha_i in range(tx_csi.shape[2]):
                                    for antena_i in range(tx_csi.shape[3]):
                                        for antena_j in range(tx_csi.shape[3]):
                                            for subportadora_i in range(tx_csi.shape[4]):
                                                sinal_bruto_i = tx_csi[snapshot_i, array_i, linha_i, antena_i, subportadora_i] # Criar um vetor por array (4 vetores) a partir daqui para aplicação da PCA/KPCA a nível de antenas
                                                sinal_bruto_j = tx_csi[snapshot_i, array_i, linha_i, antena_j, subportadora_i]
                                                R[array_i, antena_i, antena_j] = R[array_i, antena_i, antena_j] + np.float32(sinal_bruto_i * np.conj(sinal_bruto_j) / tx_csi.shape[0])

                        print("R_old - R: ", np.sum(R_old - R))
                        continue
                    continue
                continue
            continue