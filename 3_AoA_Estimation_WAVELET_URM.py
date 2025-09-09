from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os
import time
import sys

import WAVELET

# --- 1. Read experiment parameters from command line arguments ---
if len(sys.argv) < 5:
    print("Usage: python 3_AoA_Estimation_WAVELET_URM.py <WAVELET_FAMILY> <DECOMPOSITION_LEVEL> <THRESHOLD_MODE> <THRESHOLD_SCALE>")
    sys.exit(1)

WAVELET_FAMILY = sys.argv[1]
DECOMPOSITION_LEVEL = int(sys.argv[2])
THRESHOLD_MODE = sys.argv[3]
THRESHOLD_SCALE = float(sys.argv[4])

# --- 2. Create directory structure based on parameters ---
experiment_name = f"{WAVELET_FAMILY}_level{DECOMPOSITION_LEVEL}_{THRESHOLD_MODE}_scale{str(THRESHOLD_SCALE).replace('.', 'p')}"
print(f"--- Starting AoA Estimation for: {experiment_name} ---")

main_output_folder = "AoA_Estimation_Wavelets"
output_experiment_dir = os.path.join(main_output_folder, experiment_name)
output_aoa_dir = os.path.join(output_experiment_dir, "estimates")

# output_plots_dir = os.path.join(output_experiment_dir, "plots") 

os.makedirs(output_aoa_dir, exist_ok=True)
# os.makedirs(output_plots_dir, exist_ok=True)


# --- 3. Data Loading and Preprocessing ---
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)


# --- 4. Unitary Root-MUSIC implementation ---
def get_unitary_rootmusic_estimator(chunksize = 4, shed_coeff_ratio = 0):
    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis = -1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))
    def unitary_rootmusic(R):
        assert(len(R) == chunksize); C = np.real(Q.H @ R @ Q); eig_val, eig_vec = np.linalg.eigh(C)
        eig_val = eig_val[::-1]; eig_vec = eig_vec[:,::-1]; source_count = 1
        En = eig_vec[:,source_count:]; ENSQ = Q @ En @ En.T @ Q.H
        coeffs = np.asarray([np.trace(ENSQ, offset = diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        roots = np.roots(coeffs); roots = roots[abs(roots) < 1.0]
        if not roots.size > 0: return np.nan, np.nan
        largest_root = np.argmax(1 / (1.0 - np.abs(roots)))
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])
    return unitary_rootmusic

umusic = get_unitary_rootmusic_estimator(4)


# --- 5. Main AoA Estimation Loop ---
start_time = time.perf_counter()
for dataset in tqdm(all_datasets, desc=f"Processing Datasets for {experiment_name}"):
    dataset['cluster_aoa_angles'] = []; dataset['cluster_aoa_powers'] = []

    for cluster in tqdm(dataset['clusters'], desc=f"Processing Clusters in {os.path.basename(dataset['filename'])}", leave=False):
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            noclutter_csi = CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx])
            
            denoised_csi = WAVELET.wavelet_denoise_csi(
                noclutter_csi,
                wavelet=WAVELET_FAMILY,
                level=DECOMPOSITION_LEVEL,
                mode=THRESHOLD_MODE,
                threshold_scale=THRESHOLD_SCALE
            )
            csi_by_transmitter_noclutter.append(denoised_csi)

        R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
        for tx_csi in csi_by_transmitter_noclutter:
            R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]

        music_results = [umusic(R[array]) for array in range(R.shape[0])]
        dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) if not np.isnan(angle_power[0]) else np.nan for angle_power in music_results]))
        dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])

end_time = time.perf_counter()
elapsed_time_total = end_time - start_time
print(f"\nExecution time for this run ({experiment_name}): {elapsed_time_total:.2f} seconds \n")


# --- 6. Save AoA Estimates ---
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join(output_aoa_dir, dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join(output_aoa_dir, dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))
print(f"AoA estimates saved to: {os.path.abspath(output_aoa_dir)}")


# --- 7. Evaluation and Saving MAE to TXT ---
mae_results_lines = []
mae_results_lines.append(f"MAE Results for Experiment: {experiment_name}\n")
mae_results_lines.append("-" * 50 + "\n")
mae_results_lines.append(f"Total Execution Time: {elapsed_time_total:.2f} seconds\n\n")


for dataset in tqdm(test_set_robot + test_set_human, desc="Calculating MAE"):
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    ideal_aoas = np.arctan2(right, normal)
    
    # Calcula os erros de estimação
    estimation_errors = dataset['cluster_aoa_angles'] - ideal_aoas
    
    # Extrai o nome base do dataset para identificação
    safe_dataset_basename = os.path.basename(dataset['filename'])
    
    # Adiciona uma linha de identificação para o dataset atual
    mae_results_lines.append(f"Dataset: {safe_dataset_basename}\n")

    # Calcula e registra o MAE para cada antena (array)
    for b in range(estimation_errors.shape[1]):
        valid_errors = estimation_errors[:,b][~np.isnan(estimation_errors[:,b])]
        
        # Calcula o MAE se houver erros válidos, senão registra como NaN
        if valid_errors.size > 0:
            mae = np.mean(np.abs(np.rad2deg(valid_errors)))
            mae_results_lines.append(f"  - Array {b}: MAE = {mae:.4f}°\n")
        else:
            mae_results_lines.append(f"  - Array {b}: MAE = NaN\n")
    
    mae_results_lines.append("\n") # Adiciona uma linha em branco para separar os datasets

# Define o caminho do arquivo de resultados
output_txt_path = os.path.join(output_experiment_dir, "mae_summary.txt")

# Escreve todas as linhas coletadas no arquivo de texto
with open(output_txt_path, 'w') as f:
    f.writelines(mae_results_lines)

print(f"MAE summary saved to: {os.path.abspath(output_txt_path)}")