# 3_AoA_Estimation_WAVELET_URM.py (Versão com organização de resultados)
from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import time
import sys

# Importa o seu novo módulo de denoising
import WAVELET

# =============================================================================
# ===                      BLOCO DE CONFIGURAÇÃO DO EXPERIMENTO             ===
# =============================================================================
# Altere os parâmetros aqui para cada novo experimento.
# Os nomes das pastas de resultados serão gerados automaticamente.

WAVELET_FAMILY = 'db4'      # Ex: 'db4', 'sym8', 'coif5'
DECOMPOSITION_LEVEL = 2     # Nível de decomposição (use 2 para evitar warnings)
THRESHOLD_MODE = 'soft'     # Modo de thresholding: 'soft' ou 'hard'
THRESHOLD_SCALE = 1.0       # Fator de escala para o threshold (1.0 = padrão)

# Pega o número da rodada a partir do argumento da linha de comando
round_num = sys.argv[1] if len(sys.argv) > 1 else '1'
# =============================================================================


# --- 1. Criação dinâmica dos nomes de diretório ---
# Cria um nome descritivo para a pasta de resultados deste experimento
experiment_name = f"{WAVELET_FAMILY}_level{DECOMPOSITION_LEVEL}_{THRESHOLD_MODE}_scale{str(THRESHOLD_SCALE).replace('.', 'p')}"
print(f"--- Iniciando Experimento: {experiment_name} ---")

# Define os diretórios de saída para os resultados .npy e para os gráficos
base_aoa_dir = "aoa_estimates_WAVELET_URM"
base_plots_dir = "plots_3_AoA_Estimation_WAVELET_URM"

output_aoa_dir = os.path.join(base_aoa_dir, experiment_name)
output_plots_dir = os.path.join(base_plots_dir, experiment_name)
round_plots_dir = os.path.join(output_plots_dir, f"Round_{round_num}")

os.makedirs(output_aoa_dir, exist_ok=True)
os.makedirs(round_plots_dir, exist_ok=True)


# --- 2. Data Loading and Preprocessing (sem alterações) ---
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
all_datasets = training_set_robot + test_set_robot + test_set_human

for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)


# --- 3. Implementação do Unitary Root-MUSIC (sem alterações) ---
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


# --- 4. Main AoA Estimation Loop ---
start_time = time.perf_counter()
for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")
    dataset['cluster_aoa_angles'] = []; dataset['cluster_aoa_powers'] = []

    for cluster in tqdm(dataset['clusters']):
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            noclutter_csi = CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx])
            
            # Chama a função de denoising com os parâmetros definidos no início do script
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
print(f"\n--- Tempo de Execução Total ({experiment_name}): {elapsed_time_total:.2f} segundos ---\n")


# --- 5. Save Results ---
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    # Salva os resultados .npy na nova pasta de experimento
    np.save(os.path.join(output_aoa_dir, dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join(output_aoa_dir, dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))
print(f"Resultados de AoA salvos em: {os.path.abspath(output_aoa_dir)}")


# --- 6. Evaluation and Visualization ---
for dataset in tqdm(test_set_robot + test_set_human):
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    ideal_aoas = np.arctan2(right, normal)
    dataset['cluster_groundtruth_aoas'] = ideal_aoas
    estimation_errors = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']
    
    norm = mcolors.Normalize(vmin=-45, vmax=45)
    for b in range(estimation_errors.shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        valid_errors = estimation_errors[:,b][~np.isnan(estimation_errors[:,b])]
        mae = np.mean(np.abs(np.rad2deg(valid_errors))) if valid_errors.size > 0 else float('nan')

        axes[0].set_title(f"Wavelet+URM AoA Estimates from Array {b}")
        axes[1].set_title(f"Ideal AoAs from Array {b}")
        axes[2].set_title(f"Wavelet+URM AoA Error, MAE = {mae:.2f}°")
        
        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        im3 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(estimation_errors[:,b]), norm = norm)
        for ax in axes: ax.set_xlabel("x coordinate in m"); ax.set_ylabel("y coordinate in m")
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]); fig.colorbar(im1, cax=cbar_ax, label="Angle in Degrees")
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        plot_filename = f"wavelet_urm_aoa_array{b}_{safe_dataset_basename}.png"
        # Salva os gráficos na nova pasta de experimento
        plt.savefig(os.path.join(round_plots_dir, plot_filename))
        plt.close(fig)

print(f"Gráficos salvos em: {os.path.abspath(round_plots_dir)}")