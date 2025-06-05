from tqdm.auto import tqdm
import espargos_0007
import cluster_utils
import numpy as np
import CRAP
import os

# --- MODIFICAÇÃO: Adicionar matplotlib e definir backend 'Agg' ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors # Import original mantido
import matplotlib.pyplot as plt    # Import original mantido
# --- FIM MODIFICAÇÃO ---


# Loading all the datasets can take some time...
training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

all_datasets = training_set_robot + test_set_robot + test_set_human


# Carregamento original dos dados de clutter
for dataset in all_datasets:
    dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))

os.makedirs("aoa_estimates", exist_ok=True)

# Clustering original
for dataset in all_datasets:
    cluster_utils.cluster_dataset(dataset)


# Função get_unitary_rootmusic_estimator original
def get_unitary_rootmusic_estimator(chunksize = 32, shed_coeff_ratio = 0):
    I = np.eye(chunksize // 2)
    J = np.flip(np.eye(chunksize // 2), axis = -1)
    Q = np.asmatrix(np.block([[I, 1.0j * I], [J, -1.0j * J]]) / np.sqrt(2))
    
    def unitary_rootmusic(R):
        assert(len(R) == chunksize)
        C = np.real(Q.H @ R @ Q)
    
        eig_val, eig_vec = np.linalg.eigh(C)
        eig_val = eig_val[::-1]
        eig_vec = eig_vec[:,::-1]

        source_count = 1
        En = eig_vec[:,source_count:]
        ENSQ = Q @ En @ En.T @ Q.H
    
        coeffs = np.asarray([np.trace(ENSQ, offset = diag) for diag in range(1, len(R))])
        coeffs = coeffs[:int(len(coeffs) * (1 - shed_coeff_ratio))]

        # Remove some of the smaller noise coefficients, trade accuracy for speed
        coeffs = np.hstack((coeffs[::-1], np.trace(ENSQ), coeffs.conj()))
        roots = np.roots(coeffs)
        roots = roots[abs(roots) < 1.0] # Linha original mantida
        largest_root = np.argmax(1 / (1.0 - np.abs(roots))) # Linha original mantida
        
        return np.angle(roots[largest_root]), np.abs(roots[largest_root])

    return unitary_rootmusic

umusic = get_unitary_rootmusic_estimator(4)


# Loop de estimativa de AoA original
for dataset in tqdm(all_datasets):
    print(f"AoA estimation for dataset: {dataset['filename']}")

    dataset['cluster_aoa_angles'] = []
    dataset['cluster_aoa_powers'] = []

    for cluster in tqdm(dataset['clusters']): # tqdm original sem leave=False
        csi_by_transmitter_noclutter = []
        for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
            csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))

        R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype = np.complex64)

        for tx_csi in csi_by_transmitter_noclutter:
            R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]

        music_results = [umusic(R[array]) for array in range(R.shape[0])] # 'array' como no original
        # Linha original para cálculo de ângulos AoA:
        dataset['cluster_aoa_angles'].append(np.asarray([np.arcsin(angle_power[0] / np.pi) for angle_power in music_results]))
        dataset['cluster_aoa_powers'].append(np.asarray([angle_power[1] for angle_power in music_results]))

    dataset['cluster_aoa_angles'] = np.asarray(dataset['cluster_aoa_angles'])
    dataset['cluster_aoa_powers'] = np.asarray(dataset['cluster_aoa_powers'])


# Salvamento original dos arquivos .npy de AoA
# O script original não criava o diretório "aoa_estimates" aqui, assumia que existia.
# Mantendo esse comportamento para ser "exatamente igual".
# Se o diretório não existir, np.save falhará, como no original.
for dataset in all_datasets:
    dataset_name = os.path.basename(dataset['filename'])
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_angles.npy"), np.asarray(dataset["cluster_aoa_angles"]))
    np.save(os.path.join("aoa_estimates", dataset_name + ".aoa_powers.npy"), np.asarray(dataset["cluster_aoa_powers"]))


# Imports de matplotlib já estavam no original, mas movidos para o topo com a modificação do backend
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt


# --- MODIFICAÇÃO: Definir e criar diretório para salvar os plots ---
plots_output_dir = "plots_3_AoA_Estimation" 
os.makedirs(plots_output_dir, exist_ok=True)
# --- FIM MODIFICAÇÃO ---


for dataset in tqdm(test_set_robot + test_set_human):
    # Compute expected AoA for position (original)
    relative_pos = dataset['cluster_positions'][:,np.newaxis,:] - espargos_0007.array_positions
    normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
    right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
    up = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_upvectors)
    ideal_aoas = np.arctan2(right, normal)
    ideal_eles = -np.arctan2(up, normal) # Original calculava, mas não usava nos plots 2D

    dataset['cluster_groundtruth_aoas'] = ideal_aoas
    # Cálculo original do erro de AoA (simples subtração)
    dataset['cluster_aoa_estimation_errors'] = dataset['cluster_aoa_angles'] - dataset['cluster_groundtruth_aoas']

    norm = mcolors.Normalize(vmin=-45, vmax=45) # Original
    for b in range(dataset["cluster_aoa_angles"].shape[1]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # Original

        # Estimated AoAs (plot original)
        im1 = axes[0].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_angles"][:,b]), norm = norm)
        axes[0].set_title(f"AoA Estimates seen from Array {b}")
        axes[0].set_xlabel("x coordinate in m")
        axes[0].set_ylabel("y coordinate in m")
        # axes[0].axis('equal') # O original não tinha, mas o plot do notebook parece ter. Removido para ser "exatamente igual" ao .py original.

        # Ideal AoAs (plot original)
        im2 = axes[1].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_groundtruth_aoas"][:,b]), norm = norm)
        axes[1].set_title(f"Ideal AoAs seen from Array {b}")
        axes[1].set_xlabel("x coordinate in m")
        axes[1].set_ylabel("y coordinate in m")
        # axes[1].axis('equal') # Removido

        # AoA Errors (plot original, MAE calculado com base no erro de subtração simples)
        # A variável 'im2' era sobrescrita aqui no original, mantendo esse comportamento.
        im2 = axes[2].scatter(dataset["cluster_positions"][:,0], dataset["cluster_positions"][:,1], c = np.rad2deg(dataset["cluster_aoa_estimation_errors"][:,b]), norm = norm)
        axes[2].set_title(f"AoA Estimation Error seen from Array {b}, MAE = {np.mean(np.abs(np.rad2deg(dataset['cluster_aoa_estimation_errors'][:,b]))):.2f}°")
        axes[2].set_xlabel("x coordinate in m")
        axes[2].set_ylabel("y coordinate in m")
        # axes[2].axis('equal') # Removido
        
        # Colorbar como no original .py (que era igual ao notebook neste ponto)
        cbar_ax = fig.add_axes([1.00, 0.2, 0.02, 0.6])
        fig.colorbar(im1, cax=cbar_ax)
        
        plt.tight_layout() # Original
        
        # --- MODIFICAÇÃO: Salvar plot em vez de mostrar ---
        safe_dataset_basename = os.path.basename(dataset['filename']).replace(".tfrecords", "")
        plot_filename = f"aoa_array{b}_{safe_dataset_basename}.png"
        full_plot_path = os.path.join(plots_output_dir, plot_filename)
        
        plt.savefig(full_plot_path)  # Salvando antes de fechar
        plt.close(fig) # Fechar a figura específica
        # --- FIM MODIFICAÇÃO ---
        
print(f"Plots de AoA salvos em: {os.path.abspath(plots_output_dir)}")