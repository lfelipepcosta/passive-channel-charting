import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings

import espargos_0007
import cluster_utils
import CRAP

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_grid_per_array_curves(dataset_name, results_for_dataset, num_to_plot=20):
    rows, cols = 5, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), constrained_layout=True)
    fig.suptitle(f'Curva de Carga por Array (Amostra de {num_to_plot} clusters)\nDataset: {dataset_name}', fontsize=20)
    axes = axes.flatten()
    num_clusters_available = len(results_for_dataset.get(0, []))
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for i in range(min(num_to_plot, num_clusters_available)):
        ax = axes[i]
        for array_idx in range(espargos_0007.ARRAY_COUNT):
            if i < len(results_for_dataset[array_idx]):
                explained_variance = results_for_dataset[array_idx][i]
                cumulative_variance = np.cumsum(explained_variance)
                plot_x = np.insert(np.arange(1, 5), 0, 0); plot_y = np.insert(cumulative_variance, 0, 0)
                ax.plot(plot_x, plot_y, '-o', markersize=4, color=colors[array_idx], label=f'Array {array_idx}')
        ax.set_title(f'Cluster da Amostra #{i+1}'); ax.set_xticks(range(0, 5)); ax.set_xlabel('Nº de Componentes'); ax.set_ylabel('Energia Acumulada')
        ax.set_ylim(-0.05, 1.05); ax.grid(True, linestyle=':'); ax.legend(fontsize='small')
    for i in range(min(num_to_plot, num_clusters_available), len(axes)): axes[i].set_visible(False)
    safe_name = os.path.basename(dataset_name).replace(".tfrecords", ""); output_filename = f"curva_carga_GRID_por_array_{safe_name}.png"
    plt.savefig(output_filename); print(f"-> Gráfico de grade comparando arrays salvo como: '{output_filename}'"); plt.close(fig)

def plot_geral(results, title, legend_title, output_filename):
    plt.figure(figsize=(12, 8))
    for name, variances in results.items():
        if not variances: continue
        variances = np.array(variances); mean_variances = np.nanmean(variances, axis=0); std_variances = np.nanstd(variances, axis=0)
        mean_cumulative_curve = np.cumsum(mean_variances); std_cumulative_curve = np.sqrt(np.cumsum(std_variances**2))
        x_axis = np.arange(1, 5); plot_x = np.insert(x_axis, 0, 0); plot_y = np.insert(mean_cumulative_curve, 0, 0); plot_std = np.insert(std_cumulative_curve, 0, 0)
        line, = plt.plot(plot_x, plot_y, '-o', markersize=6, label=name)
        plt.fill_between(plot_x, plot_y - plot_std, plot_y + plot_std, alpha=0.15, color=line.get_color())
    plt.title(title, fontsize=16); plt.xlabel('Número de Componentes Principais', fontsize=12)
    plt.ylabel('Energia Acumulada Média', fontsize=12); plt.xticks(range(0, 5)); plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(title=legend_title, fontsize=10); plt.grid(True, linestyle=':'); plt.ylim(0, 1.05)
    plt.savefig(output_filename); print(f"-> Gráfico geral '{output_filename}' salvo."); plt.close()


if __name__ == "__main__":
    print("--- Análise de Energia ---")

    print("Carregando todos os datasets...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
    all_datasets = training_set_robot + test_set_robot + test_set_human

    for dataset in tqdm(all_datasets, desc="Pré-processando datasets"):
        dataset['clutter_acquisitions'] = np.load(os.path.join("clutter_channel_estimates", os.path.basename(dataset['filename']) + ".npy"))
        cluster_utils.cluster_dataset(dataset)

    all_results = {}
    
    for dataset in tqdm(all_datasets, desc="Analisando datasets"):
        dataset_name = os.path.basename(dataset['filename'])
        all_results[dataset_name] = {0: [], 1: [], 2: [], 3: []}
        
        for cluster in tqdm(dataset['clusters'], desc=f"Analisando clusters de {dataset_name}", leave=False):
            try:
                csi_by_transmitter_noclutter = []
                for tx_idx, csi in enumerate(cluster['csi_freq_domain']):
                    csi_by_transmitter_noclutter.append(CRAP.remove_clutter(csi, dataset['clutter_acquisitions'][tx_idx]))
                if not csi_by_transmitter_noclutter: continue

                R = np.zeros((espargos_0007.ARRAY_COUNT, espargos_0007.COL_COUNT, espargos_0007.COL_COUNT), dtype=np.complex64)
                for tx_csi in csi_by_transmitter_noclutter:
                    R = R + np.einsum("dbrms,dbrns->bmn", tx_csi, np.conj(tx_csi)) / tx_csi.shape[0]

                for array_idx in range(espargos_0007.ARRAY_COUNT):
                    R_para_analise = R[array_idx]
                    eigenvalues, _ = np.linalg.eigh(R_para_analise)
                    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
                    
                    sum_eig = np.sum(sorted_eigenvalues)
                    if sum_eig > 1e-9:
                        explained_variance = sorted_eigenvalues / sum_eig
                    else:
                        explained_variance = np.zeros(4)
                    
                    all_results[dataset_name][array_idx].append(explained_variance)
            except Exception:
                for array_idx in range(espargos_0007.ARRAY_COUNT):
                    all_results[dataset_name][array_idx].append(np.zeros(4))
                continue

    if not all_results or not any(variances for array_data in all_results.values() for variances in array_data.values()):
        print("\n\nNenhum cluster válido foi processado com sucesso.")
    else:
        print("\n\n" + "="*65); print("--- RESUMO NUMÉRICO: MÉDIA POR DATASET ---"); print("="*65)
        results_by_dataset = {os.path.basename(name): [var for i in range(4) for var in array_data[i]] for name, array_data in all_results.items()}
        for dataset_name, variances in results_by_dataset.items():
            if not variances: continue
            avg_variances = np.nanmean(variances, axis=0); cumulative_avg = np.cumsum(avg_variances)
            print(f"\nDataset: {dataset_name}"); print(f"{len(variances)} observações")
            print("------------------------------------------------------------")
            for i in range(len(cumulative_avg)): n_comp = i + 1; avg_energy = cumulative_avg[i] * 100; print(f"Com {n_comp} componente(s): {avg_energy:.2f}% da energia")
            print("------------------------------------------------------------")

        print("\n\n" + "="*65); print("--- RESUMO NUMÉRICO: MÉDIA POR ARRAY DE ANTENAS ---"); print("="*65)
        results_by_array = {0: [], 1: [], 2: [], 3: []}
        for array_data in all_results.values():
            for i in range(4): results_by_array[i].extend(array_data[i])
        for array_idx, variances in results_by_array.items():
            if not variances: continue
            avg_variances = np.nanmean(variances, axis=0); cumulative_avg = np.cumsum(avg_variances)
            print(f"\nArray de Antenas: {array_idx}")
            print("------------------------------------------------------------")
            for i in range(len(cumulative_avg)): n_comp = i + 1; avg_energy = cumulative_avg[i] * 100; print(f"Com {n_comp} componente(s): {avg_energy:.2f}% da energia")
            print("------------------------------------------------------------")

        # Geração dos Gráficos
        print("\n--- Gerando Gráficos ---")
        for name, variances_dict in all_results.items():
            if all(len(v) >= 20 for v in variances_dict.values()):
                plot_grid_per_array_curves(name, variances_dict)
        plot_geral(results_by_dataset, 'Curva de Carga Média por Dataset', 'Dataset', 'curva_carga_geral_por_dataset.png')
        results_by_array_labels = {f"Array {i}": v for i, v in results_by_array.items()}
        plot_geral(results_by_array_labels, 'Curva de Carga Média por Array', 'Array de Antenas', 'curva_carga_geral_por_array.png')