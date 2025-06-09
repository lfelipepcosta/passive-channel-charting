import multiprocessing as mp
from tqdm.auto import tqdm
import scipy.optimize
import scipy.stats
import scipy.special
import espargos_0007
import cluster_utils
import numpy as np
import os
import CCEvaluation

# --- MODIFICAÇÃO: Adicionar matplotlib e definir backend 'Agg' ---
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt # Não é usado diretamente para plotar neste script
# --- FIM MODIFICAÇÃO ---

os.makedirs("triangulation_estimates", exist_ok=True)

# Funções originais mantidas como estavam
def aoa_power_to_kappa(aoa_power):
    return 5 * np.array(aoa_power)**6

def get_likelihood_function(aoa_datapoint, aoa_power_datapoint):
    kappas = aoa_power_to_kappa(aoa_power_datapoint)
    def likelihood_func(pos):
        relative_pos = pos[:,np.newaxis,:] - espargos_0007.array_positions
        normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
        right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
        ideal_aoas = np.arctan2(right, normal)
        
        i0e_kappas = scipy.special.i0e(kappas)
        safe_i0e_kappas = np.where(i0e_kappas == 0, np.finfo(float).eps, i0e_kappas)
        log_exp_term = kappas * (np.cos(ideal_aoas - aoa_datapoint) - 1)
        log_exp_term = np.maximum(log_exp_term, -700)
        aoa_likelihoods_stable = (1.0 / (2 * np.pi * safe_i0e_kappas)) * np.exp(log_exp_term)
        
        return np.prod(aoa_likelihoods_stable, axis = -1)
    return likelihood_func

def aoa_estimation_worker_top_level( # Assinatura original mantida
    todo_queue,
    output_queue,
    current_dataset_cluster_aoa_angles,
    current_dataset_cluster_aoa_powers,
    current_dataset_HEIGHT,
    current_dataset_candidate_initial_positions,
    worker_id 
):
    def estimate_position_aoa(index):
        aoa_datapoint = current_dataset_cluster_aoa_angles[index]
        aoa_power_datapoint = current_dataset_cluster_aoa_powers[index]
        
        likelihood_func = get_likelihood_function(aoa_datapoint, aoa_power_datapoint)
        
        final_pos = np.array([np.nan, np.nan]) 
        max_likelihood_val = np.nan

        try:
            likelihood_values_on_grid = likelihood_func(current_dataset_candidate_initial_positions)
            if np.sum(np.isnan(likelihood_values_on_grid)) == likelihood_values_on_grid.size or np.sum(likelihood_values_on_grid) == 0 :
                initial_point = current_dataset_candidate_initial_positions[0] 
                final_pos = np.asarray([initial_point[0], initial_point[1]])
                max_likelihood_val = likelihood_values_on_grid[0] if likelihood_values_on_grid.size > 0 else np.nan
            else:
                initial_point_idx = np.nanargmax(likelihood_values_on_grid)
                initial_point = current_dataset_candidate_initial_positions[initial_point_idx]
                init_value = np.asarray([initial_point[0], initial_point[1]])
                objective_function = lambda pos_2d: -likelihood_func(np.asarray([[pos_2d[0], pos_2d[1], current_dataset_HEIGHT]]))
                
                optimize_res = scipy.optimize.minimize(
                    objective_function, init_value, method='L-BFGS-B',
                    options = {"gtol": 1e-7, "maxiter": 200, "eps": 1e-9} # Opções originais do script fornecido
                )
                
                final_pos = np.asarray([optimize_res.x[0], optimize_res.x[1]])
                max_likelihood_val = -optimize_res.fun
            
        except Exception: 
            pass # Tratamento de exceção original
            
        return final_pos, max_likelihood_val

    while True:
        index_tuple = todo_queue.get()
        if index_tuple is None:
            break 
        
        index = index_tuple[0] # Lógica original para pegar o índice
        position, likelihood = estimate_position_aoa(index)
        output_queue.put((index, position, likelihood))


if __name__ == '__main__':
    mp.freeze_support()

    # --- MODIFICAÇÃO: Definir e criar diretório de plots ---
    plots_output_dir = "plots_4_Triangulation" 
    os.makedirs(plots_output_dir, exist_ok=True)
    # --- FIM MODIFICAÇÃO ---

    print("Loading datasets...")
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)
    print("Datasets loaded.")

    all_datasets = training_set_robot + test_set_robot + test_set_human

    print("Loading AOA estimates and clustering datasets...")
    # Caminho original para estimativas de AoA
    aoa_estimates_dir_original = "aoa_estimates" 
    # O script original não criava o diretório aoa_estimates aqui, apenas o usava.
    # Para ser "exatamente igual", não adiciono os.makedirs aqui.
    for dataset in all_datasets:
        dataset_name = os.path.basename(dataset['filename'])
        path_aoa_angles = os.path.join(aoa_estimates_dir_original, dataset_name + ".aoa_angles.npy")
        path_aoa_powers = os.path.join(aoa_estimates_dir_original, dataset_name + ".aoa_powers.npy")

        if not os.path.exists(path_aoa_angles) or not os.path.exists(path_aoa_powers):
            # Mensagem de erro original
            print(f"ERROR: AOA files not found for {dataset_name}. Please ensure they are in the '{aoa_estimates_dir_original}' directory. Exiting.")
            exit()
        dataset['cluster_aoa_angles'] = np.load(path_aoa_angles)
        dataset['cluster_aoa_powers'] = np.load(path_aoa_powers)
        
        print(f"Clustering dataset: {dataset['filename']}")
        cluster_utils.cluster_dataset(dataset)
    print("AOA estimates loaded and datasets clustered.")

    print("Starting triangulation position estimation for all datasets...")
    for dataset_idx_main, dataset_main in enumerate(tqdm(all_datasets, desc="Processing Datasets")):
        print(f"\nProcessing dataset: {dataset_main['filename']}")
        if 'cluster_aoa_angles' not in dataset_main or 'cluster_positions' not in dataset_main: # Lógica original
            print(f"WARN: Dataset {dataset_main['filename']} is missing 'cluster_aoa_angles' or 'cluster_positions'. Skipping triangulation for this dataset.")
            # O original não adicionava a chave 'triangulation_position_estimates' aqui, o que poderia levar a erro se acessada depois.
            # Mantendo "exatamente igual", não adiciono.
            continue

        # Lógica original para HEIGHT e TOTAL_CLUSTERS
        HEIGHT = np.mean(dataset_main['groundtruth_positions'][:,2])
        TOTAL_CLUSTERS = dataset_main['cluster_aoa_angles'].shape[0]
        
        num_actual_cluster_positions = dataset_main['cluster_positions'].shape[0]
        if TOTAL_CLUSTERS != num_actual_cluster_positions: # Lógica original
            print(f"WARN: Mismatch in cluster counts for {dataset_main['filename']}:")
            print(f"  cluster_aoa_angles (used for TOTAL_CLUSTERS) has {TOTAL_CLUSTERS} entries.")
            print(f"  cluster_positions (used for ground truth) has {num_actual_cluster_positions} entries.")
            print("  This may lead to issues in evaluation if not handled. Ensure AOA data aligns with clustered data.")

        if TOTAL_CLUSTERS == 0: # Lógica original
            print(f"WARN: No AOA clusters to process for {dataset_main['filename']}. Skipping.")
            dataset_main['triangulation_position_estimates'] = np.empty((0, 2))
            continue
        
        grid_resolution = 100
        candidate_xrange = np.linspace(np.min(espargos_0007.array_positions[:,0]) - 1, np.max(espargos_0007.array_positions[:,0]) + 1, grid_resolution)
        candidate_yrange = np.linspace(np.min(espargos_0007.array_positions[:,1]) - 1, np.max(espargos_0007.array_positions[:,1]) + 1, grid_resolution)
        candidate_initial_positions_for_current_dataset = np.transpose(np.meshgrid(candidate_xrange, candidate_yrange, HEIGHT)).reshape(-1, 3)

        todo_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = []
        num_processes_to_start = mp.cpu_count()
        
        print(f"  Spawning {num_processes_to_start} worker processes for {TOTAL_CLUSTERS} tasks...")
        for i_worker in range(num_processes_to_start):
            p = mp.Process(target=aoa_estimation_worker_top_level,
                             args=(todo_queue, output_queue,
                                   dataset_main['cluster_aoa_angles'],
                                   dataset_main['cluster_aoa_powers'],
                                   HEIGHT,
                                   candidate_initial_positions_for_current_dataset,
                                   i_worker # worker_id original
                                   ))
            p.start()
            processes.append(p)

        print(f"  Adding {TOTAL_CLUSTERS} tasks to todo_queue...")
        for i in range(TOTAL_CLUSTERS):
            todo_queue.put((i,)) # Original enviava tupla

        print(f"  Adding {num_processes_to_start} termination signals to todo_queue...")
        for _ in range(num_processes_to_start):
            todo_queue.put(None)

        dataset_main['triangulation_position_estimates'] = np.zeros((TOTAL_CLUSTERS, 2)) # Original
        print(f"  Collecting {TOTAL_CLUSTERS} results for {os.path.basename(dataset_main['filename'])}...")
        
        results_collected_count = 0
        with tqdm(total=TOTAL_CLUSTERS, desc=f"  Estimating Positions ({os.path.basename(dataset_main['filename'])})", leave=False) as pbar:
            while results_collected_count < TOTAL_CLUSTERS: # Lógica original de coleta
                try:
                    i_res, pos_res, lik_res = output_queue.get(timeout=120) # Timeout original
                    
                    if i_res is None: # Lógica original de tratamento de None (embora improvável para tarefas válidas)
                        print(f"  UNEXPECTED WARN: Received None as task index from output_queue before all tasks collected. Skipping this item.")
                        continue 
                    # A atribuição original era um pouco diferente, mas o efeito é o mesmo.
                    # Esta versão atribui NaN se pos_res for None.
                    if pos_res is not None:
                        dataset_main['triangulation_position_estimates'][i_res,:] = pos_res
                    else: 
                        print(f"  WARN: Received pos_res=None for task index {i_res}. Using NaN for position.")
                        dataset_main['triangulation_position_estimates'][i_res,:] = np.array([np.nan, np.nan])
                    results_collected_count += 1
                    pbar.update(1)
                except mp.queues.Empty: # queue.Empty é o tipo de exceção original
                    print(f"  TIMEOUT/ERROR: Output queue timed out waiting for result. Collected {results_collected_count}/{TOTAL_CLUSTERS}. Possible dead/stuck worker(s). Breaking collection.")
                    break 
                except Exception as e_q_get:
                    print(f"  ERROR: Exception during output_queue.get(): {e_q_get}. Collected {results_collected_count}/{TOTAL_CLUSTERS}. Breaking collection.")
                    break

        print(f"  Finished collecting task results ({results_collected_count}/{TOTAL_CLUSTERS}) for {os.path.basename(dataset_main['filename'])}. Joining worker processes...")
        for p_idx, p in enumerate(processes): # Lógica original de join/terminate
            p.join(timeout=10) 
            if p.is_alive():
                print(f"  WARN: Worker process {p_idx} (PID {p.pid}) did not terminate cleanly after tasks and None signals. Forcing termination.")
                p.terminate()
                p.join() 
        print(f"  Finished processing for {dataset_main['filename']}.")

    print("Saving triangulation estimates...")
    # Caminho original para salvar estimativas de triangulação
    triangulation_estimates_dir_original = "triangulation_estimates" 
    os.makedirs(triangulation_estimates_dir_original, exist_ok=True) # Adicionado para garantir que exista
    for dataset in all_datasets:
        if 'triangulation_position_estimates' in dataset: 
            np.save(os.path.join(triangulation_estimates_dir_original, os.path.basename(dataset["filename"])) + ".npy", dataset["triangulation_position_estimates"])
    print("Triangulation estimates saved.")

    print("\nStarting evaluation...")
    for dataset in (test_set_robot + test_set_human): # Original iterava assim
        print(f"\nEvaluation for {dataset['filename']}")
        # Verificações originais
        if 'triangulation_position_estimates' not in dataset or \
           'cluster_positions' not in dataset or \
           dataset['triangulation_position_estimates'].shape[0] == 0:
            print(f"  Skipping evaluation for {dataset['filename']} due to missing/empty estimates or cluster_positions.")
            continue
        
        estimates_for_eval = dataset['triangulation_position_estimates'][:,:2]
        groundtruth_for_eval = dataset['cluster_positions'][:,:2]

        min_len = min(estimates_for_eval.shape[0], groundtruth_for_eval.shape[0])
        if min_len == 0: # Original
            print(f"  No points to evaluate for {dataset['filename']} after length alignment. Skipping.")
            continue
        
        estimates_for_eval = estimates_for_eval[:min_len]
        groundtruth_for_eval = groundtruth_for_eval[:min_len]

        nan_mask_estimates = ~np.isnan(estimates_for_eval).any(axis=1)
        
        if np.sum(nan_mask_estimates) == 0 : # Original
            print(f"  WARN: All estimates are NaN for {dataset['filename']} after alignment. Skipping evaluation.")
            continue
        
        if not np.all(nan_mask_estimates): # Original
            print(f"  WARN: Found {np.sum(~nan_mask_estimates)} NaN estimates in {dataset['filename']}. Removing them and corresponding ground truth for evaluation.")
            estimates_for_eval = estimates_for_eval[nan_mask_estimates]
            groundtruth_for_eval = groundtruth_for_eval[nan_mask_estimates] 
        
        if estimates_for_eval.shape[0] == 0: # Original
            print(f"  No valid (non-NaN) points to evaluate for {dataset['filename']}. Skipping.")
            continue

        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(estimates_for_eval, groundtruth_for_eval)
        suptitle_text_eval = f"{os.path.splitext(os.path.basename(dataset['filename']))[0]}" # Renomeado para evitar conflito de escopo
        title_text_eval = f"Triangulation: MAE = {mae:.3f}m, CEP = {cep:.3f}m" # Renomeado
        
        # --- MODIFICAÇÃO: Salvar plot CCEvaluation.plot_colorized ---
        plot_filename_scatter = f"triangulation_scatter_{suptitle_text_eval}_mae{mae:.2f}_cep{cep:.2f}.png"
        full_plot_path_scatter = os.path.join(plots_output_dir, plot_filename_scatter)
        CCEvaluation.plot_colorized(estimates_for_eval, groundtruth_for_eval, 
                                    suptitle=suptitle_text_eval, title=title_text_eval, 
                                    show=False, outfile=full_plot_path_scatter) # show=False, outfile adicionado
        # print(f"Plot salvo em: {full_plot_path_scatter}") # Opcional, removido para ser mais "exatamente igual"
        # --- FIM MODIFICAÇÃO ---
        
        try: # Try-except original
            len_for_metrics = estimates_for_eval.shape[0]
            # A lógica original de ajuste de n_neighbors em CCEvaluation permanecia lá.
            if int(0.05 * len_for_metrics) < 1 and len_for_metrics >=1 : # Condição original
                 print(f"  INFO: Adjusting n_neighbors for CT/TW metrics as 0.05*N ({0.05*len_for_metrics:.2f}) is less than 1 for {len_for_metrics} points. Using n_neighbors=1 if N>=20, else skipping these metrics.")
                 
            metrics = CCEvaluation.compute_all_performance_metrics(estimates_for_eval, groundtruth_for_eval)
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        except Exception as e_metrics: # Tratamento de exceção original
            print(f"  ERROR computing some performance metrics for {dataset['filename']}: {e_metrics}")
            print(f"    MAE: {mae:.3f}, CEP: {cep:.3f} (calculados separadamente com sucesso)")

        if errors.size > 0: # Condição original
            # --- MODIFICAÇÃO: Salvar plot CCEvaluation.plot_error_ecdf ---
            ecdf_filename = f"triangulation_ecdf_{suptitle_text_eval}.jpg"
            full_ecdf_path = os.path.join(plots_output_dir, ecdf_filename)
            CCEvaluation.plot_error_ecdf(estimates_for_eval, groundtruth_for_eval, outfile=full_ecdf_path) # outfile adicionado
            # print(f"ECDF plot salvo em: {full_ecdf_path}") # Opcional
            # --- FIM MODIFICAÇÃO ---
        else:
            print(f"  Skipping ECDF plot for {dataset['filename']} as there are no errors to plot.")

    print(f"\nPlots de Triangulação salvos em: {os.path.abspath(plots_output_dir)}") # Adicionado para informar