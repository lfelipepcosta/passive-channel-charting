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
import matplotlib
import time
matplotlib.use('Agg')

# Create a directory to store the final triangulation estimates
os.makedirs("triangulation_estimates", exist_ok=True)

def aoa_power_to_kappa(aoa_power):
    """
    Heuristic function to convert AoA power from MUSIC to a concentration parameter.
    """
    return 5 * np.array(aoa_power)**6

def get_likelihood_function(aoa_datapoint, aoa_power_datapoint):
    """
    Factory function that creates the AoA likelihood function for a given data point.
    This implements Eq. 1 from the source paper.
    """
    # Convert AoA power to the kappa concentration parameter of the von Mises distribution
    kappas = aoa_power_to_kappa(aoa_power_datapoint)
    def likelihood_func(pos):
        # For a candidate position 'pos', calculate the ideal AoA from each array
        relative_pos = pos[:,np.newaxis,:] - espargos_0007.array_positions
        normal = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_normalvectors)
        right = np.einsum("dax,ax->da", relative_pos, espargos_0007.array_rightvectors)
        ideal_aoas = np.arctan2(right, normal)
        
        # Calculate the likelihood using the von Mises distribution probability density function.
        # This part includes calculations for numerical stability
        i0e_kappas = scipy.special.i0e(kappas)
        safe_i0e_kappas = np.where(i0e_kappas == 0, np.finfo(float).eps, i0e_kappas)
        log_exp_term = kappas * (np.cos(ideal_aoas - aoa_datapoint) - 1)
        log_exp_term = np.maximum(log_exp_term, -700)
        aoa_likelihoods_stable = (1.0 / (2 * np.pi * safe_i0e_kappas)) * np.exp(log_exp_term)
        
        # The total likelihood is the product of the likelihoods from all arrays
        return np.prod(aoa_likelihoods_stable, axis = -1)
    return likelihood_func

def aoa_estimation_worker_top_level(
    todo_queue,
    output_queue,
    current_dataset_cluster_aoa_angles,
    current_dataset_cluster_aoa_powers,
    current_dataset_HEIGHT,
    current_dataset_candidate_initial_positions,
    worker_id 
):
    """
    A worker process that finds the most likely position for a given AoA measurement.
    """
    def estimate_position_aoa(index):
        # Get the AoA data for this specific task (cluster)
        aoa_datapoint = current_dataset_cluster_aoa_angles[index]
        aoa_power_datapoint = current_dataset_cluster_aoa_powers[index]
        
        # Create the likelihood function for this specific data point
        likelihood_func = get_likelihood_function(aoa_datapoint, aoa_power_datapoint)
        
        final_pos = np.array([np.nan, np.nan]) 
        max_likelihood_val = np.nan

        try:
            # Coarse Grid Search
            # Evaluate the likelihood on a grid of candidate positions to find a good starting point
            likelihood_values_on_grid = likelihood_func(current_dataset_candidate_initial_positions)

            # Handle cases where the grid search fails
            if np.sum(np.isnan(likelihood_values_on_grid)) == likelihood_values_on_grid.size or np.sum(likelihood_values_on_grid) == 0 :
                initial_point = current_dataset_candidate_initial_positions[0] 
                final_pos = np.asarray([initial_point[0], initial_point[1]])
                max_likelihood_val = likelihood_values_on_grid[0] if likelihood_values_on_grid.size > 0 else np.nan
            else:
                # Find the best point from the grid search
                initial_point_idx = np.nanargmax(likelihood_values_on_grid)
                initial_point = current_dataset_candidate_initial_positions[initial_point_idx]
                init_value = np.asarray([initial_point[0], initial_point[1]])

                # Fine-grained Numerical Optimization
                # Define the objective function (we minimize the *negative* likelihood)
                objective_function = lambda pos_2d: -likelihood_func(np.asarray([[pos_2d[0], pos_2d[1], current_dataset_HEIGHT]]))
                
                # Use SciPy's L-BFGS-B optimizer to find the precise position that maximizes the likelihood
                optimize_res = scipy.optimize.minimize(
                    objective_function, init_value, method='L-BFGS-B',
                    options = {"gtol": 1e-7, "maxiter": 200, "eps": 1e-9}
                )
                
                final_pos = np.asarray([optimize_res.x[0], optimize_res.x[1]])
                max_likelihood_val = -optimize_res.fun
            
        except Exception: 
            pass
            
        return final_pos, max_likelihood_val

    # Main loop for the worker process
    while True:
        # Get a task (an index) from the queue
        index_tuple = todo_queue.get()
        if index_tuple is None:             # None is the signal to terminate
            break 
        
        # Perform the estimation and put the result on the output queue
        index = index_tuple[0]
        position, likelihood = estimate_position_aoa(index)
        output_queue.put((index, position, likelihood))


if __name__ == '__main__':
    # Necessary for multiprocessing on Windows
    mp.freeze_support()

    # The 'aoa_algorithm' variable is defined by the user input block at the start of the script.

    # Load data, AoA estimates, and cluster
    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

    all_datasets = training_set_robot + test_set_robot + test_set_human
    
    # Dictionary mapping the algorithm names to their respective results folders
    aoa_algorithms_folders = {
        "unitary root music": "aoa_estimates",
        "music": "aoa_estimates_MUSIC",
        "esprit": "aoa_estimates_ESPRIT",
        "delay and sum": "aoa_estimates_DAS",
        "capon": "aoa_estimates_CAPON",
        "sscapon": "aoa_estimates_SSCAPON"
    }

    # Loop to ensure a valid option is chosen
    while True:
        # Prompt the user to choose an algorithm
        prompt = ("\nChoose the AoA algorithm to use for the next steps:\n"
                " -> Unitary Root-MUSIC\n"
                " -> MUSIC\n"
                " -> ESPRIT\n"
                " -> Delay and Sum\n"
                " -> Capon\n"
                " -> SSCapon\n"
                "Your choice: ")
        
        aoa_algorithm = input(prompt)
        
        # Standardize the input: convert to lowercase and replace hyphens with spaces
        aoa_algorithm = aoa_algorithm.lower().replace('-', ' ')
        
        # Check if the choice is valid
        if aoa_algorithm in aoa_algorithms_folders:
            aoa_estimates_dir_original = aoa_algorithms_folders[aoa_algorithm]
            print(f"OK, using results from the folder: '{aoa_estimates_dir_original}'")
            break # Exit the loop if the choice is valid
        else:
            print("\n*** Error: Invalid option. Please type the name of one of the algorithms from the list. ***")

        # The variable 'aoa_estimates_dir_original' now contains the correct path
        # and can be used in the rest of the script.
    
    # The 'aoa_algorithm' variable is defined by the user input block at the start of the script.

    # Check for the special case of the default algorithm
    if aoa_algorithm == "unitary root music":
        # If it's the default, keep the original folder name
        plots_output_dir = "plots_4_Triangulation"
    else:
        # For all other algorithms, create a dynamic folder name
        # Convert the algorithm name to uppercase and replace spaces with underscores for a clean folder name
        algorithm_suffix = aoa_algorithm.upper().replace(' ', '_')
        plots_output_dir = f"plots_4_Triangulation_{algorithm_suffix}"

    os.makedirs(plots_output_dir, exist_ok=True)
    
    for dataset in all_datasets:
        # Load AoA data generated by the previous script
        dataset_name = os.path.basename(dataset['filename'])
        path_aoa_angles = os.path.join(aoa_estimates_dir_original, dataset_name + ".aoa_angles.npy")
        path_aoa_powers = os.path.join(aoa_estimates_dir_original, dataset_name + ".aoa_powers.npy")

        if not os.path.exists(path_aoa_angles) or not os.path.exists(path_aoa_powers):
            print(f"ERROR: AOA files not found for {dataset_name}. Please ensure they are in the '{aoa_estimates_dir_original}' directory. Exiting.")
            exit()
        dataset['cluster_aoa_angles'] = np.load(path_aoa_angles)
        dataset['cluster_aoa_powers'] = np.load(path_aoa_powers)
        
        print(f"Clustering dataset: {dataset['filename']}")
        cluster_utils.cluster_dataset(dataset)

    # Main Triangulation Loop (using multiprocessing)
    print("Starting triangulation position estimation for all datasets...")
    start_time = time.perf_counter()
    for dataset_idx_main, dataset_main in enumerate(tqdm(all_datasets, desc="Processing Datasets")):
        print(f"\nProcessing dataset: {dataset_main['filename']}")
        if 'cluster_aoa_angles' not in dataset_main or 'cluster_positions' not in dataset_main:
            print(f"WARN: Dataset {dataset_main['filename']} is missing 'cluster_aoa_angles' or 'cluster_positions'. Skipping triangulation for this dataset.")
            continue

        HEIGHT = np.mean(dataset_main['groundtruth_positions'][:,2])
        TOTAL_CLUSTERS = dataset_main['cluster_aoa_angles'].shape[0]
        
        num_actual_cluster_positions = dataset_main['cluster_positions'].shape[0]
        if TOTAL_CLUSTERS != num_actual_cluster_positions:
            print(f"WARN: Mismatch in cluster counts for {dataset_main['filename']}:")
            print(f"  cluster_aoa_angles (used for TOTAL_CLUSTERS) has {TOTAL_CLUSTERS} entries.")
            print(f"  cluster_positions (used for ground truth) has {num_actual_cluster_positions} entries.")
            print("  This may lead to issues in evaluation if not handled. Ensure AOA data aligns with clustered data.")

        if TOTAL_CLUSTERS == 0:
            print(f"WARN: No AOA clusters to process for {dataset_main['filename']}. Skipping.")
            dataset_main['triangulation_position_estimates'] = np.empty((0, 2))
            continue
        
        # Create a grid of candidate positions for the initial coarse search
        grid_resolution = 100
        candidate_xrange = np.linspace(np.min(espargos_0007.array_positions[:,0]) - 1, np.max(espargos_0007.array_positions[:,0]) + 1, grid_resolution)
        candidate_yrange = np.linspace(np.min(espargos_0007.array_positions[:,1]) - 1, np.max(espargos_0007.array_positions[:,1]) + 1, grid_resolution)
        candidate_initial_positions_for_current_dataset = np.transpose(np.meshgrid(candidate_xrange, candidate_yrange, HEIGHT)).reshape(-1, 3)

        # Setup the multiprocessing pool
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
                                   i_worker
                                   ))
            p.start()
            processes.append(p)

        # Distribute tasks to workers
        # Each task is just an index corresponding to a cluster
        print(f"  Adding {TOTAL_CLUSTERS} tasks to todo_queue...")
        for i in range(TOTAL_CLUSTERS):
            todo_queue.put((i,))

        # Add termination signals for the workers
        print(f"  Adding {num_processes_to_start} termination signals to todo_queue...")
        for _ in range(num_processes_to_start):
            todo_queue.put(None)

        # Collect results from workers
        dataset_main['triangulation_position_estimates'] = np.zeros((TOTAL_CLUSTERS, 2))
        print(f"  Collecting {TOTAL_CLUSTERS} results for {os.path.basename(dataset_main['filename'])}...")
        
        results_collected_count = 0
        with tqdm(total=TOTAL_CLUSTERS, desc=f"  Estimating Positions ({os.path.basename(dataset_main['filename'])})", leave=False) as pbar:
            while results_collected_count < TOTAL_CLUSTERS:
                try:
                    i_res, pos_res, lik_res = output_queue.get(timeout=120)
                    
                    if i_res is None:
                        print(f"  UNEXPECTED WARN: Received None as task index from output_queue before all tasks collected. Skipping this item.")
                        continue 

                    if pos_res is not None:
                        dataset_main['triangulation_position_estimates'][i_res,:] = pos_res
                    else: 
                        print(f"  WARN: Received pos_res=None for task index {i_res}. Using NaN for position.")
                        dataset_main['triangulation_position_estimates'][i_res,:] = np.array([np.nan, np.nan])
                    results_collected_count += 1
                    pbar.update(1)
                except mp.queues.Empty:
                    print(f"  TIMEOUT/ERROR: Output queue timed out waiting for result. Collected {results_collected_count}/{TOTAL_CLUSTERS}. Possible dead/stuck worker(s). Breaking collection.")
                    break 
                except Exception as e_q_get:
                    print(f"  ERROR: Exception during output_queue.get(): {e_q_get}. Collected {results_collected_count}/{TOTAL_CLUSTERS}. Breaking collection.")
                    break
        
        # Cleanly shut down worker processes
        print(f"  Finished collecting task results ({results_collected_count}/{TOTAL_CLUSTERS}) for {os.path.basename(dataset_main['filename'])}. Joining worker processes...")
        for p_idx, p in enumerate(processes):
            p.join(timeout=10) 
            if p.is_alive():
                print(f"  WARN: Worker process {p_idx} (PID {p.pid}) did not terminate cleanly after tasks and None signals. Forcing termination.")
                p.terminate()
                p.join() 
        print(f"  Finished processing for {dataset_main['filename']}.")
    end_time = time.perf_counter() # <--- FIM DO TIMER
    elapsed_time = end_time - start_time
    print(f"\n--- Total Triangulation Execution Time: {elapsed_time:.2f} seconds ---\n")
    
    # Save and Evaluate the Triangulation Results
    print("Saving triangulation estimates...")

    # The 'aoa_algorithm' variable is defined by the user input block at the start of the script.

    # Check for the special case of the default algorithm
    if aoa_algorithm == "unitary root music":
        # If it's the default, keep the original folder name
        triangulation_estimates_dir = "triangulation_estimates"
    else:
        # For all other algorithms, create a dynamic folder name
        algorithm_suffix = aoa_algorithm.upper().replace(' ', '_')
        triangulation_estimates_dir = f"triangulation_estimates_{algorithm_suffix}"

    # Now, ensure the directory exists and save the files there
    os.makedirs(triangulation_estimates_dir, exist_ok=True)
    for dataset in all_datasets:
        if 'triangulation_position_estimates' in dataset: 
            file_path = os.path.join(triangulation_estimates_dir, os.path.basename(dataset["filename"]) + ".npy")
            np.save(file_path, dataset["triangulation_position_estimates"])

    print(f"Triangulation estimates saved to: {os.path.abspath(triangulation_estimates_dir)}")
    
    print("\nStarting evaluation...")
    for dataset in (test_set_robot + test_set_human):
        print(f"\nEvaluation for {dataset['filename']}")

        if 'triangulation_position_estimates' not in dataset or \
           'cluster_positions' not in dataset or \
           dataset['triangulation_position_estimates'].shape[0] == 0:
            print(f"  Skipping evaluation for {dataset['filename']} due to missing/empty estimates or cluster_positions.")
            continue
        
        estimates_for_eval = dataset['triangulation_position_estimates'][:,:2]
        groundtruth_for_eval = dataset['cluster_positions'][:,:2]

        min_len = min(estimates_for_eval.shape[0], groundtruth_for_eval.shape[0])
        if min_len == 0:
            print(f"  No points to evaluate for {dataset['filename']} after length alignment. Skipping.")
            continue
        
        estimates_for_eval = estimates_for_eval[:min_len]
        groundtruth_for_eval = groundtruth_for_eval[:min_len]

        nan_mask_estimates = ~np.isnan(estimates_for_eval).any(axis=1)
        
        if np.sum(nan_mask_estimates) == 0 :
            print(f"  WARN: All estimates are NaN for {dataset['filename']} after alignment. Skipping evaluation.")
            continue
        
        if not np.all(nan_mask_estimates):
            print(f"  WARN: Found {np.sum(~nan_mask_estimates)} NaN estimates in {dataset['filename']}. Removing them and corresponding ground truth for evaluation.")
            estimates_for_eval = estimates_for_eval[nan_mask_estimates]
            groundtruth_for_eval = groundtruth_for_eval[nan_mask_estimates] 
        
        if estimates_for_eval.shape[0] == 0:
            print(f"  No valid (non-NaN) points to evaluate for {dataset['filename']}. Skipping.")
            continue

        errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(estimates_for_eval, groundtruth_for_eval)
        suptitle_text_eval = f"{os.path.splitext(os.path.basename(dataset['filename']))[0]}"
        title_text_eval = f"Triangulation: MAE = {mae:.3f}m, CEP = {cep:.3f}m"

        plot_filename_scatter = f"triangulation_scatter_{suptitle_text_eval}_mae{mae:.2f}_cep{cep:.2f}.png"
        full_plot_path_scatter = os.path.join(plots_output_dir, plot_filename_scatter)
        CCEvaluation.plot_colorized(estimates_for_eval, groundtruth_for_eval, 
                                    suptitle=suptitle_text_eval, title=title_text_eval, 
                                    show=False, outfile=full_plot_path_scatter)

        
        try:
            len_for_metrics = estimates_for_eval.shape[0]
            if int(0.05 * len_for_metrics) < 1 and len_for_metrics >=1 :
                 print(f"  INFO: Adjusting n_neighbors for CT/TW metrics as 0.05*N ({0.05*len_for_metrics:.2f}) is less than 1 for {len_for_metrics} points. Using n_neighbors=1 if N>=20, else skipping these metrics.")
                 
            metrics = CCEvaluation.compute_all_performance_metrics(estimates_for_eval, groundtruth_for_eval)
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        except Exception as e_metrics:
            print(f"  ERROR computing some performance metrics for {dataset['filename']}: {e_metrics}")
            print(f"    MAE: {mae:.3f}, CEP: {cep:.3f} (calculados separadamente com sucesso)")

        if errors.size > 0:
            ecdf_filename = f"triangulation_ecdf_{suptitle_text_eval}.jpg"
            full_ecdf_path = os.path.join(plots_output_dir, ecdf_filename)
            CCEvaluation.plot_error_ecdf(estimates_for_eval, groundtruth_for_eval, outfile=full_ecdf_path)
        else:
            print(f"  Skipping ECDF plot for {dataset['filename']} as there are no errors to plot.")

    print(f"\nPlots for Triangulation saved to: {os.path.abspath(plots_output_dir)}")