import FeatureEngineering
import tensorflow as tf
import espargos_0007 # Presume que este arquivo está no mesmo diretório ou no PYTHONPATH
import cluster_utils   # Presume que este arquivo está no mesmo diretório ou no PYTHONPATH
import neural_network_utils # Presume que este arquivo está no mesmo diretório ou no PYTHONPATH
import CCEvaluation    # Presume que este arquivo está no mesmo diretório ou no PYTHONPATH
import numpy as np
import os
import multiprocessing # Adicionado para freeze_support

# --- MODIFICAÇÃO: Adicionar matplotlib e definir backend 'Agg' ---
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt # Não é usado diretamente para plotar neste script, CCEvaluation lida com isso
# --- FIM MODIFICAÇÃO ---

# As definições de funções (combine_datasets, to_supervised_dataset, train_model)
# permanecem como estavam no seu código original.
# Certifique-se de que elas estejam definidas antes do bloco if __name__ == '__main__':
# ou importadas corretamente.

def combine_datasets(datasets):
    all_cluster_features = []
    all_cluster_positions = []

    for dataset_idx, dataset in enumerate(datasets): # Adicionado enumerate para debug
        print(f"DEBUG: combine_datasets - processando dataset {dataset_idx}, filename: {dataset.get('filename', 'N/A')}")
        if "cluster_features" not in dataset:
            print(f"ERROR: 'cluster_features' não encontrado no dataset {dataset.get('filename', 'N/A')} dentro de combine_datasets.")
            continue
        if "cluster_positions" not in dataset:
            print(f"ERROR: 'cluster_positions' não encontrado no dataset {dataset.get('filename', 'N/A')} dentro de combine_datasets.")
            continue

        all_cluster_features.append(dataset["cluster_features"])
        all_cluster_positions.append(dataset["cluster_positions"][:,:2]) 

    if not all_cluster_features: 
        print("ERROR: combine_datasets - all_cluster_features está vazia. Retornando arrays vazios.")
        return np.array([]), np.array([])


    print(f"DEBUG: combine_datasets - número de arrays de features para concatenar: {len(all_cluster_features)}")
    print(f"DEBUG: combine_datasets - número de arrays de posições para concatenar: {len(all_cluster_positions)}")
    
    for i, arr in enumerate(all_cluster_features):
        print(f"DEBUG: combine_datasets - shape de all_cluster_features[{i}]: {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
    for i, arr in enumerate(all_cluster_positions):
        print(f"DEBUG: combine_datasets - shape de all_cluster_positions[{i}]: {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")

    final_features = np.concatenate(all_cluster_features)
    final_positions = np.concatenate(all_cluster_positions)
    
    print(f"DEBUG: combine_datasets - shape final_features: {final_features.shape}")
    print(f"DEBUG: combine_datasets - shape final_positions: {final_positions.shape}")
    return final_positions, final_features


def to_supervised_dataset(datasets):
    print(f"DEBUG: to_supervised_dataset - recebendo {len(datasets)} datasets.")
    if not datasets:
        print("ERROR: to_supervised_dataset - lista de datasets de entrada está vazia. Retornando dataset TF vazio.")
        return tf.data.Dataset.from_tensor_slices((np.array([]).astype(np.float32), 
                                                   np.array([]).astype(np.float32)))

    print(f"DEBUG: to_supervised_dataset - processando primeiro dataset: {datasets[0].get('filename', 'N/A')}")
    if "cluster_features" not in datasets[0] or "cluster_positions" not in datasets[0]:
        print(f"ERROR: Primeiro dataset em to_supervised_dataset está faltando 'cluster_features' ou 'cluster_positions'.")
        return tf.data.Dataset.from_tensor_slices((np.array([]).astype(np.float32),
                                                   np.array([]).astype(np.float32)))


    first_dataset_features = datasets[0]["cluster_features"]
    first_dataset_positions = datasets[0]["cluster_positions"][:,:2]
    print(f"DEBUG: to_supervised_dataset - shapes do primeiro dataset: features {first_dataset_features.shape}, positions {first_dataset_positions.shape}")

    try:
        supervised_dataset = tf.data.Dataset.from_tensor_slices((first_dataset_features.astype(np.complex64),
                                                               first_dataset_positions.astype(np.float32)))
    except Exception as e:
        print(f"ERROR: Falha ao criar TensorSlice para o primeiro dataset: {e}")
        print(f"DEBUG: Tipos de dados - features: {first_dataset_features.dtype}, positions: {first_dataset_positions.dtype}")
        if first_dataset_features.size == 0 or first_dataset_positions.size == 0:
            print("ERROR: Features ou posições do primeiro dataset estão vazias antes de criar TensorSlice.")
            return tf.data.Dataset.from_tensor_slices((np.array([]).astype(np.complex64),
                                                       np.array([]).astype(np.float32)))
        raise e


    for idx, dataset_dict in enumerate(datasets[1:], 1):
        print(f"DEBUG: to_supervised_dataset - processando dataset subsequente {idx}: {dataset_dict.get('filename', 'N/A')}")
        if "cluster_features" not in dataset_dict or "cluster_positions" not in dataset_dict:
            print(f"ERROR: Dataset subsequente {dataset_dict.get('filename', 'N/A')} está faltando 'cluster_features' ou 'cluster_positions'. Pulando.")
            continue

        features = dataset_dict["cluster_features"]
        positions = dataset_dict["cluster_positions"][:,:2]
        print(f"DEBUG: to_supervised_dataset - shapes do dataset subsequente {idx}: features {features.shape}, positions {positions.shape}")
        
        if features.size == 0 or positions.size == 0:
            print(f"WARN: Features ou posições do dataset {dataset_dict.get('filename', 'N/A')} estão vazias. Pulando concatenação para este dataset.")
            continue

        try:
            current_tf_dataset = tf.data.Dataset.from_tensor_slices((features.astype(np.complex64),
                                                                   positions.astype(np.float32)))
            supervised_dataset = supervised_dataset.concatenate(current_tf_dataset)
        except Exception as e:
            print(f"ERROR: Falha ao criar ou concatenar TensorSlice para o dataset {dataset_dict.get('filename', 'N/A')}: {e}")
            print(f"DEBUG: Tipos de dados - features: {features.dtype}, positions: {positions.dtype}")

    print("DEBUG: to_supervised_dataset - concatenação de datasets TF concluída.")
    return supervised_dataset


def train_model(training_features, training_labels):
    print(f"DEBUG: train_model - training_features shape: {training_features.shape}, training_labels shape: {training_labels.shape}")
    if training_features.shape[0] == 0:
        print("ERROR: train_model - training_features está vazio. Não é possível treinar.")
        return None 

    TRAINING_BATCHES = 4000
    BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]

    print("DEBUG: train_model - Construindo modelo Keras...")
    supervised_model = neural_network_utils.construct_model(input_shape = FeatureEngineering.FEATURE_SHAPE, name = "SupervisedBaselineModel")

    training_input = tf.keras.layers.Input(shape = (), dtype = tf.int64)
    featprov = neural_network_utils.FeatureProviderLayer(dtype = tf.int64)
    featprov.set_features(training_features)
    csi_layer = featprov(training_input)
    output = supervised_model(csi_layer)
    training_model = tf.keras.models.Model(training_input, output, name = "TrainingModel")

    training_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.MeanSquaredError())
    print("DEBUG: train_model - Modelo compilado.")

    def random_index_batch_generator():
        batch_count = 0
        while True:
            current_batch_size_index = min(len(BATCH_SIZES) - 1, int(np.floor(batch_count / (TRAINING_BATCHES + 1) * len(BATCH_SIZES))))
            batch_size = BATCH_SIZES[current_batch_size_index]
            batch_count = batch_count + 1
            
            if training_features.shape[0] == 0:
                print("ERROR: random_index_batch_generator - training_features.shape[0] é 0. Não pode gerar índices.")
                yield np.array([], dtype=np.int64), np.array([[]], dtype=np.float32).reshape(0,2) 
                continue 

            indices = np.random.randint(training_features.shape[0], size = batch_size)
            positions = training_labels[indices,:2]
            yield indices.astype(np.int64), positions.astype(np.float32)

    print("DEBUG: train_model - Criando TensorFlow Dataset from_generator...")
    training_dataset = tf.data.Dataset.from_generator(random_index_batch_generator,
        output_signature=(tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)))

    print("DEBUG: train_model - Iniciando model.fit()...")
    training_model.fit(training_dataset, steps_per_epoch = TRAINING_BATCHES)
    print("DEBUG: train_model - model.fit() concluído.")
    return supervised_model


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # --- MODIFICAÇÃO: Definir e criar diretório de plots ---
    plots_output_dir = "plots_2_SupervisedBaseline" 
    os.makedirs(plots_output_dir, exist_ok=True)
    # --- FIM MODIFICAÇÃO ---

    print("DEBUG: Script principal iniciado.")

    training_set_robot = espargos_0007.load_dataset(espargos_0007.TRAINING_SET_ROBOT_FILES)
    test_set_robot = espargos_0007.load_dataset(espargos_0007.TEST_SET_ROBOT_FILES)
    test_set_human = espargos_0007.load_dataset(espargos_0007.TEST_SET_HUMAN_FILES)

    print(f"DEBUG: Datasets carregados. training_set_robot: {len(training_set_robot)} items, test_set_robot: {len(test_set_robot)} items, test_set_human: {len(test_set_human)} items.")

    all_datasets = training_set_robot + test_set_robot + test_set_human

    for dataset_idx, dataset in enumerate(all_datasets):
        print(f"DEBUG: Processando clutter para all_datasets[{dataset_idx}] - {dataset['filename']}")
        # Caminho original para arquivos de clutter
        clutter_file_path = os.path.join("clutter_channel_estimates", os.path.basename(dataset["filename"]) + ".npy")
        if os.path.exists(clutter_file_path):
            dataset["clutter_acquisitions"] = np.load(clutter_file_path)
        else:
            print(f"ERROR: Clutter file not found: {clutter_file_path}. Este dataset não terá 'clutter_acquisitions'.")
            
    print("DEBUG: Clutter carregado (ou tentado carregar).")

    for dataset_idx, dataset in enumerate(all_datasets):
        print(f"DEBUG: Clustering para all_datasets[{dataset_idx}] - {dataset['filename']}")
        if "clutter_acquisitions" not in dataset:
            print(f"Error: Dataset {dataset['filename']} is missing 'clutter_acquisitions'. Skipping clustering.")
            continue
        cluster_utils.cluster_dataset(dataset)

    print("DEBUG: Clustering concluído.")

    valid_datasets_for_feature_engineering = []
    for ds_idx, ds in enumerate(all_datasets):
        print(f"DEBUG: Verificando validade para feature engineering: all_datasets[{ds_idx}] - {ds.get('filename', 'N/A')}")
        has_clutter = "clutter_acquisitions" in ds
        has_clusters = "clusters" in ds
        if has_clutter and has_clusters:
            valid_datasets_for_feature_engineering.append(ds)
            print(f"DEBUG: Dataset {ds.get('filename', 'N/A')} é válido.")
        else:
            print(f"Skipping feature engineering for {ds.get('filename', 'N/A')} due to missing data (clutter_acquisitions: {has_clutter}, clusters: {has_clusters}).")

    if valid_datasets_for_feature_engineering:
        print("DEBUG: Iniciando FeatureEngineering.precompute_features...")
        FeatureEngineering.precompute_features(valid_datasets_for_feature_engineering)
        print("DEBUG: FeatureEngineering.precompute_features concluído.")
    else:
        print("ERROR: No valid datasets found for feature engineering. Exiting.")
        exit()

    print("DEBUG: Iniciando combine_datasets para training_set_robot...")
    valid_training_robot_sets = [ds for ds in training_set_robot if "cluster_features" in ds]
    if not valid_training_robot_sets:
        print("ERROR: Nenhum dataset válido em training_set_robot após feature engineering.")
    training_set_robot_groundtruth_positions, training_set_robot_features = combine_datasets(valid_training_robot_sets)
    print(f"DEBUG: training_set_robot_features shape: {training_set_robot_features.shape if isinstance(training_set_robot_features, np.ndarray) else type(training_set_robot_features)}")
    print(f"DEBUG: training_set_robot_groundtruth_positions shape: {training_set_robot_groundtruth_positions.shape if isinstance(training_set_robot_groundtruth_positions, np.ndarray) else type(training_set_robot_groundtruth_positions)}")

    print("DEBUG: Iniciando combine_datasets para test_set_robot...")
    valid_test_robot_sets = [ds for ds in test_set_robot if "cluster_features" in ds]
    test_set_robot_groundtruth_positions, test_set_robot_features = combine_datasets(valid_test_robot_sets)
    print(f"DEBUG: test_set_robot_features shape: {test_set_robot_features.shape if isinstance(test_set_robot_features, np.ndarray) else type(test_set_robot_features)}")

    print("DEBUG: Iniciando combine_datasets para test_set_human...")
    valid_test_human_sets = [ds for ds in test_set_human if "cluster_features" in ds]
    test_set_human_groundtruth_positions, test_set_human_features = combine_datasets(valid_test_human_sets)
    print(f"DEBUG: test_set_human_features shape: {test_set_human_features.shape if isinstance(test_set_human_features, np.ndarray) else type(test_set_human_features)}")
    print("DEBUG: combine_datasets concluído para todos os conjuntos.")

    print("DEBUG: Iniciando to_supervised_dataset para training_set_robot...")
    training_set_robot_supervised = to_supervised_dataset(valid_training_robot_sets) 
    print("DEBUG: Iniciando to_supervised_dataset para test_set_robot...")
    test_set_robot_supervised = to_supervised_dataset(valid_test_robot_sets)
    print("DEBUG: Iniciando to_supervised_dataset para test_set_human...")
    test_set_human_supervised = to_supervised_dataset(valid_test_human_sets)
    print("DEBUG: to_supervised_dataset concluído para todos os conjuntos.")

    if not isinstance(training_set_robot_features, np.ndarray) or training_set_robot_features.shape[0] == 0 or \
       not isinstance(training_set_robot_groundtruth_positions, np.ndarray) or training_set_robot_groundtruth_positions.shape[0] == 0:
        print("ERROR: Training features ou labels estão vazios ou não são NumPy arrays após combine_datasets. Verifique os passos anteriores.")
        exit()
    
    print(f"DEBUG: Shape das features de treinamento antes de treinar: {training_set_robot_features.shape}")
    print(f"DEBUG: Shape dos labels de treinamento antes de treinar: {training_set_robot_groundtruth_positions.shape}")

    print("Starting model training...")
    supervised_model = train_model(training_set_robot_features, training_set_robot_groundtruth_positions)
    
    if supervised_model is None:
        print("ERROR: Treinamento do modelo falhou. Encerrando.")
        exit()
    print("Model training finished.")

    print("Evaluating on Training Set (Robot)...")
    if not isinstance(training_set_robot_supervised, tf.data.Dataset) or training_set_robot_supervised.cardinality().numpy() == 0:
        print("ERROR: training_set_robot_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de treino.")
    else:
        training_set_robot_predictions = supervised_model.predict(training_set_robot_supervised.batch(1024))
        if training_set_robot_groundtruth_positions.shape[0] > 0:
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(training_set_robot_predictions, training_set_robot_groundtruth_positions)
            suptitle = f"Evaluated on Training Set (Robot)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            # --- MODIFICAÇÃO: Salvar plot em vez de mostrar ---
            plot_filename_train_robot = f"supervised_train_robot_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_train_robot = os.path.join(plots_output_dir, plot_filename_train_robot)
            CCEvaluation.plot_colorized(training_set_robot_predictions, training_set_robot_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_train_robot) # show=False, outfile adicionado
            print(f"Plot salvo em: {full_plot_path_train_robot}")
            # --- FIM MODIFICAÇÃO ---
        else:
            print("WARN: training_set_robot_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")


    print("Evaluating on Test Set (Robot)...")
    if not isinstance(test_set_robot_supervised, tf.data.Dataset) or test_set_robot_supervised.cardinality().numpy() == 0:
        print("ERROR: test_set_robot_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de teste robot.")
    else:
        test_set_robot_predictions = supervised_model.predict(test_set_robot_supervised.batch(1024))
        if test_set_robot_groundtruth_positions.shape[0] > 0:
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_robot_predictions, test_set_robot_groundtruth_positions)
            suptitle = f"Evaluated on Test Set (Robot)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            # --- MODIFICAÇÃO: Salvar plot em vez de mostrar ---
            plot_filename_test_robot_scatter = f"supervised_test_robot_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_test_robot_scatter = os.path.join(plots_output_dir, plot_filename_test_robot_scatter)
            CCEvaluation.plot_colorized(test_set_robot_predictions, test_set_robot_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_test_robot_scatter) # show=False, outfile adicionado
            print(f"Plot salvo em: {full_plot_path_test_robot_scatter}")
            
            metrics = CCEvaluation.compute_all_performance_metrics(test_set_robot_predictions, test_set_robot_groundtruth_positions)
            
            ecdf_filename_test_robot = "supervised_test_robot_ecdf.pdf"
            full_ecdf_path_test_robot = os.path.join(plots_output_dir, ecdf_filename_test_robot)
            CCEvaluation.plot_error_ecdf(test_set_robot_predictions, test_set_robot_groundtruth_positions, 
                                          outfile=full_ecdf_path_test_robot) # outfile adicionado
            print(f"ECDF plot salvo em: {full_ecdf_path_test_robot}")
            # --- FIM MODIFICAÇÃO ---
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        else:
            print("WARN: test_set_robot_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")


    print("Evaluating on Test Set (Human)...")
    if not isinstance(test_set_human_supervised, tf.data.Dataset) or test_set_human_supervised.cardinality().numpy() == 0:
        print("ERROR: test_set_human_supervised está vazio ou não é um Dataset TF válido. Pulando avaliação no conjunto de teste human.")
    else:
        test_set_human_predictions = supervised_model.predict(test_set_human_supervised.batch(1024))
        if test_set_human_groundtruth_positions.shape[0] > 0:
            errorvectors, errors, mae, cep = CCEvaluation.compute_localization_metrics(test_set_human_predictions, test_set_human_groundtruth_positions)
            suptitle = f"Evaluated on Test Set (Human)"
            title = f"Supervised: MAE = {mae:.3f}m, CEP = {cep:.3f}m"
            # --- MODIFICAÇÃO: Salvar plot em vez de mostrar ---
            plot_filename_test_human_scatter = f"supervised_test_human_scatter_mae{mae:.3f}_cep{cep:.3f}.png"
            full_plot_path_test_human_scatter = os.path.join(plots_output_dir, plot_filename_test_human_scatter)
            CCEvaluation.plot_colorized(test_set_human_predictions, test_set_human_groundtruth_positions, 
                                        suptitle=suptitle, title=title, 
                                        show=False, outfile=full_plot_path_test_human_scatter) # show=False, outfile adicionado
            print(f"Plot salvo em: {full_plot_path_test_human_scatter}")
            
            metrics = CCEvaluation.compute_all_performance_metrics(test_set_human_predictions, test_set_human_groundtruth_positions)
            # ECDF para o conjunto humano (se desejado, o original não plotava ECDF para humanos)
            ecdf_filename_test_human = "supervised_test_human_ecdf.pdf"
            full_ecdf_path_test_human = os.path.join(plots_output_dir, ecdf_filename_test_human)
            CCEvaluation.plot_error_ecdf(test_set_human_predictions, test_set_human_groundtruth_positions, 
                                          outfile=full_ecdf_path_test_human) # outfile adicionado
            print(f"ECDF plot salvo em: {full_ecdf_path_test_human}")
            # --- FIM MODIFICAÇÃO ---
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.upper().rjust(6, ' ')}: {metric_value:.3f}")
        else:
            print("WARN: test_set_human_groundtruth_positions está vazio. Não é possível calcular métricas de localização.")

    print(f"Plots do Supervised Baseline salvos em: {os.path.abspath(plots_output_dir)}")
    print("Script execution finished.")