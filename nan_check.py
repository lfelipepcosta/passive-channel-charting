import numpy as np
import os
import espargos_0007 # Importa as definições do seu dataset

def check_aoa_data():
    """
    Este script verifica os arquivos de 'aoa_angles' e 'aoa_powers'
    gerados pelo ESPRIT e conta o número de valores NaN em cada um.
    """
    
    aoa_dir = "aoa_estimates_ESPRIT"
    training_files = espargos_0007.TRAINING_SET_ROBOT_FILES
    
    # Adicionamos os dois tipos de arquivo que queremos verificar
    types_to_check = ["angles", "powers"]
    
    print(f"--- Verificando arquivos no diretório: {aoa_dir} ---\n")
    
    if not os.path.exists(aoa_dir):
        print(f"ERRO: O diretório '{aoa_dir}' não foi encontrado.")
        return

    # Itera sobre os tipos de arquivo (angles e powers)
    for file_type in types_to_check:
        print(f"==================[ Verificando: aoa_{file_type} ]==================")
        total_nans = 0
        total_values = 0

        # Itera sobre os arquivos de dados que compõem o conjunto de treino
        for filepath in training_files:
            original_filename = os.path.basename(filepath)
            target_filename = f"{original_filename}.aoa_{file_type}.npy"
            target_filepath = os.path.join(aoa_dir, target_filename)
            
            print(f"Analisando arquivo: {target_filename}")
            
            if os.path.exists(target_filepath):
                data_array = np.load(target_filepath)
                nan_count = np.isnan(data_array).sum()
                num_values = data_array.size
                
                total_nans += nan_count
                total_values += num_values
                
                print(f"  -> Valores NaN encontrados: {nan_count}")
            else:
                print(f"  -> ARQUIVO NÃO ENCONTRADO.")
        
        print(f"\n--- Resumo para aoa_{file_type} ---")
        print(f"Total de valores NaN encontrados: {total_nans}")
        print(f"Total de valores nos arquivos: {total_values}")
        if total_values > 0:
            percentage = (total_nans / total_values) * 100
            print(f"Porcentagem de NaN: {percentage:.2f}%\n")

if __name__ == "__main__":
    check_aoa_data()