import numpy as np
import os
import espargos_0007 # Importa as definições do seu dataset

def check_power_values():
    """
    Verifica os arquivos ...aoa_powers.npy para múltiplos algoritmos,
    procurando por valores máximos e contagens de valores > 1.0.
    """
    
    # Dicionário com os nomes de exibição e os diretórios de cada algoritmo
    algorithms_to_check = {
        "Unitary Root-MUSIC": "aoa_estimates",
        "MUSIC": "aoa_estimates_MUSIC",
        "ESPRIT": "aoa_estimates_ESPRIT",
        "Delay and Sum": "aoa_estimates_DAS",
        "Capon": "aoa_estimates_CAPON",
        "SS Capon": "aoa_estimates_SSCAPON"
    }
    
    training_files = espargos_0007.TRAINING_SET_ROBOT_FILES
    
    print("--- Verificando os valores de 'aoa_powers' para cada algoritmo ---\n")
    
    # Itera sobre cada algoritmo no dicionário
    for display_name, aoa_dir in algorithms_to_check.items():
        print(f"==================[ Algoritmo: {display_name} ]==================")
        
        if not os.path.exists(aoa_dir):
            print(f"Diretório '{aoa_dir}' não encontrado. Pulando.\n")
            continue
            
        total_values_above_one = 0
        max_power_found = -np.inf # Inicia com o menor valor possível
        
        # Constrói a lista de arquivos .npy a serem verificados
        power_files_to_check = [
            os.path.join(aoa_dir, os.path.basename(f) + ".aoa_powers.npy")
            for f in training_files
        ]
        
        files_found = 0
        for power_filepath in power_files_to_check:
            if os.path.exists(power_filepath):
                files_found += 1
                powers = np.load(power_filepath)
                
                # Remove NaNs antes de calcular o máximo para evitar warnings
                valid_powers = powers[~np.isnan(powers)]
                if valid_powers.size > 0:
                    current_max = np.max(valid_powers)
                    if current_max > max_power_found:
                        max_power_found = current_max
                
                # Conta quantos valores são maiores que 1.0
                total_values_above_one += np.sum(valid_powers > 1.0)
        
        if files_found > 0:
            if np.isinf(max_power_found):
                print("  -> Valor Máximo Encontrado: N/A (nenhum valor válido encontrado)")
            else:
                print(f"  -> Valor Máximo Encontrado: {max_power_found:.6f}")
            
            print(f"  -> Total de valores > 1.0: {total_values_above_one}")
        else:
            print("  -> Nenhum arquivo .npy de 'powers' encontrado para este algoritmo.")
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    check_power_values()