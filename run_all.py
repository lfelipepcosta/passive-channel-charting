import os
import subprocess
import datetime

NUM_ROUNDS = 3
BASE_LOGS_DIR = "experiment_logs"

ALGORITHMS = {
    "1": {"name": "Unitary Root-MUSIC", "script": "3_AoA_Estimation.py", "arg_name": "unitary root music"},
    "2": {"name": "MUSIC", "script": "3_AoA_Estimation_MUSIC.py", "arg_name": "music"},
    "3": {"name": "ESPRIT", "script": "3_AoA_Estimation_ESPRIT.py", "arg_name": "esprit"},
    "4": {"name": "Delay and Sum", "script": "3_AoA_Estimation_DAS.py", "arg_name": "delay and sum"},
    "5": {"name": "Capon", "script": "3_AoA_Estimation_CAPON.py", "arg_name": "capon"},
    "6": {"name": "SS Capon", "script": "3_AoA_Estimation_SSCapon.py", "arg_name": "ss capon"}
}

def run_command(command, log_file=None):
    """Executa um comando no terminal, salvando a saída em um log e mostrando no console."""
    print(f"--- Executando: {command} ---")
    
    try:
        process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        with open(log_file, 'w') as f:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                f.write(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        print(f"--- Log salvo em: {log_file} ---")
        print(f"--- Concluído: {command} ---\n")

    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERRO AO EXECUTAR COMANDO !!!!!!")
        print(f"Comando que falhou: {e.cmd}")
        print(f"Veja o arquivo de log para detalhes: {log_file}")
        raise e

def main():

    print("Escolha o algoritmo para rodar o pipeline 5 vezes:")
    for key, config in ALGORITHMS.items():
        print(f"  {key}: {config['name']}")
    
    choice = input("Digite o número do algoritmo: ")
    
    if choice not in ALGORITHMS:
        print("Opção inválida. Saindo.")
        return
        
    selected_algo = ALGORITHMS[choice]
    name = selected_algo['name']
    arg_name = selected_algo['arg_name']
    aoa_script = selected_algo['script']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = os.path.join(BASE_LOGS_DIR, f"{arg_name.replace(' ', '_')}_{timestamp}")
    os.makedirs(run_logs_dir, exist_ok=True)
    print(f"Logs para esta execução serão salvos em: {run_logs_dir}")

    for i in range(1, NUM_ROUNDS + 1):
        print(f"\n\n==================[ INICIANDO RODADA {i}/{NUM_ROUNDS} PARA: {name.upper()} ]==================")
        round_num = str(i)
        
        try:
            print("\n--- Etapa 3: Estimação de AoA ---")
            log_path_3 = os.path.join(run_logs_dir, f"3_aoa_round_{round_num}.txt")
            run_command(f"python {aoa_script} {round_num}", log_file=log_path_3)

            print("\n--- Etapa 4: Triangulação ---")
            log_path_4 = os.path.join(run_logs_dir, f"4_triangulation_round_{round_num}.txt")
            run_command(f"python 4_Triangulation.py \"{arg_name}\" {round_num}", log_file=log_path_4)

            print("\n--- Etapa 5: Matriz de Dissimilaridade ---")
            log_path_5 = os.path.join(run_logs_dir, f"5_dissimilarity_matrix_round_{round_num}.txt")
            run_command(f"python 5_DissimilarityMatrix.py \"{arg_name}\" {round_num}", log_file=log_path_5)
            
            print("\n--- Etapa 6: Channel Charting ---")
            log_path_6 = os.path.join(run_logs_dir, f"6_channel_charting_round_{round_num}.txt")
            run_command(f"python 6_ChannelCharting.py \"{arg_name}\" {round_num}", log_file=log_path_6)

        except subprocess.CalledProcessError:
            print(f"!!!!!! A RODADA {i} FALHOU. Interrompendo a execução para o algoritmo {name.upper()}. !!!!!!")
            break

    print(f"\n\n==================[ EXPERIMENTO PARA {name.upper()} CONCLUÍDO ]==================")

if __name__ == "__main__":
    main()