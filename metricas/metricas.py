import matplotlib.pyplot as plt
import re
import os

# Caminho do arquivo fornecido
log_path = "/home/amanda/Documentos/2025.4/lacis/VAR/FL-REST/fl_logs/simulation_20251217_150935.log"

def plot_federated_results(path):
    if not os.path.exists(path):
        print(f"Erro: O arquivo não foi encontrado em {path}")
        return

    rounds = []
    accuracies = []
    losses = []

    # Expressão regular para capturar os dados do servidor
    pattern = r"Round (\d+) Complete\. Accuracy: ([\d.]+)%, Loss: ([\d.]+)"

    with open(path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                rounds.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
                losses.append(float(match.group(3)))

    if not rounds:
        print("Nenhum dado de rodada (Round Complete) foi encontrado no log.")
        return

    # Criando a visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Resultados da Simulação Federada\nArquivo: {os.path.basename(path)}', fontsize=14)

    # Gráfico de Acurácia
    ax1.plot(rounds, accuracies, marker='o', color='#2ecc71', linestyle='-', linewidth=2, label='Acurácia Global')
    ax1.set_title('Evolução da Acurácia')
    ax1.set_xlabel('Rodada (Round)')
    ax1.set_ylabel('Acurácia (%)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Gráfico de Perda (Loss)
    ax2.plot(rounds, losses, marker='s', color='#e74c3c', linestyle='-', linewidth=2, label='Perda Global')
    ax2.set_title('Evolução da Perda (Loss)')
    ax2.set_xlabel('Rodada (Round)')
    ax2.set_ylabel('Valor da Perda')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Lógica para salvar na pasta 'metricas' ---
    
    # Pega o diretório pai de 'fl_logs' (que é 'FL-REST') e aponta para 'metricas'
    base_dir = os.path.dirname(os.path.dirname(path)) 
    output_dir = os.path.join(base_dir, "metricas")
    
    # Cria a pasta 'metricas' se ela não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define o nome do arquivo de saída baseado no nome do log
    file_name = os.path.splitext(os.path.basename(path))[0] + ".png"
    output_path = os.path.join(output_dir, file_name)
    
    # Salva a imagem
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico salvo com sucesso em: {output_path}")
    
    plt.show()

# Executar a função
plot_federated_results(log_path)