import sys
import nbformat
from nbclient import NotebookClient

notebook_path = "projeto_semantix_churn.ipynb"
print(f"Lendo o notebook '{notebook_path}'...")

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
except Exception as e:
    print(f"Erro ao abrir o arquivo do notebook '{notebook_path}': {e}")
    sys.exit(1)

print("Executando todas as células do notebook...")
client = NotebookClient(nb, timeout=1200, kernel_name="python3")

try:
    client.execute()
    print("Notebook executado com sucesso!")
except Exception as e:
    print(f"FALHA CRÍTICA: Ocorreu um erro durante a execução das células: {e}")
    sys.exit(1)

print(f"Salvando o notebook atualizado em '{notebook_path}'...")
try:
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Concluído!")
except Exception as e:
    print(f"Erro ao salvar o notebook atualizado: {e}")
    sys.exit(1)
