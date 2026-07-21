import nbformat
from nbclient import NotebookClient

notebook_path = "projeto semantix churn.ipynb"
print(f"Lendo o notebook '{notebook_path}'...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Executando todas as células do notebook...")
client = NotebookClient(nb, timeout=1200, kernel_name="python3")

try:
    client.execute()
    print("Notebook executado com sucesso!")
except Exception as e:
    print(f"Ocorreu um erro durante a execução: {e}")

print(f"Salvando o notebook atualizado em '{notebook_path}'...")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Concluído!")
