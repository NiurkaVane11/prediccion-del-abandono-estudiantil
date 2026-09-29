"""
subir_a_hf.py
Despliega la API en un Hugging Face Space (tipo Docker). Lo ejecuta
GitHub Actions después de que pasan los tests y el build de Docker.

Variables de entorno necesarias:
    HF_TOKEN      token de Hugging Face con permiso de escritura
    HF_SPACE_ID   id del Space, ej: "usuario/prediccion-abandono"
    API_KEY       clave que exigirá la API desplegada (se guarda como
                  secret del Space, nunca en el código)
"""

import os

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Solo lo necesario para correr la API. data/, notebooks/ y tests/ no se
# suben: no hacen falta en producción y así el Space queda más liviano.
ARCHIVOS_SUELTOS = ["Dockerfile", "requirements.txt"]
CARPETAS = ["api", "src", "modelos"]


def listar_archivos() -> dict:
    """Devuelve {ruta_en_el_space: ruta_local} de todo lo que se sube."""
    archivos = {nombre: os.path.join(ROOT_DIR, nombre) for nombre in ARCHIVOS_SUELTOS}
    for carpeta in CARPETAS:
        for raiz, dirs, nombres in os.walk(os.path.join(ROOT_DIR, carpeta)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for nombre in nombres:
                local = os.path.join(raiz, nombre)
                archivos[os.path.relpath(local, ROOT_DIR).replace(os.sep, "/")] = local
    # El README del Space lleva la configuración que Hugging Face necesita
    # (sdk: docker, puerto). Es distinto del README de GitHub.
    archivos["README.md"] = os.path.join(ROOT_DIR, "deploy", "README_space.md")
    return archivos


def main():
    space_id = os.environ["HF_SPACE_ID"]
    api = HfApi(token=os.environ["HF_TOKEN"])

    # Crea el Space la primera vez; si ya existe no hace nada
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(space_id, "API_KEY", os.environ["API_KEY"])

    archivos = listar_archivos()
    operaciones = [
        CommitOperationAdd(path_in_repo=destino, path_or_fileobj=origen)
        for destino, origen in archivos.items()
    ]
    # Borra del Space los archivos que ya no existen en el repo (ej: un
    # módulo renombrado), para que el Space sea copia fiel de GitHub.
    for existente in api.list_repo_files(space_id, repo_type="space"):
        if existente not in archivos and existente != ".gitattributes":
            operaciones.append(CommitOperationDelete(path_in_repo=existente))

    # Un solo commit = un solo rebuild del Space
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operaciones,
        commit_message=f"Deploy desde GitHub ({sha})",
    )
    print(f"Deploy enviado: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
