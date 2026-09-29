import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

# La API exige API_KEY al importarse. Definimos una clave falsa SOLO para
# los tests (antes de que test_api.py importe api.main), así los tests
# corren igual en local y en GitHub Actions sin exponer la clave real.
os.environ.setdefault("API_KEY", "clave-de-prueba")
