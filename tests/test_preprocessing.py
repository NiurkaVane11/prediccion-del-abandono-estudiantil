import pytest
import numpy as np
from preprocessing import preparar_input


class ScalerFalso:
    """Simula un StandardScaler: resta 0 y no cambia nada, solo para
    verificar que preparar_input arma el DataFrame en el orden correcto."""
    def transform(self, df):
        return df.values


def test_preparar_input_ordena_columnas_correctamente():
    columnas_esperadas = ["a", "b", "c"]
    datos = {"c": 3, "a": 1, "b": 2}

    resultado = preparar_input(datos, ScalerFalso(), columnas_esperadas)

    assert list(resultado[0]) == [1, 2, 3]


def test_preparar_input_lanza_error_si_faltan_columnas():
    columnas_esperadas = ["a", "b", "c"]
    datos = {"a": 1, "b": 2}  # falta "c"

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        preparar_input(datos, ScalerFalso(), columnas_esperadas)


def test_preparar_input_devuelve_array_numpy():
    columnas_esperadas = ["x", "y"]
    datos = {"x": 5, "y": 10}

    resultado = preparar_input(datos, ScalerFalso(), columnas_esperadas)

    assert resultado.shape == (1, 2)