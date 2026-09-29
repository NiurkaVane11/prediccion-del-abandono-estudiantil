"""
cliente.py
Lógica de la interfaz Streamlit, separada de la parte visual para poder
testearla: traduce lo que el usuario llena en el formulario a las 28
features que espera la API, y llama a la API en Render.
"""

import httpx

# Carreras en el orden de los códigos 1-17 del dataset. Verificado con los
# datos: los códigos 3 y 17 son 100% turno noche y el 12 (Enfermería) es la
# carrera más grande (766 estudiantes), igual que en el dataset original de UCI.
CARRERAS = {
    1: "Tecnologías de Producción de Biocombustibles",
    2: "Diseño de Animación y Multimedia",
    3: "Servicio Social (turno noche)",
    4: "Agronomía",
    5: "Diseño de Comunicación",
    6: "Enfermería Veterinaria",
    7: "Ingeniería Informática",
    8: "Equinocultura",
    9: "Administración",
    10: "Servicio Social",
    11: "Turismo",
    12: "Enfermería",
    13: "Higiene Oral",
    14: "Publicidad y Marketing",
    15: "Periodismo y Comunicación",
    16: "Educación Básica",
    17: "Administración (turno noche)",
}
CARRERAS_TURNO_NOCHE = {3, 17}

# Valores típicos (moda o mediana de los datos de entrenamiento) para las
# features que el formulario no pregunta: códigos de ocupación/estudios de
# los padres, modalidad de ingreso, nacionalidad y contexto macroeconómico.
VALORES_POR_DEFECTO = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 27,
    "Mother's occupation": 10,
    "Father's occupation": 10,
    "International": 0,
    "Educational special needs": 0,
    "Unemployment rate": 11.1,
    "Inflation rate": 1.4,
    "GDP": 0.32,
}


class ErrorAPI(Exception):
    """Error con un mensaje listo para mostrar al usuario."""


def construir_features(
    carrera: int,
    edad: int,
    genero_masculino: bool,
    desplazado: bool,
    becado: bool,
    deudor: bool,
    matricula_al_dia: bool,
    materias_inscritas: int,
    materias_evaluadas: int,
    materias_aprobadas: int,
    materias_convalidadas: int,
    materias_sin_evaluacion: int,
    nota_promedio: float,
) -> dict:
    """Arma el diccionario de 28 features con los nombres exactos del modelo."""
    if carrera not in CARRERAS:
        raise ValueError(f"Carrera desconocida: {carrera}")
    if materias_aprobadas > materias_inscritas:
        raise ValueError("No puede haber más materias aprobadas que inscritas")

    return {
        **VALORES_POR_DEFECTO,
        "Course": carrera,
        # El turno se deduce de la carrera: así no puede quedar inconsistente
        "Daytime/evening attendance": 0 if carrera in CARRERAS_TURNO_NOCHE else 1,
        "Age at enrollment": edad,
        "Gender": int(genero_masculino),
        "Displaced": int(desplazado),
        "Scholarship holder": int(becado),
        "Debtor": int(deudor),
        "Tuition fees up to date": int(matricula_al_dia),
        "Curricular units 1st sem (enrolled)": materias_inscritas,
        "Curricular units 1st sem (evaluations)": materias_evaluadas,
        "Curricular units 1st sem (approved)": materias_aprobadas,
        "Curricular units 1st sem (credited)": materias_convalidadas,
        "Curricular units 1st sem (without evaluations)": materias_sin_evaluacion,
        "Curricular units 1st sem (grade)": nota_promedio,
    }


def predecir(features: dict, api_url: str, api_key: str, cliente: httpx.Client | None = None) -> dict:
    """
    Llama a POST /predecir. Traduce los errores HTTP a mensajes claros.
    El timeout es largo porque Render (plan gratis) puede tardar ~50 s en
    "despertar" el servidor después de un rato sin uso.
    """
    cliente = cliente or httpx.Client(timeout=90)
    try:
        respuesta = cliente.post(
            f"{api_url.rstrip('/')}/predecir",
            json={"features": features},
            headers={"X-API-Key": api_key},
        )
    except httpx.TimeoutException:
        raise ErrorAPI("La API tardó demasiado en responder. Inténtalo de nuevo en un momento.")
    except httpx.HTTPError:
        raise ErrorAPI("No se pudo conectar con la API.")

    if respuesta.status_code == 200:
        return respuesta.json()
    if respuesta.status_code == 401:
        raise ErrorAPI("La app no está autorizada para usar la API (revisar API_KEY en los secrets).")
    if respuesta.status_code == 429:
        raise ErrorAPI("Demasiadas consultas seguidas. Espera un minuto y vuelve a intentar.")
    if respuesta.status_code == 422:
        raise ErrorAPI("La API rechazó los datos enviados.")
    raise ErrorAPI(f"Error inesperado de la API (código {respuesta.status_code}).")
