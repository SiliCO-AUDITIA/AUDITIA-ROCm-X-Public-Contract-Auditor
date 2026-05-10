import os
import sys

# --- 1. CONFIGURACIÓN ESTRICTA DE ROCm / MI300X ---
os.environ["JAX_PLATFORMS"] = "rocm"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Parche para Docker: Evita que MIOpen falle por no tener permisos de escritura en la caché
os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/miopen-db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/tmp/miopen-cache"

import jax
import jax.numpy as jnp

# --- 2. INICIALIZACIÓN BLINDADA DE JAX (DEBE IR ANTES QUE PYTORCH) ---
print("⚙️ [AUDITIA] Despertando núcleos de la MI300X...")
try:
    dispositivos = jax.devices()
    print(f"🚀 MI300X detectada: {dispositivos}")
    
    # Forzamos la inicialización de MIOpen AQUÍ, cuando JAX es el único dueño absoluto
    rng = jax.random.PRNGKey(0)
    _ = jax.device_put(jnp.ones(1))
    print("✅ Motor DNN (MIOpen) inicializado y asegurado.")
except Exception as e:
    print(f"❌ Error crítico de hardware: {e}")
    sys.exit(1)

# --- 3. AHORA SÍ: IMPORTAR EL RESTO DE LIBRERÍAS ---
print("📚 Cargando IA de Lenguaje...")
from sentence_transformers import SentenceTransformer
modelo_texto = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

from flax.serialization import from_bytes
import joblib
from fastapi import FastAPI
import uvicorn

# IMPORTACIÓN MAESTRA: Traemos la arquitectura original directamente de Brain.py
from Brain import AutoencoderAUDITIA

# --- 4. CARGA EN CALIENTE ---
print("⚖️ Levantando reglas y pesos en VRAM...")
scaler = joblib.load('auditia_scaler_numerico.joblib')
encoder = joblib.load('auditia_encoder_categorico.joblib')

DIM_VECTOR = 460
# Usamos la clase importada
modelo_ae = AutoencoderAUDITIA(dim_entrada=DIM_VECTOR)
variables = modelo_ae.init(rng, jnp.ones((1, DIM_VECTOR)))

with open("pesos_auditia_autoencoder.msgpack", "rb") as f:
    pesos_entrenados = from_bytes(variables, f.read())

print("🟢 AUDITIA LISTO PARA PRODUCCIÓN")
# --- 5. CARGA EN CALIENTE ---
print("⚖️ Levantando reglas y pesos en VRAM...")
scaler = joblib.load('auditia_scaler_numerico.joblib')
encoder = joblib.load('auditia_encoder_categorico.joblib')

DIM_VECTOR = 460
modelo_ae = AutoencoderAUDITIA(dim_entrada=DIM_VECTOR)
variables = modelo_ae.init(rng, jnp.ones((1, DIM_VECTOR)))

with open("pesos_auditia_autoencoder.msgpack", "rb") as f:
    pesos_entrenados = from_bytes(variables, f.read())

print("🟢 AUDITIA LISTO PARA PRODUCCIÓN")
def vectorizar_contrato_api(datos: dict):
    """
    Transforma un contrato individual en el vector de 460 dimensiones.
    """
    # A. Procesamiento de Texto (384 dimensiones) [cite: 1016]
    # Se genera el embedding semántico del objeto del contrato
    emb_texto = modelo_texto.encode([datos['objeto_contrato']]) 
    
    # B. Procesamiento Numérico (2 dimensiones) [cite: 1007]
    # Aplicamos el Z-score usando el scaler que ya conoce el promedio y desviación histórica
    datos_num = [[datos['valor_contrato'], datos['duracion_numerica']]]
    num_escalado = scaler.transform(datos_num)
    
    # C. Procesamiento Categórico (74 dimensiones) [cite: 1012]
    # Aplicamos One-Hot Encoding para las categorías de departamento y tipo
    # Se usa .toarray() porque JAX requiere matrices densas
    cat_datos = [[datos['departamento'], datos['tipo_contrato'], datos['modalidad_contratacion']]]
    cat_enc = encoder.transform(cat_datos).toarray()
    
    # D. Ensamblaje Maestro (Concatenación Horizontal) [cite: 1017]
    # Es CRÍTICO mantener el mismo orden que se usó en el entrenamiento
    vector_final = jnp.hstack((emb_texto, num_escalado, cat_enc))
    
    return vector_final

# Aquí debajo ya va tu código de FastAPI...
# app = FastAPI()
# @app.post("/auditar")
# ... etc.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="AUDITIA API - JAX/Flax Inferencia")

# Definición del esquema de datos que recibirá el endpoint
class ContratoEntrada(BaseModel):
    id_contrato: str
    objeto_contrato: str
    valor_contrato: float
    departamento: str
    duracion_numerica: float
    # Puedes añadir más campos aquí según tu vector de 460 dimensiones [cite: 690]

@app.post("/auditar-contrato/")
async def api_auditar_contrato(contrato: ContratoEntrada):
    try:
        # 1. Convertimos el JSON a vector de 460 posiciones
        x = vectorizar_contrato_api(contrato.dict())
        
        # 2. Inferencia con JAX/Flax (XLA) [cite: 1431, 1433]
        # x ya está en el formato correcto para modelo_ae.apply
        reconstruccion = modelo_ae.apply(pesos_entrenados, x)
        
        # 3. Cálculo del Score de Anomalía (Error de Reconstrucción) [cite: 1181]
        score = float(jnp.mean(jnp.square(x - reconstruccion)))
        
        return {
            "id_contrato": contrato.id_contrato,
            "score_riesgo": score,
            "alerta": score > 0.05  # Umbral definido por tu entrenamiento [cite: 1182]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en vectorización: {str(e)}")
