import numpy as np
import os
import jax
import jax.numpy as jnp

# Forzamos visibilidad del dispositivo 0
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

try:
    # Intentamos una operación simple para despertar a la MI300X
    _ = jax.device_put(jnp.ones(1))
    print(f"🚀 MI300X despertada con éxito: {jax.devices()}")
except Exception as e:
    print(f"⚠️ Error al despertar la GPU: {e}")
    print("🔄 Cayendo a modo CPU para diagnóstico...")
    os.environ["JAX_PLATFORMS"] = "cpu"

# Ahora sí, el resto de la carga
rng = jax.random.PRNGKey(0)
import flax.serialization
import joblib
from sentence_transformers import SentenceTransformer
from Brain import AutoencoderAUDITIA

print("⚙️ [AUDITIA] Levantando modelos y pesos en memoria")

# ==========================================
# 1. CARGA EN MEMORIA (ESTADO GLOBAL)
# ==========================================
# A. Reglas Matemáticas
try:
    scaler = joblib.load('auditia_scaler_numerico.joblib')
    encoder = joblib.load('auditia_encoder_categorico.joblib')
except FileNotFoundError as e:
    raise RuntimeError(f"Falta un archivo de reglas matemáticas: {e}")

# B. Modelo de Lenguaje (MiniLM en GPU)
modelo_texto = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',device='cpu')

# C. Cerebro JAX/Flax (Autoencoder)
DIM_VECTOR = 460
arquitectura = AutoencoderAUDITIA(dim_entrada=DIM_VECTOR)
rng = jax.random.PRNGKey(0)
pesos_vacios = arquitectura.init(rng, jnp.ones((1, DIM_VECTOR)))

try:
    with open("pesos_auditia_autoencoder.msgpack", "rb") as f:
        pesos_entrenados = flax.serialization.from_bytes(pesos_vacios, f.read())
except FileNotFoundError:
    raise RuntimeError("Falta el archivo de pesos: pesos_auditia_autoencoder.msgpack")

# D. Compilación XLA (Se compila una vez, se usa infinitas veces)
@jax.jit
def _calcular_mse_compilado(pesos, x):
    reconstruccion = arquitectura.apply(pesos, x)
    return jnp.mean(jnp.square(x - reconstruccion))

print("✅ [AUDITIA] Motor listo y a la espera de contratos.")

# ==========================================
# 2. FUNCIÓN PÚBLICA DE EVALUACIÓN
# ==========================================
def evaluar_contrato(datos_contrato: dict) -> float:
    """
    Recibe un diccionario con los datos crudos del contrato, 
    lo vectoriza en tiempo real y devuelve el Score de Anomalía (MSE).
    """
    # 1. Extraer y procesar Texto
    texto = datos_contrato.get('objeto_contrato', '')
    vector_texto = modelo_texto.encode([texto])[0]

    # 2. Extraer y procesar Números
    valor = datos_contrato.get('valor_contrato', 0.0)
    duracion = datos_contrato.get('duracion_numerica', 0.0)
    # Le damos forma 2D (1 fila, 2 columnas) para el scaler
    valores_num = np.array([[valor, duracion]])
    vector_num = scaler.transform(valores_num)[0]

    # 3. Extraer y procesar Categorías
    depto = datos_contrato.get('departamento', 'Desconocido')
    tipo = datos_contrato.get('tipo_contrato', 'Desconocido')
    modalidad = datos_contrato.get('modalidad_contratacion', 'Desconocido')
    # Le damos forma 2D (1 fila, 3 columnas) para el encoder
    valores_cat = np.array([[depto, tipo, modalidad]])
    vector_cat = encoder.transform(valores_cat)[0]

    # 4. Ensamblaje Final (Debe sumar 460 dimensiones)
    vector_final = np.concatenate((vector_num, vector_cat, vector_texto))
    vector_final = vector_final.reshape(1, -1)

    # 5. Inferencia en la RTX 5070
    vector_gpu = jax.device_put(vector_final)
    score = _calcular_mse_compilado(pesos_entrenados, vector_gpu)
    
    # JAX devuelve un array de 0 dimensiones, usamos .item() para convertirlo a float estándar de Python
    return float(score.item())
