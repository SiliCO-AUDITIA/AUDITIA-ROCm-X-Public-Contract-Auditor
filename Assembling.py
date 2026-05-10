import polars as pl
import numpy as np
import joblib  # Para guardar nuestras reglas matemáticas
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("🚀 Iniciando Ensamblaje Maestro de AUDITIA...")

# ==========================================
# 1. CARGAR DATOS BASE
# ==========================================
print("📂 Cargando dataset limpio...")
df = pl.read_parquet('Data-Limpia.parquet')

# ==========================================
# 2. PROCESAMIENTO NUMÉRICO (Z-SCORE)
# ==========================================
print("🔢 Calculando Z-Scores para variables financieras y de tiempo...")
columnas_numericas = ['valor_contrato', 'duracion_numerica']

# Extraemos como matriz de NumPy directamente para ahorrar RAM
X_num = df.select(columnas_numericas).fill_null(0).to_numpy()

# El StandardScaler calcula el promedio y desviación, y transforma los datos
scaler = StandardScaler()
X_num_escalado = scaler.fit_transform(X_num)

# ¡CRÍTICO! Guardamos este "molde" matemático para usarlo en inferencia
joblib.dump(scaler, 'auditia_scaler_numerico.joblib')
print("✅ Scaler numérico guardado en disco.")

# ==========================================
# 3. PROCESAMIENTO CATEGÓRICO (ONE-HOT)
# ==========================================
print("🔠 Vectorizando categorías...")
# Puedes ajustar estas columnas según las que hayas dejado en tu limpieza
columnas_categoricas = ['departamento', 'tipo_contrato', 'modalidad_contratacion']

# Rellenamos posibles nulos con 'Desconocido'
X_cat = df.select(columnas_categoricas).fill_null('Desconocido').to_numpy()

# Creamos el codificador One-Hot
# sparse_output=False nos devuelve una matriz NumPy clásica
# handle_unknown='ignore' evita errores si en inferencia llega un departamento nuevo
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_codificado = encoder.fit_transform(X_cat)

# Guardamos el mapeo de categorías
joblib.dump(encoder, 'auditia_encoder_categorico.joblib')
print(f"✅ Encoder guardado. Se generaron {X_cat_codificado.shape[1]} columnas binarias.")

# ==========================================
# 4. CARGAR TEXTOS Y ENSAMBLAR
# ==========================================
print("🧠 Cargando los embeddings de texto previos (384 dimensiones)...")
# Cargamos el archivo que ya procesaste antes
X_texto = np.load('Embeddings_Textos.npy')

print("🔗 Ensamblando la matriz definitiva...")
# hstack pega las matrices horizontalmente (columnas al lado de columnas)
Vectores_Completos = np.hstack((X_num_escalado, X_cat_codificado, X_texto))

print(f"📏 Dimensión final de cada contrato: {Vectores_Completos.shape[1]} características.")

# ==========================================
# 5. GUARDAR EL ARCHIVO MAESTRO
# ==========================================
print("💾 Guardando Vectores_Completos.npy en el disco...")
np.save('Vectores_Completos.npy', Vectores_Completos)

print("-" * 50)
print("🎉 ¡Ensamblaje completado con éxito!")
print(f"Forma total de la matriz: {Vectores_Completos.shape}")
print("-" * 50)
