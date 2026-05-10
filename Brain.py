import jax.numpy as jnp
import flax.linen as nn

class AutoencoderAUDITIA(nn.Module):
    # Definimos la longitud de entrada para asegurar que la salida sea idéntica
    dim_entrada: int 

    @nn.compact
    def __call__(self, x):
        # ----------------------------------------------------
        # 1. ENCODER (Fase de Compresión / Resumen)
        # ----------------------------------------------------
        # Vamos reduciendo las dimensiones gradualmente
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)  # Función de activación no lineal
        
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        
        # Cuello de botella: La "normalidad" comprimida en 64 números
        x = nn.Dense(features=64)(x)
        z = nn.relu(x)

        # ----------------------------------------------------
        # 2. DECODER (Fase de Reconstrucción)
        # ----------------------------------------------------
        # Empezamos a expandir de nuevo hacia la dimensión original
        x = nn.Dense(features=128)(z)
        x = nn.relu(x)
        
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # Capa de salida: Debe tener exactamente la misma dimensión que tu entrada original
        reconstruccion = nn.Dense(features=self.dim_entrada)(x)
        
        return reconstruccion
