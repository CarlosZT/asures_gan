# Descripción
Este trabajo comprende el desarrollo de un modelo generativo cuyo objetivo es incrementar la resolución de un segmento de audio. El modelo recibe como entrada un audio en formato WAV con tasa de muestreo de 5512.5 Hz (22050/4 Hz), el cual es convertido a Mono y normalizado para un rango de valores de 0 a 1. Para realizar la inferencia de componentes de alta frecuencia, el modelo pretende aprovechar los patrones de frecuencia generados por el timbre del audio, disponibles en la banda de baja frecuencia.

# ¿Qué contiene este repositorio?
  · Modelo generativo y discriminador en formato tf para Tensorflow 2.x (generator_4_.tf; discriminator_4_.tf).
  
  · Carpeta de resultados.
  
    - Ejemplos de entrada de baja resolución.
    
    - Perfil de ruido del modelo parametrizado en 0, 0.5 y 1 para filtros de audio.
    
    - Audio generado por el modelo + versión filtrada.
    
  · Código fuente.
  
    - Notebook del proyecto para Anaconda o Google Colab (Python 3.7 + Tensorflow 2.x).
    
    - Script demostrativo del modelo.
    
