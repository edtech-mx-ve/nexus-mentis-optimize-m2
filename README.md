# NEXUS MENTIS — NEXUS OPTIMIZE (M2)

## Repositorios y alcance de las mediciones

- **Repositorio del Módulo 1:** [NEXUS ANALYZE — Código fuente y documentación en GitHub](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1).
- **Repositorio del Módulo 2:** [NEXUS OPTIMIZE — Código fuente y documentación en GitHub](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2).

**Alcance de las mediciones:** Las comparaciones HTTP se realizaron dentro de un servicio local; sus resultados no garantizan la misma velocidad en otros equipos o despliegues.


**Módulo 2 · Optimización de inferencia y tiempo de respuesta**

Proyecto académico de la **Maestría en Inteligencia Artificial para la Transformación Digital** del **Instituto Internacional de Aguascalientes**, para la asignatura *Tópicos de IA para la Transformación Digital*.

> **Práctica 2.** Analizar, implementar y evaluar técnicas de inteligencia artificial aplicadas a un dataset de redes sociales para optimizar el tiempo de respuesta.

## ¿Qué hace este módulo?

**NEXUS ANALYZE (M1)** desarrolló un clasificador binario de sentimientos sobre tweets de **Sentiment140** mediante procesamiento del lenguaje natural (NLP), TF-IDF y regresión logística. **NEXUS OPTIMIZE (M2)** compara dos formas de ejecutar **el mismo modelo entrenado**:

- **Ruta original M1:** normaliza el tweet, construye la representación TF-IDF y aplica el clasificador.
- **Ruta optimizada M2:** agiliza la normalización y fusiona el cómputo TF-IDF con la puntuación logística, evitando materializar toda la matriz intermedia de características.

**El modelo no se reentrena ni se modifican sus coeficientes.** La equivalencia de etiquetas se comprobó en las pruebas realizadas; no constituye una garantía formal para cualquier entrada.

La interfaz explica las diferencias entre ambas rutas, permite consultar resultados de rendimiento medidos previamente y ofrece una prueba interactiva de clasificación.

**Módulo original:** [NEXUS ANALYZE (M1)](https://nexus-mentis-analyze-m1.onrender.com/) · [Repositorio de M1](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1).

## Resultados de rendimiento

En un escenario experimental **HTTP local**, con **10.000 tweets reales y únicos**, **4 solicitudes simultáneas** y **muestra A**:

| Indicador | M1 original | M2 optimizado | Variación |
|---|---:|---:|---:|
| Latencia HTTP p95 por solicitud | 47,34 ms | 40,60 ms | **−14,25 %** |
| Capacidad de procesamiento | 26.256 tweets/s | 32.756 tweets/s | **+24,75 %** |
| Etiquetas de sentimiento | Modelo M1 | Equivalentes en las pruebas | Sin reentrenamiento |

La evaluación incluyó dos muestras de textos distintos. **Los 10 escenarios principales evaluados mejoraron**, sin regresiones principales en ese protocolo; los casos sin suficientes solicitudes concurrentes se excluyeron. La interfaz consulta un reporte agregado y verificado, no ejecuta los benchmarks al cambiar los controles.

**Interpretación:** la latencia p95 indica el tiempo que no superó el 95 % de las solicitudes HTTP; los tweets/s describen el rendimiento del volumen procesado. Estas cifras **no son resultados medidos en Render**, ni una garantía para otras máquinas, tasas de tráfico o condiciones de producción.

**Referencia de calidad predictiva heredada de M1:** F1 macro histórico en el conjunto reservado de prueba: **79,97 %** (29.400 tweets). M2 optimiza la ejecución, no la exactitud del modelo.

## ¿Qué aportó el Algoritmo Genético?

Se implementaron y evaluaron búsquedas evolutivas de configuraciones de procesamiento (selección, cruce, mutación y evaluación). **No se demostró una ventaja adicional reproducible frente a la ruta de cálculo fusionado.** El Algoritmo Genético pertenece al proceso experimental; **no se ejecuta en cada solicitud de clasificación**.

## Características de la aplicación

- Comparación didáctica de latencia p95 y tweets/s para los volúmenes y concurrencias realmente evaluados.
- Explicación gráfica de la arquitectura original M1 frente a la optimizada M2.
- Prueba interactiva de ambas rutas y comprobación de coincidencia de etiquetas.
- Pestañas **Acerca de, Dataset, Fundamentos, Trivia, Ayuda e Institucional**.
- API FastAPI, interfaz responsiva y documentación Swagger disponible en modo local.
- Modo de demostración pública con endpoints de investigación ocultos, entrada acotada y limitación básica de solicitudes.

## Estructura del repositorio público

Este repositorio contiene exclusivamente los archivos necesarios
para desplegar la demostración académica:

- `nexus_optimize/`: API FastAPI, inferencia y aplicación web.
- `reference/`: modelo original M1 y manifiesto de integridad.
- `public_reports/`: reporte E14 público y verificado.
- `Dockerfile`: construcción de la imagen.
- `render.yaml`: configuración de Render.
- `requirements.deploy.txt`: dependencias de ejecución.
- `README.md`: documentación del proyecto.

Los datasets, experimentos, pruebas automatizadas, scripts
auxiliares y respaldos permanecen en el entorno local de desarrollo.

## Despliegue

La aplicación se despliega desde GitHub mediante Docker
utilizando `render.yaml`.

La imagen comprueba la integridad del modelo y la presencia
del reporte público antes de iniciar el servicio.

La validación local no garantiza que el despliegue en Render
tenga idéntico rendimiento o disponibilidad.
## Publicación y seguridad

El repositorio puede conectarse a Render mediante el `Dockerfile` y `render.yaml`. **La publicación real debe verificarse** en la plataforma de destino; no se infiere de las pruebas locales.

Antes de cada publicación, comprobar que los archivos excluidos por `.gitignore` no lleguen a GitHub: `.env`, `.venv`, datasets completos, resultados de experimentación, credenciales y copias de respaldo. Conservar únicamente el reporte agregado público validado.

La limitación de solicitudes incluida es **básica y por proceso**: no equivale a autenticación, WAF ni protección integral frente a ataques de denegación de servicio. Las mediciones históricas de rendimiento no caracterizan automáticamente un despliegue gratuito en Render.

## Dataset y referencias

- **Sentiment140:** corpus histórico de 1.600.000 tweets en inglés; el trabajo original de M1 preparó 195.997 registros de una muestra inicial de 200.000.
- **Publicación académica:** Go, A.; Bhayani, R.; Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*.
- **Origen del dataset:** [Stanford — Sentiment140](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip).

## Créditos académicos

**Alumno:** Antonio Nicolás Toro González
**Tutor:** Dr. Francisco Javier Luna Rosas
**Institución:** Instituto Internacional de Aguascalientes
**Programa:** Maestría en Inteligencia Artificial para la Transformación Digital
**Asignatura:** Tópicos de IA para la Transformación Digital

---

**NEXUS MENTIS · Módulo 2:** mejorar el tiempo de respuesta de la clasificación de sentimientos conservando el modelo predictivo original.
