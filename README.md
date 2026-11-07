<!-- PORTADA_M2_INICIO -->
<p align="center">
  <img src="nexus_optimize/web/nexus-logo.webp" width="140" alt="Logotipo de NEXUS MENTIS">
</p>

<div align="center">

# 🧠 NEXUS MENTIS — NEXUS OPTIMIZE

### ⚡ Módulo 2 · Optimización del tiempo de respuesta en análisis de sentimientos

**NLP · TF-IDF · Machine Learning · Optimización de inferencia · Evaluación HTTP**

**🐍 Python 3.11 · ⚡ FastAPI · 🧮 scikit-learn · 🌐 HTML/CSS/JavaScript · 🐳 Docker**

[📁 GitHub — Módulo 2](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2) · [🧠 GitHub — Módulo 1](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1) · [🌐 Aplicación M1](https://nexus-mentis-analyze-m1.onrender.com/)

</div>

---

**Alcance de las mediciones:** Las comparaciones HTTP se realizaron dentro de un servicio local; sus resultados no garantizan la misma velocidad en otros equipos o despliegues.

<!-- PORTADA_M2_FIN -->
<p align="center">
  <img src="nexus_optimize/web/nexus-logo.webp" width="140" alt="Logotipo de NEXUS MENTIS">
</p>

<div align="center">

# 🧠 NEXUS MENTIS — NEXUS OPTIMIZE

### ⚡ Módulo 2 · Optimización del tiempo de respuesta en análisis de sentimientos

**Procesamiento del lenguaje natural · TF-IDF · Machine Learning · Optimización de inferencia · Evaluación HTTP reproducible**

**🐍 Python 3.11 · ⚡ FastAPI · 🧮 scikit-learn · 🌐 HTML/CSS/JavaScript · 🐳 Docker**

[📁 GitHub — Módulo 2](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2) · [🧠 GitHub — Módulo 1](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1) · [🌐 Aplicación M1 en Render](https://nexus-mentis-analyze-m1.onrender.com/)

</div>

---

## 1. 📍 Estado actual

**NEXUS OPTIMIZE** es el segundo módulo académico de **NEXUS MENTIS**. Su propósito es **reducir el tiempo de respuesta** de un clasificador de sentimientos ya entrenado, comparando la ejecución original del Módulo 1 (**M1**) con una ruta optimizada (**M2/E9**).

| Elemento | Estado del Módulo 2 |
|---|---|
| Proyecto | **NEXUS MENTIS — NEXUS OPTIMIZE** |
| Práctica académica | **Práctica 2: análisis, implementación y prueba de técnicas de IA a un dataset de redes sociales para optimizar el tiempo de respuesta** |
| Dataset de referencia | **Sentiment140** (tweets históricos en inglés) |
| Clasificador heredado de M1 | **TF-IDF + regresión logística** |
| Estado del modelo | **Congelado: sin reentrenamiento ni cambios en sus coeficientes** |
| Ruta de referencia | **M1**, vectorización TF-IDF e inferencia originales |
| Ruta optimizada | **E9**, normalización eficiente y cómputo TF-IDF + puntuación logística fusionados |
| Calidad predictiva histórica | **F1 macro 79,97 %** sobre **29.400 tweets** de prueba de M1 |
| Evidencia de rendimiento | **E13** (comparación HTTP local) y **E14** (conciliación de métricas) |
| Escenarios principales válidos | **10/10 con mejora**, dentro del protocolo local evaluado |
| Algoritmo Genético | **Implementado y evaluado; sin mejora adicional reproducible frente a E9** |
| Verificación local reportada | **221 pruebas aprobadas** e **`integrity_ok`** |
| Publicación del código | [Repositorio GitHub M2](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2) |
| Despliegue M2 en Render | **Pendiente de verificación pública de arranque, API e inferencia** |

> **Distinción fundamental:** el **79,97 %** mide calidad predictiva histórica del modelo M1; las mejoras de M2 miden **tiempo de respuesta y capacidad de procesamiento**. Son preguntas experimentales diferentes.

---

## 2. 🎯 Objetivo y alcance

**Objetivo:** analizar, implementar y evaluar técnicas de optimización de un sistema de clasificación de sentimientos para reducir su tiempo de respuesta, manteniendo como referencia el mismo modelo entrenado en el Módulo 1.

**Alcance del Módulo 2:**

- Reutilización y auditoría del artefacto original de M1.
- Establecimiento de una línea base y separación entre mediciones de inferencia local y HTTP.
- Optimización de normalización, representación TF-IDF y cálculo de clasificación.
- Exploración de técnicas de caché, deduplicación y búsqueda mediante **Algoritmo Genético (AG)**.
- Evaluación de rutas M1 y M2 sobre **tweets únicos**, sin atribuir a la optimización ganancias derivadas de duplicados artificiales.
- Comparación controlada de **latencia p95**, **tweets por segundo**, concurrencia y coincidencia de etiquetas.
- Presentación didáctica de resultados históricos y demostración interactiva.

**Fuera de alcance:** reentrenar el clasificador, cambiar sus coeficientes, recolectar X en tiempo real, procesar tráfico distribuido o demostrar rendimientos garantizados en producción.

---

## 3. 🖥️ ¿Qué hace la aplicación?

La interfaz reúne la explicación técnica, las métricas verificadas y una demostración de clasificación.

| Área | Función |
|---|---|
| **📊 Comparación M1 vs. M2** | Muestra latencia HTTP p95 y rendimiento para escenarios incluidos en el reporte E14 |
| **⚙️ Técnica de optimización** | Explica el procesamiento original frente al cómputo fusionado E9 |
| **🧪 Clasificar tweets** | Envía textos al modelo mediante las rutas M1 y E9 y contrasta sus etiquetas |
| **📂 Dataset** | Expone el origen y las restricciones del corpus Sentiment140 |
| **📚 Fundamentos** | Introduce NLP, TF-IDF, regresión logística, algoritmos de búsqueda y las 3V de Big Data |
| **ℹ️ Información y apoyo** | Pestañas **Acerca de, Dataset, Fundamentos, Trivia, Ayuda e Institucional** |

Los selectores de volumen, concurrencia y cohorte **consultan resultados ya medidos**. Cambiarlos **no ejecuta un nuevo benchmark** ni mide la infraestructura del visitante.

---

## 4. 🔁 Flujo de inferencia: M1 frente a M2

```text
                     Tweet en inglés
                            │
                            ▼
                   Validación de entrada
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
          M1 · Referencia        M2 · E9 optimizado
                  │                   │
          Normalización         Normalización eficiente
                  │                   │
          Vectorizador          Cálculo fusionado:
          TF-IDF original       TF-IDF + puntuación
                  │             logística
          Matriz de                  │
          características             │
                  │                   │
          Regresión logística         │
                  │                   │
                  └─────────┬─────────┘
                            ▼
                 Sentimiento positivo
                      o negativo
                            │
                            ▼
               Comparación de etiquetas
```

**La optimización es computacional, no predictiva.** E9 evita materializar parte de la representación intermedia y reutiliza los parámetros del modelo original. Las pruebas verificaron coincidencia de etiquetas en las entradas evaluadas, sin constituir una demostración formal de equivalencia para cualquier texto posible.

La ruta de comparación no reentrena el modelo ni ejecuta el AG durante cada solicitud.

---

## 5. 📚 Dataset: Sentiment140 y relación con M1

**Sentiment140** contiene aproximadamente **1.600.000 tweets históricos en inglés**, con etiquetas originales **`0` (negativo)** y **`4` (positivo)**.

| Elemento | Cantidad | Función |
|---|---:|---|
| Corpus original Sentiment140 | **1.600.000** | Fuente académica |
| Muestra original estudiada en M1 | **200.000** | Construcción del clasificador |
| Registros preparados en M1 | **195.997** | Textos conservados tras limpieza |
| Prueba reservada de M1 | **29.400** | Medición formal de calidad predictiva |
| Cohortes de E13/E14 | **A y B** | Comparación de rendimiento con tweets reales y únicos |

M2 **no publica el CSV original** y **no necesita el dataset para clasificar**: carga el modelo persistido. Los datos empleados para las pruebas de rendimiento se conservaron fuera del repositorio público.

En términos de **Big Data**, el caso sirve para estudiar *volumen*, *variedad* y, principalmente en M2, *velocidad*. No implementa análisis distribuido ni streaming en tiempo real.

---

## 6. 🔤 Fundamentos: NLP, TF-IDF y regresión logística

El modelo conservado desde M1 utiliza una representación de texto mediante **TF-IDF** y un clasificador de **regresión logística**.

| Concepto | Papel en el sistema |
|---|---|
| **Normalización** | Prepara el texto antes de extraer características |
| **N-gramas** | Representan términos individuales y secuencias cortas |
| **TF-IDF** | Pondera términos según su relevancia en el corpus |
| **Regresión logística** | Calcula la clasificación binaria con los parámetros ya entrenados |
| **Fusión E9** | Evita trabajo intermedio innecesario durante la inferencia |
| **Latencia p95** | Resume el extremo alto de los tiempos observados de solicitudes |
| **Throughput** | Indica tweets procesados por segundo bajo una carga definida |

La clasificación sigue teniendo limitaciones ante **sarcasmo, ambigüedad, contexto y cambios de dominio**. Optimizar el cálculo no elimina estos problemas.

---

## 7. 🤖 Modelo original y control de integridad

M2 reutiliza el pipeline **TF-IDF + regresión logística** de M1 sin reentrenarlo.

| Indicador de referencia M1 | Valor histórico |
|---|---:|
| F1 macro en prueba reservada | **79,97 %** |
| Tweets de prueba | **29.400** |
| Python esperado | **3.11** |
| scikit-learn esperado | **1.9.1** |
| SHA-256 del modelo | `70d3ad62893c433120175f2b53f08236e3599e6737079d846284699c81329109` |

La comprobación:

```powershell
python -m nexus_optimize.cli audit
```

debe mostrar **`"status": "integrity_ok"`**. Esta verificación contrasta los hashes del modelo y del reporte con el manifiesto incluido; **no certifica que cualquier archivo `joblib` o `pickle` sea seguro**.

**F1 macro de M1 ≠ rendimiento HTTP de M2.** No se calculó una F1 nueva por mejorar la ejecución.

---

## 8. 🧬 Técnicas de optimización investigadas

El proyecto distingue las mejoras de ingeniería de las propuestas exploradas mediante búsqueda.

| Enfoque | Alcance y conclusión |
|---|---|
| **Línea base M1** | Referencia de inferencia y HTTP sobre el modelo persistido |
| **Procesamiento por lotes** | Disponible en la base de comparación; no se atribuye como descubrimiento de M2 |
| **Deduplicación y caché** | Útiles en cargas con textos repetidos, sin extrapolarlas a tweets nuevos |
| **Algoritmo Genético (AG)** | Búsqueda y evaluación de configuraciones de procesamiento |
| **E9: cálculo fusionado** | Ruta optimizada que obtuvo mejoras en el protocolo de tweets únicos |
| **E13/E14: evaluación HTTP** | Contraste de M1/E9 y conciliación de las métricas reportadas |

El AG se implementó y evaluó mediante selección, cruce, mutación y medición de candidatos. **No produjo una mejora adicional reproducible frente a la solución de ingeniería E9** en los escenarios de tweets únicos estudiados. Por ello, no se le atribuyen los resultados finales ni interviene en cada clasificación pública.

---

## 9. 🧪 Diseño de la evaluación HTTP

Los resultados principales corresponden a un **servicio HTTP local**, con condiciones controladas, no a un despliegue de Render.

| Elemento del protocolo | Descripción |
|---|---|
| Diseño de datos | **Dos cohortes independientes**: A y B |
| Entradas principales | **Tweets reales y únicos**, sin caché de duplicados |
| Repeticiones | **9 por escenario evaluado** |
| Volúmenes analizados | Incluyen escenarios de **1.000 y 10.000 tweets** |
| Concurrencia | **1 y 4** para 1.000; **1, 4 y 8** para 10.000, según escenarios válidos |
| Comparación | Misma carga sobre ruta original M1 y ruta optimizada E9 |
| Métricas | Latencia HTTP p95, tweets/s, coincidencia de etiquetas y condiciones de validez |
| Resultado de decisión | **10/10 escenarios principales válidos mejoraron** |

Los escenarios no válidos o no aplicables **no se contabilizan como mejoras**. La metodología distingue el coste de procesamiento de inferencia pura del coste observado en solicitudes HTTP.

**Método oficial de ganancia de rendimiento E14:**

\[
\text{Mejora de throughput (\%)} =
\left(
\frac{\operatorname{mediana}(\text{tweets/s E9})}
     {\operatorname{mediana}(\text{tweets/s M1})} - 1
\right)\times 100
\]

La mediana de mejoras **pareadas** del experimento E13 se conserva como estadística diagnóstica distinta; no debe confundirse con el cociente entre medianas del cierre E14.

---

## 10. 📈 Resultados experimentales

**Ejemplo representativo:** cohorte **A**, **10.000 tweets reales y únicos**, concurrencia **4**, en el servicio **HTTP local**.

| Indicador | M1 original | M2 / E9 | Variación |
|---|---:|---:|---:|
| **Latencia HTTP p95 por solicitud** | 47,343 ms | 40,596 ms | **−14,25 %** |
| **Capacidad de procesamiento** | 26.256 tweets/s | 32.756 tweets/s | **+24,75 %** |
| **Modelo entrenado** | Original M1 | Mismo modelo M1 | Sin reentrenamiento |
| **Etiquetas de clasificación** | Referencia | Coincidentes en las pruebas | Sin cambio observado |

**Balance del protocolo:** los **10 escenarios principales válidos** registraron mejora; no se observaron regresiones principales en ese conjunto. Esta conclusión se limita al diseño, equipo y configuración evaluados.

> **Las comparaciones HTTP se realizaron dentro de un servicio local; sus resultados no garantizan la misma velocidad en otros equipos o despliegues.**

Las cifras anteriores son evidencia **experimental local**; **no representan mediciones de latencia ni capacidad en Render**. La interfaz muestra datos del reporte E14 previamente verificado y no promete una velocidad fija para cada consulta interactiva.

---

## 11. 🏗️ Arquitectura y estructura del repositorio público

El repositorio M2 se ha preparado con una **selección de 34 archivos esenciales**, sin datasets, reportes originales de experimentación ni pruebas de desarrollo.

```text
nexus-mentis-optimize-m2/
│
├── README.md
├── Dockerfile
├── render.yaml
├── requirements.deploy.txt
├── .gitignore
├── .dockerignore
│
├── reference/
│   ├── model.joblib
│   ├── manifest.json
│   └── report_m1.json
│
├── public_reports/
│   └── e14_metrics_closure_public.json
│
└── nexus_optimize/
    ├── __init__.py
    ├── app.py
    ├── serve.py
    ├── cli.py
    ├── integrity.py
    ├── nlp.py
    ├── fast_nlp.py
    ├── fused_tfidf.py
    ├── optimization.py
    ├── benchmark.py
    ├── compare.py
    ├── data.py
    ├── global_http.py
    ├── metrics_closure.py
    ├── publish_reports.py
    ├── real_http.py
    ├── real_process.py
    ├── selective.py
    ├── telemetry.py
    ├── public_guard.py
    └── web/
        ├── index.html
        ├── styles.css
        ├── app.js
        └── nexus-logo.webp
```

### Responsabilidades principales

| Componente | Responsabilidad |
|---|---|
| `nexus_optimize/app.py` | API FastAPI, validaciones y entrega del panel web |
| `nexus_optimize/serve.py` | Inicio de Uvicorn en un proceso |
| `nexus_optimize/fused_tfidf.py` | Inferencia optimizada E9 |
| `nexus_optimize/integrity.py` | Auditoría del artefacto original |
| `nexus_optimize/metrics_closure.py` | Validación y presentación del reporte público E14 |
| `nexus_optimize/web/` | Interfaz didáctica responsiva |
| `reference/` | Artefactos del modelo original M1 |
| `public_reports/` | Solo métricas agregadas autorizadas |
| `Dockerfile` y `render.yaml` | Contenedor y configuración del despliegue |

El código de búsqueda genética, los experimentos completos, las pruebas y los scripts auxiliares **permanecen en el entorno local de investigación**. El repositorio mínimo permite ejecutar la demostración, pero **no contiene todo lo necesario para reproducir desde cero el estudio experimental**.

---

## 12. 🐳 Ejecución con Docker

Desde la raíz del repositorio, con Docker instalado:

```bash
docker build -t nexus-optimize-m2 .
docker run --rm -p 10000:10000 \
  -e PORT=10000 \
  -e NEXUS_TRUST_LOCAL_MODEL=1 \
  -e NEXUS_PUBLIC_DEMO=1 \
  -e NEXUS_PUBLIC_ENABLE_COMPARE=1 \
  -e NEXUS_REQUIRE_PUBLIC_REPORTS=1 \
  -e NEXUS_PUBLIC_MAX_BATCH=8 \
  -e NEXUS_PUBLIC_REQUESTS_PER_MINUTE=30 \
  -e NEXUS_INFERENCE_STRATEGY=reference \
  -e NEXUS_CACHE_CAPACITY=0 \
  -e NEXUS_EXPOSE_METRICS=0 \
  nexus-optimize-m2
```

**Interfaz:** http://localhost:10000/

El `Dockerfile` utiliza **Python 3.11**, instala las dependencias de despliegue, copia el modelo y el reporte público E14 y **comprueba su integridad durante la construcción**.

> **Estado de validación:** las pruebas del proyecto se aprobaron en Windows; la construcción y ejecución del contenedor en Render **requieren verificación independiente**. No se afirma aquí que el servicio M2 ya esté desplegado.

---

## 13. 🐍 Instalación local con Python

Se recomienda **Python 3.11** y mantener los artefactos del modelo original únicamente si provienen de una fuente conocida y autorizada.

**Terminal 1 — Servidor** (PowerShell, desde el repositorio):

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.deploy.txt
python -m nexus_optimize.cli audit

$env:NEXUS_TRUST_LOCAL_MODEL = "1"
$env:NEXUS_PUBLIC_DEMO = "1"
$env:NEXUS_PUBLIC_ENABLE_COMPARE = "1"
$env:NEXUS_REQUIRE_PUBLIC_REPORTS = "1"
$env:NEXUS_PUBLIC_MAX_BATCH = "8"
$env:NEXUS_PUBLIC_REQUESTS_PER_MINUTE = "30"
$env:NEXUS_INFERENCE_STRATEGY = "reference"
$env:NEXUS_CACHE_CAPACITY = "0"
$env:NEXUS_EXPOSE_METRICS = "0"
$env:NEXUS_HOST = "127.0.0.1"
$env:PORT = "8014"

python -m nexus_optimize.serve
```

Abre **http://127.0.0.1:8014/**.

**Terminal 2 — Comprobaciones** (otro PowerShell, desde la misma carpeta):

```powershell
.\.venv\Scripts\Activate.ps1

Invoke-RestMethod "http://127.0.0.1:8014/healthz"
Invoke-RestMethod "http://127.0.0.1:8014/api/status"
Invoke-RestMethod "http://127.0.0.1:8014/api/metrics-closure/results"
```

La versión mínima publicada **no contiene** la carpeta `tests/` ni `scripts/smoke_public.py`. Esas verificaciones automatizadas se ejecutaron y permanecen en el **proyecto local de desarrollo**, no en el repositorio público reducido.

---

## 14. 🔌 API y comprobación funcional

| Método | Ruta | Función en modo público |
|---|---|---|
| `GET` | `/` | Interfaz didáctica |
| `GET` | `/healthz` | Preparación del modelo; responde 200 si está listo |
| `GET` | `/api/status` | Estado mínimo del demostrador |
| `GET` | `/api/history` | Referencia histórica M1 con información acotada |
| `GET` | `/api/metrics-closure/results` | Métricas verificadas E14 |
| `POST` | `/api/pipeline/batch` | Comparación de clasificación de M1 y E9 |
| `POST` | `/api/predict` | Predicción individual |
| `POST` | `/api/predict/batch` | Predicción por lotes |
| `GET` | `/docs` | **404 intencional** en modo público |
| `GET` | `/openapi.json` | **404 intencional** en modo público |

La demostración pública limita las solicitudes de inferencia, entre otras restricciones, a **8 textos por lote**, **500 caracteres por texto** y **30 solicitudes de inferencia por minuto por proceso**, según `render.yaml`.

**Ejemplo PowerShell** (el mismo texto enviado por ambas rutas):

```powershell
$base = "http://127.0.0.1:8014"
$textos = @("I love this movie!", "I hate this delay.")

$m1 = @{
    texts = $textos
    language = "en"
    strategy = "m1"
} | ConvertTo-Json -Depth 3

$e9 = @{
    texts = $textos
    language = "en"
    strategy = "e9"
} | ConvertTo-Json -Depth 3

$original = Invoke-RestMethod -Uri "$base/api/pipeline/batch" `
    -Method Post -ContentType "application/json" -Body $m1

$optimizado = Invoke-RestMethod -Uri "$base/api/pipeline/batch" `
    -Method Post -ContentType "application/json" -Body $e9

$original.sentiments
$optimizado.sentiments
```

Esta comprobación contrasta etiquetas para **dos ejemplos**. No constituye una nueva evaluación estadística ni un benchmark de rendimiento.

---

## 15. ☁️ Publicación en GitHub y Render

**📁 Repositorio de M2:** [edtech-mx-ve/nexus-mentis-optimize-m2](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2)

**📁 Repositorio de M1:** [edtech-mx-ve/nexus-mentis-analyze-m1](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1)

| Configuración | Valor previsto |
|---|---|
| Proveedor | **Render** |
| Creación del servicio | **New → Blueprint** |
| Repositorio | `edtech-mx-ve/nexus-mentis-optimize-m2` |
| Rama | `main` |
| Runtime | **Docker** |
| Archivo de configuración | `render.yaml` |
| Health check | `/healthz` |
| Modelo de inferencia | Artefacto original M1, auditado |
| Evidencia pública | Reporte E14 sanitizado |
| Base de datos | No requerida |

`render.yaml` define el modo público y los límites de la demostración. Tras crear y arrancar el servicio, se debe comprobar la **URL real** asignada por Render. **No se incluye una URL M2 no verificada.**

### 🔎 Comprobaciones tras el despliegue

Sustituir la dirección solo cuando Render haya proporcionado la URL:

```powershell
$base = "https://URL-REAL-DE-M2.onrender.com"

(Invoke-WebRequest "$base/" -UseBasicParsing).StatusCode
Invoke-RestMethod "$base/healthz"
Invoke-RestMethod "$base/api/status"
Invoke-RestMethod "$base/api/metrics-closure/results"
```

El éxito de la página principal **no demuestra por sí solo** que el modelo esté cargado o que la comparación funcione. Deben probarse también las predicciones y los enlaces institucionales.

---

## 16. ✅ Verificaciones, trazabilidad y reproducibilidad

El desarrollo local registró:

- **221 pruebas automatizadas aprobadas** en la revisión previa a la publicación.
- **`integrity_ok`** para el modelo y reporte M1.
- Exportación verificada de `e14_metrics_closure_public.json` a partir de la evidencia E13/E14 original.
- Prueba de humo pública local con rutas internas ocultas y coincidencia de etiquetas M1/E9.

**Comprobaciones disponibles en el repositorio público mínimo:**

```powershell
python -m nexus_optimize.cli audit
python -m py_compile `
  .\nexus_optimize\app.py `
  .\nexus_optimize\serve.py `
  .\nexus_optimize\fused_tfidf.py `
  .\nexus_optimize\metrics_closure.py

git diff --check
git status --short
```

La compilación verifica sintaxis y la auditoría contrasta integridad. **No sustituyen la prueba de arranque ni la validación HTTP en Render.**

La reproducción completa del estudio exige las cohortes, reportes originales y utilidades de experimentación, **conservados fuera de este repositorio público**.

---

## 17. 🔐 Robustez y seguridad

- Validación de texto, idioma y cantidad de elementos enviados.
- Restricción básica de frecuencia por proceso para la demostración pública.
- Ocultamiento de documentación Swagger y rutas experimentales internas en modo público.
- Reporte E14 público limitado a resultados agregados y verificados, sin publicar tweets de investigación.
- Separación entre datos experimentales internos y archivos de despliegue.
- Comprobación SHA-256 del artefacto M1 al auditar.
- Un único proceso Uvicorn para limitar duplicación de memoria.

**Precauciones:**

- No enviar información personal o sensible al servicio de demostración.
- No confiar en modelos `joblib`/`pickle` de origen desconocido: **la coincidencia SHA-256 no garantiza seguridad**.
- La limitación por proceso no es autenticación ni protección completa frente a abusos.
- El repositorio es público: `reference/model.joblib` puede descargarse; su redistribución debe estar autorizada.
- El README no concede licencias adicionales sobre Sentiment140 ni contenidos de terceros.

---

## 18. ⚠️ Limitaciones e interpretación de resultados

- **Entorno:** los benchmarks HTTP se realizaron localmente; otros equipos y despliegues pueden rendir de forma distinta.
- **Alcance estadístico:** 10/10 mejoras se refiere a **escenarios principales válidos**, no a todo tráfico imaginable.
- **Modelo:** clasificación binaria heredada de M1; no se ha reentrenado ni vuelto a evaluar F1.
- **Lenguaje:** clasificación de referencia en inglés; M2 no incorpora traducción ES→EN.
- **Equivalencia:** se verificó coincidencia de etiquetas en las pruebas realizadas, no identidad universal de todas las salidas numéricas.
- **AG:** se evaluó, pero no aportó una ganancia adicional reproducible al diseño finalmente elegido.
- **Datos:** tweets históricos y muestras seleccionadas; no representa un monitoreo de X en tiempo real.
- **Infraestructura:** el plan gratuito de Render puede imponer pausas, reinicios y restricciones de memoria o CPU.
- **Carácter académico:** no es un servicio de decisiones de alto impacto ni un sistema comercial con garantías de disponibilidad.

> **Las comparaciones HTTP se realizaron dentro de un servicio local; sus resultados no garantizan la misma velocidad en otros equipos o despliegues.**

---

## 19. 🎓 Contexto académico y referencias

| Dato | Información |
|---|---|
| 🏛️ Institución | **Instituto Internacional de Aguascalientes** |
| 🎓 Programa | **Maestría en Inteligencia Artificial para la Transformación Digital** |
| 📘 Asignatura | **Tópicos de IA para la Transformación Digital** |
| 🧑‍🎓 Estudiante | **Antonio Nicolás Toro González** |
| 👨‍🏫 Tutor | **Dr. Francisco Javier Luna Rosas** |
| 🧠 Proyecto | **NEXUS MENTIS — NEXUS OPTIMIZE (Módulo 2)** |

### Referencias técnicas y académicas

- [Stanford — Sentiment140: archivo histórico](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- [Go, Bhayani y Huang (2009) — *Twitter Sentiment Classification using Distant Supervision*](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
- [scikit-learn — Persistencia de modelos](https://scikit-learn.org/stable/model_persistence.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Render — Web Services](https://render.com/docs/web-services)
- [Render — Blueprint YAML](https://render.com/docs/blueprint-spec)
- [Instituto Internacional de Aguascalientes](https://www.iinternacional.edu.mx/)

### Enlaces del proyecto

- **📁 Repositorio del Módulo 1:** [NEXUS ANALYZE — Código fuente y documentación en GitHub](https://github.com/edtech-mx-ve/nexus-mentis-analyze-m1)
- **📁 Repositorio del Módulo 2:** [NEXUS OPTIMIZE — Código fuente y documentación en GitHub](https://github.com/edtech-mx-ve/nexus-mentis-optimize-m2)
- **🌐 Aplicación M1 en Render:** https://nexus-mentis-analyze-m1.onrender.com/
- **🗃️ Dataset:** https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

---

<div align="center">

### 🧠 NEXUS MENTIS — NEXUS OPTIMIZE

**📥 Tweet → 🔤 NLP + TF-IDF → ⚙️ E9 (cálculo fusionado) → 😊/😞 Sentimiento**

**📊 Ejemplo HTTP local: −14,25 % latencia p95 · +24,75 % tweets/s**

*Práctica 2 · Módulo 2 · Instituto Internacional de Aguascalientes*

</div>
