# Predicción del nivel de riesgo cardíaco (Bajo / Medio / Alto) – Prototipo con Streamlit

**Curso:** Minería de Datos – Universidad de la Costa · Tercer corte
**Integrantes:** Sebastián Andrés Muñoz Lemus  <!-- agrega a todos -->
**App pública:** <URL de Streamlit Community Cloud>

> Prototipo académico. No es un dispositivo médico ni sustituye la decisión de un profesional de la salud.

## Descripción
Clasificación multiclase del nivel de riesgo cardíaco con ocho variables clínicas del
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano, 2021).
`nivel_riesgo` es una etiqueta construida por el grupo (diagnóstico + tres alertas de la prueba de esfuerzo); esas variables
no se usan como predictoras.

## Contenido del repositorio
- `app.py` – aplicación Streamlit
- `artefactos/modelo_riesgo.joblib` – pipeline completo (imputación, escalado, codificación y modelo)
- `artefactos/metricas.json`, `datos_eda.csv`, `muestra_prueba.csv` – métricas, datos para gráficos y registros de verificación
- `Avance2.ipynb` – notebook actualizado
- `requirements.txt`

## Ejecutar en local
```
pip install -r requirements.txt
streamlit run app.py
```

## Cambios frente al segundo corte
El accuracy del corte 2 (0.674) estaba por debajo de 0.70 y la clase Medio casi no se detectaba (recall 0.135), por lo que se probó una regresión logística con `class_weight="balanced"` (mismo pipeline, mismo split y misma semilla).

| Modelo | CV F1 macro | Test accuracy | Test F1 macro | Recall Alto | Recall Medio | Alto→Bajo |
|---|---|---|---|---|---|---|
| Regresión logística (corte 2) | 0.544 ± 0.008 | 0.674 | 0.572 | 0.849 | 0.135 | 9 |
| **Regresión logística balanceada (final)** | 0.578 ± 0.034 | 0.652 | 0.609 | 0.731 | 0.346 | 7 |
| Árbol de decisión (prof. 2) | 0.594 ± 0.034 | 0.591 | 0.550 | 0.742 | 0.308 | 10 |

<!-- Justifica aquí la elección con tus palabras: el costo es menor recall de Alto y menor accuracy. -->

## Pruebas realizadas
| Prueba | Resultado |
|---|---|
| Caso normal | <completar> |
| Entrada inválida | <completar> |
| Registro del conjunto de prueba (app = notebook) | <completar: id y clase> |

## Uso de inteligencia artificial
- **Herramienta:** <completar>
- **Propósito:** <completar>
- **Prompt principal:** <completar>
- **Cómo verifiqué la respuesta:** <completar: ejecución, comparación app vs. notebook, revisión del código>
- **Problemas encontrados:** <completar>
- **Ajustes realizados:** <completar>
