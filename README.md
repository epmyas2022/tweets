# API With Model Emotion Detection

Este proyecto es la API de un modelo de detección de emociones en imágenes. El modelo fue entrenado con 25000 textos
clasisifocados como positivos, negativos o netutrales. El modelo fue entrenado con la librería de machine learning de python tensorflow.

> [!NOTE]
> Este proyecto es una practica de la creación de una API con un modelo de machine learning. El modelo fue entrenado con 25000 textos

## Requisitos

- Python 3.10
- Pip
- Docker (Opcional)
- Docker Compose (Opcional)

## Instalación de dependencias

Para instalar las dependencias del proyecto, se debe ejecutar el siguiente comando:

```bash
pip install -r requirements.txt
```

## Descargar conjunto de datos

Para descargar el conjunto de datos, se debe ejecutar el siguiente comando:

```bash
cd data
```

```bash
gdown --id 1TPi3PMkvDjgKzjo1f1gQtwXd_VA94P0j 
```

## Entrenamiento del modelo

Para entrenar el modelo, se debe ejecutar el siguiente comando:

```bash
python3 model.py
```

Para ejecutar las pruebas del modelo, se debe ejecutar el siguiente comando:

```bash
python3 pedict.py
```

## Ejecución del proyecto

Para ejecutar el proyecto, se debe ejecutar el siguiente comando:

```bash
uvicorn main:app --reload
```

## Ejecución del proyecto con Docker

Para ejecutar el proyecto con Docker, se debe ejecutar el siguiente comando:

```bash
docker-compose up -d
```

## Documentación de la API

La documentación de la API se encuentra en la siguiente URL:

`localhost:8000/docs`

## ENDPOINT

### GET HTTP://BASE_URL/{predict}

Este endpoint recibe una oracion y devuelve:

- `Predict` (**String**): La predicción de la emoción de la oración.
- `Probabilities` (**Dict**): Las probabilidades de cada emoción.
- `Text` (**String**): La oración que se predijo.

## Ejemplo

### Request

```json
{
  "Text": "I am happy"
}
```

### Response

```json
{
  "Text": "I am happy",
  "Posibilities": {
    "negative": 8,
    "positive": 88,
    "neutral": 4
  },
  "Predict": "positive"
}
```
