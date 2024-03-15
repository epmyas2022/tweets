from fastapi import FastAPI

import classifier as cl
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
classifier = cl.Classifier("./model/mymodel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def round_number(number):
    return round(float(number), 2)


@app.get("/{text}")
def read_root(text: str):
    # obtener el texto de la url fast api

    predict = classifier.classify([text])

    return {
        "Text": text,
        "Posibilities": {
            "negative": round_number(round_number(predict[0][0]) * 100),
            "positive": round_number(round_number(predict[0][1]) * 100),
            "neutral": round_number(round_number(predict[0][2]) * 100),
        },
        "Predict": list(classifier.labels.keys())[predict[0].argmax()],
    }
