FROM python:3.10.13-slim-bullseye as base

WORKDIR /app



RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the source code

VOLUME [ "/app" ]

CMD  ["uvicorn", "api:app", "--reload" ,"--host", "0.0.0.0",  "--port", "8000"]


