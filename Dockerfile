FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         ca-certificates \
        curl \
        gnupg \
        awscli && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY artifact/ artifact/
COPY data_churn/ data_churn/
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]