FROM python:3.10-slim-bullseye

WORKDIR /app

# INSTALLER LES OUTILS DE COMPILATION EN PREMIER
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        python3-dev \
        awscli \
        && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de l'application
COPY artifact/ artifact/
COPY data_churn/ data_churn/
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]