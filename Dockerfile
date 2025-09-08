FROM python:3.10-bullseye

WORKDIR /app
# tk-dev et python3-tk pour evidently AI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        awscli \
        tk-dev \  
        python3-tk \
        && rm -rf /var/lib/apt/lists/*

COPY artifact/ artifact/
COPY data_churn/ data_churn/
COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Streamlit
EXPOSE 8501

# Lancer uniquement Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
