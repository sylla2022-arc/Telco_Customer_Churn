import logging
import os

current_path =os.getcwd()
log_dir = os.path.join(current_path, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    filename=log_path,
    format="%(asctime)s- %(name)s- %(levelname)s- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    filemode='w'
)

# Crée une instance de logger nommée
logging = logging.getLogger(__name__)
