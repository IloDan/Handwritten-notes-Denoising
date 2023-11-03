#import os
import torch
#import datetime

# # Directories
# DATA_DIR = "data"
# MODEL_DIR = "models"
# LOG_DIR = "logs"


# Device
assert torch.cuda.is_available(), "Notebook non è configurato correttamente!"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# Dataset path
# ROOT = '/work/cvcs_2023_group11/dataset'
ROOT = '/work/cvcs_2023_group11/dataset_more_grids'
MEAN = 0.879
#0.858 FT
#0.8790 more grids
#0.8904 distorte e luminosità
STD =  0.1336
#0.1417 FT
#0.1336 more grids 
#0.1249 distorte e luminosità

# Hyperparameters
BATCH_SIZE = 2
NUM_WORKERS = 4

NUM_EPOCHS = 2
LEARNING_RATE = 0.001

# Image size
IMG_SIZE = 1024

# Model parameters
IN_CHANNELS = 1
OUT_CHANNELS = 1

# Print hyperparameters
print('Congrats, stai eseguendo questo codice su una GPU', torch.cuda.get_device_name())
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Image size: {IMG_SIZE}")
print(f"Root: {ROOT}")
print(f"Mean: {MEAN}")
print(f"Std: {STD}")
