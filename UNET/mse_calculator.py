import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
# Definisci la rete neurale per il calcolo del MSE
class MSEModel(nn.Module):
    def __init__(self):
        super(MSEModel, self).__init__()

    def forward(self, x1, x2):
        return torch.mean((x1 - x2) ** 2)

def load_images_from_folder(folder):
    """Carica tutte le immagini da una cartella in un dizionario con il nome dell'immagine come chiave."""
    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            #gray scale
            image = image.convert('L')
            if image is not None:
                images[filename] = image
    return images

def load_images_from_folderCV(folder):
    """Carica tutte le immagini da una cartella in un dizionario con il nome dell'immagine come chiave."""
    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            #gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image =cv2.resize(image, (1024, 1024))
            if image is not None:
                images[filename] = image
    return images


def main():
    folder1 = 'MSE_images/1'
    folder2 = 'MSE_masks'

    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
    imagesA = load_images_from_folderCV(folder1)
    imagesB = load_images_from_folderCV(folder2)


    # Inizializza il modello per il calcolo del MSE
    mse_model = MSEModel()

    # Trasformazioni per le immagini (ridimensiona e normalizza)
    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor()])

    mse_loss = nn.MSELoss()

    for filename in images1:
        if filename in images2:
            image1 = transform(images1[filename])
            image2 = transform(images2[filename])
            # Aggiungi una dimensione batch (batch_size=1) per effettuare il calcolo
            image1 = image1.unsqueeze(0)
            image2 = image2.unsqueeze(0)
            
            # Calcola il MSE tra le due immagini
            mse_value = mse_loss(image1, image2)
            ssim_score = ssim(imagesA[filename], imagesB[filename])
            print(f"MSE tra {filename}: {mse_value.item()}")
            print(f"SSIM tra {filename}: {ssim_score.item()}")

if __name__ == "__main__":
    main()
