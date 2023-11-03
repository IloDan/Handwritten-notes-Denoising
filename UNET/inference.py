import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import UNet
from src.config import IN_CHANNELS, OUT_CHANNELS, DEVICE, MEAN, STD
import matplotlib.pyplot as plt

# Carica il modello dai pesi salvati
model_name = 'Unet_1024_2epochs_check_1695964609'  # Sostituisci con il nome corretto del modello
model = UNet(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(f'/work/cvcs_2023_group11/checkpoints/{model_name}.pth'))
model.eval()

# Definisci le trasformazioni per preelaborare i dati di input, incluso il resize
resize_size = (1024, 1024)
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Cartella contenente le immagini di input
input_folder = 'test_image/'
# Cartella in cui salvare i risultati
output_folder = 'test_image'

# Aggiungi il nome del modello alla sottocartella "inference"
inference_folder = os.path.join(output_folder, f'{model_name}')
os.makedirs(inference_folder, exist_ok=True)

# Elabora tutte le immagini nella cartella di input
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        try:
            # Carica l'immagine di input
            input_image = Image.open(os.path.join(input_folder, filename))
            input_image = input_image.convert('L')
            
            # Esegui la trasformazione con il resize
            input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)

            # Esegui l'inferenza
            with torch.no_grad():
                output = model(input_tensor)

            # Rimuovi le iterazioni inutili
            output = output.cpu().squeeze(0).squeeze(0)
            output_image = (output * STD) + MEAN

            # Salva l'immagine di output nella cartella "inference_nome modello"
            output_path = os.path.join(inference_folder, filename)
                        # Visualizza l'immagine di output
            plt.figure(figsize=(10,10))
            plt.imshow(output_image , cmap='gray')
            plt.title('Output')
            plt.savefig(output_path)
            plt.show()

            print(f"Elaborazione di {filename} - Completata. Immagine di output salvata in {output_path}")

        except Exception as e:
            print(f"Errore durante l'inferenza o il salvataggio dell'immagine di output {filename}: {str(e)}")
