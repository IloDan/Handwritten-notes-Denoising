import cv2
import numpy as np
import os

# Definisci il percorso della cartella di input e della cartella di output
input_path = './input_folder/'
grid_path = './grid_folder/'
output_path = './train_mask/'
text_grid_path = './train_image/'

# Crea la cartella di output se non esiste già
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(text_grid_path):
    os.makedirs(text_grid_path)

# Loop attraverso tutte le immagini nella cartella di input
for file_name in os.listdir(input_path):
    # Verifica se il file è un'immagine
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        # Leggi l'immagine in scala di grigi
        img = cv2.imread(os.path.join(input_path, file_name), cv2.IMREAD_GRAYSCALE)

        # Applica il filtro di sfocatura mediana con kernel 3x3
        img_blur = cv2.medianBlur(img, 3)

        img_thresh=cv2.bitwise_not(img_blur)
        cv2.imwrite(os.path.join(output_path, file_name), cv2.bitwise_not(img_thresh))

    for grid_name in os.listdir(grid_path):
            # Verifica se il file è un'immagine
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            # Carica l'immagine di griglia
            grid = cv2.imread(os.path.join(grid_path, grid_name), cv2.IMREAD_GRAYSCALE)

            resized = cv2.resize(grid, (img_thresh.shape[1], img_thresh.shape[0]), interpolation=cv2.INTER_AREA)
            grid_thresh=cv2.bitwise_not(resized)
            # Esegui il massimo tra le due immagini senza applicare ulteriori soglie
            # verifica su una imm e una griglia che effetivamente rpoduca stesso risultato
            final = cv2.max(cv2.bitwise_not(grid_thresh), cv2.bitwise_not(img_thresh))

            # final = cv2.bitwise_not(grid_thresh + img_thresh)

            # Salva l'immagine finale nella cartella di output con lo stesso nome del file di input
            new_path = os.path.join(text_grid_path, file_name[:-4])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            cv2.imwrite(os.path.join(new_path, file_name[:-4] + grid_name), final)