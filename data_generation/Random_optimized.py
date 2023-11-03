import cv2
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

# Definisci il percorso della cartella di input e della cartella di output
input_path = '/work/cvcs_2023_group11/input_folder'
grid_path = '/work/cvcs_2023_group11/grids/grid_folder'
output_path = '/work/cvcs_2023_group11/dataset/train_masks/'
text_grid_path = '/work/cvcs_2023_group11/dataset/train_images/'

# Dimensioni desiderate delle immagini
desired_width, desired_height = 1024, 1024

# Numero di griglie da assegnare ad ogni immagine
num_grids_per_image = 30

# Crea la cartella di output se non esiste già
os.makedirs(output_path, exist_ok=True)
os.makedirs(text_grid_path, exist_ok=True)

# Funzione per ridimensionare un'immagine alla dimensione desiderata
def resize_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        resized_img = cv2.resize(img, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
        return resized_img
    return None

def gridDistorsion(image):

    # Define the number of rows and columns in the grid
    rows, cols = image.shape[0], image.shape[1]

    # Create a meshgrid of points
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    ampiezza = random.randint(1, 3)
    freq = random.randint(90, 150)
    
    # print('ampiezza: ', ampiezza)
    # print('freq: ', freq)

    # Add distortion to the grid (example: sine wave distortion)
    x_distorted = x + ampiezza * np.sin(y / freq)
    y_distorted = y + ampiezza * np.sin(x / freq)

    # Convert the distorted grid to float32
    x_distorted = x_distorted.astype(np.float32)
    y_distorted = y_distorted.astype(np.float32)

    # Interpolate the distorted grid to get new coordinates
    distorted_points = np.stack((x_distorted, y_distorted), axis=-1)

    # Apply the distortion to the image
    distorted_image = cv2.remap(image, distorted_points, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return distorted_image

def changeBrightness(image):    

    # Generate a random brightness factor for the entire image
    global_brightness_factor = random.uniform(0.85, 1)  # Adjust the range as needed

    # Apply brightness adjustment to the entire image
    toned_image = cv2.convertScaleAbs(image, alpha=global_brightness_factor, beta=0)

    # Define the number of regions with varying brightness
    num_regions = random.randint(1, 3)  # You can adjust the number of regions as needed

    # Generate random regions with varying brightness
    for _ in range(num_regions):

        
        # Randomly select a region by defining its top-left and bottom-right corners
        top_left = (random.randint(0, image.shape[1]), random.randint(0, image.shape[0]))
        bottom_right = (random.randint(top_left[0], image.shape[1]), random.randint(top_left[1], image.shape[0]))

        region_brightness_factor = random.uniform(0.5, 5)  # Adjust the range as needed
        # Creare una maschera ellittica sfumata
        mask = np.zeros_like(image)
        center = (random.randint(top_left[0], bottom_right[0]), random.randint(top_left[1], bottom_right[1]))
        axes_length = (random.randint(200, 800), random.randint(500, 1000))
        angle = random.randint(0, 360)  # Angolo di rotazione
        cv2.ellipse(mask, center, axes_length, angle, 0, 360, (1, 1, 1), -1)

        # Applicare la maschera sfumata per la transizione
        transition_mask = cv2.GaussianBlur(mask, (75, 75), sigmaX=100)  # Aumenta il valore di sigmaX per una sfumatura più morbida

        # brightness adjustment alla regione selezionata
        toned_image = cv2.addWeighted(toned_image, 1, mask, region_brightness_factor, 0)
        #sfocatura gaussiana per regolare la transizione fra le regioni leggera
        toned_image = cv2.GaussianBlur(toned_image, (1, 1), 10000)
        

    return toned_image


# Funzione per elaborare una singola immagine e assegnare griglie casuali
def process_image(file_name):
    # Verifica se il file è un'immagine
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        input_img_resized = resize_image(os.path.join(input_path, file_name))
        if input_img_resized is not None:
            img_blur = cv2.medianBlur(input_img_resized, 3)
            img_thresh = cv2.bitwise_not(img_blur)
            cv2.imwrite(os.path.join(output_path, file_name), cv2.bitwise_not(img_thresh))

            # Ottieni una lista di tutte le griglie nella cartella "grid_folder"
            grid_files = [f for f in os.listdir(grid_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Se ci sono meno di num_grids_per_image griglie, prendi tutte le griglie disponibili
            if len(grid_files) <= num_grids_per_image:
                selected_grids = grid_files
            else:
                # Altrimenti, seleziona casualmente num_grids_per_image griglie
                selected_grids = random.sample(grid_files, num_grids_per_image)
            
            # Ridimensiona e combina le immagini di griglia selezionate
            for grid_name in selected_grids:
                grid_img_resized = resize_image(os.path.join(grid_path, grid_name))
                if grid_img_resized is not None:
                    distorted_grid = gridDistorsion(grid_img_resized)
                    grid_thresh = cv2.bitwise_not(distorted_grid)
                    # final = cv2.bitwise_not(grid_thresh + img_thresh)
                    final = cv2.max(grid_thresh, img_thresh)
                    # riporto l'immagine da negativa a positiva
                    final = cv2.bitwise_not(final)
                    # introduco variazioni di luminosità
                    final = changeBrightness(final)


                    # Salva l'immagine finale nella cartella di output con lo stesso nome del file di input
                    output_name = f"{file_name[:-4]}_{grid_name}"
                    new_path = os.path.join(text_grid_path, file_name[:-4])
                    os.makedirs(new_path, exist_ok=True)
                    cv2.imwrite(os.path.join(new_path, output_name), final)

if __name__ == '__main__':
    # Ottieni la lista di tutti i file nella cartella di input
    all_files = os.listdir(input_path)

    # Utilizza il ThreadPoolExecutor per ridimensionare le immagini in parallelo
    with ThreadPoolExecutor() as executor, tqdm(total=len(all_files)) as pbar:
        for _ in executor.map(process_image, all_files):
            pbar.update(1)

    pbar.close()
