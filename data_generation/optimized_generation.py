import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# Definisci il percorso della cartella di input e della cartella di output
input_path = '/work/cvcs_2023_group11/input20'
grid_path = '/work/cvcs_2023_group11/grid10'
output_path = '/work/cvcs_2023_group11/data_generation/train_masks/'
text_grid_path = '/work/cvcs_2023_group11/data_generation/dataset/train_images/'

# Dimensioni desiderate delle immagini
desired_width, desired_height = 1024, 1024

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

# Funzione per elaborare una singola immagine e la relativa griglia
def process_image(file_name):
    # Verifica se il file è un'immagine
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        input_img_resized = resize_image(os.path.join(input_path, file_name))
        if input_img_resized is not None:
            img_blur = cv2.medianBlur(input_img_resized, 3)
            img_thresh = cv2.bitwise_not(img_blur)
            cv2.imwrite(os.path.join(output_path, file_name), cv2.bitwise_not(img_thresh))

            # Ridimensiona tutte le immagini di griglia nella cartella "grid_folder"
            for grid_name in os.listdir(grid_path):
                # Verifica se il file è un'immagine
                if grid_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    grid_img_resized = resize_image(os.path.join(grid_path, grid_name))
                    if grid_img_resized is not None:
                        grid_thresh = cv2.bitwise_not(grid_img_resized)
                        final = cv2.max(grid_thresh, img_thresh)
                        final = cv2.bitwise_not(final)

                        # final = cv2.bitwise_not(grid_thresh + img_thresh)

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
