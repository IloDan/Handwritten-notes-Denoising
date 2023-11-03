import os

def change_extension(input_folder, new_extension):
    # Controlla se la cartella di input esiste
    if not os.path.exists(input_folder):
        print("La cartella di input non esiste.")
        return

    # Esplora ricorsivamente la cartella di input
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Percorso completo del file di input
            input_path = os.path.join(root, file)

            # Estensione del file
            _, extension = os.path.splitext(file)

            if extension.lower() == '.jpg' or extension.lower() == '.jpeg':
                try:
                    # Percorso completo del file con la nuova estensione
                    new_path = os.path.join(root, f"{os.path.splitext(file)[0]}.{new_extension}")
                    # Rinomina il file
                    os.rename(input_path, new_path)
                    print(f"{input_path} rinominato in {new_path}")
                except Exception as e:
                    print(f"Si è verificato un errore durante la rinomina di {input_path}: {str(e)}")
            else:
                print(f"Il file {input_path} non è un'immagine .jpg o .jpeg.")

# Esempio di utilizzo
input_folder = '/work/cvcs_2023_group11/dataset/train_masks'
new_extension = 'png'
change_extension(input_folder, new_extension)
