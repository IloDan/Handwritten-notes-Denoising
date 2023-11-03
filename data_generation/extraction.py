import os
import shutil

# Percorso della cartella principale contenente le sottocartelle di immagini
input_folder = '/percorso/cartella_principale'

# Percorso della cartella di destinazione in cui spostare le immagini
output_folder = '/percorso/cartella_destinazione'

# Crea la cartella di destinazione se non esiste già
os.makedirs(output_folder, exist_ok=True)

# Dizionario per tenere traccia dei nomi dei file duplicati
duplicate_names = {}

# Itera tutte le sottocartelle nella cartella principale
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    
    # Verifica se l'elemento nella cartella principale è una sottocartella
    if os.path.isdir(subfolder_path):
        # Itera tutti i file nella sottocartella
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            
            # Verifica se il file è un'immagine
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Verifica se il nome del file è già presente nel dizionario dei duplicati
                if file_name in duplicate_names:
                    # Genera un nuovo nome per l'immagine duplicata
                    duplicate_count = duplicate_names[file_name]
                    duplicate_count += 1
                    duplicate_names[file_name] = duplicate_count
                    new_file_name = f"{file_name[:-4]}_{duplicate_count}{file_name[-4:]}"
                else:
                    # Aggiungi il nome del file al dizionario dei duplicati con contatore 1
                    duplicate_names[file_name] = 1
                    new_file_name = file_name
                
                # Crea il percorso completo del nuovo file rinominato nella cartella di destinazione
                new_file_path = os.path.join(output_folder, new_file_name)
                
                # Sposta il file rinominato nella cartella di destinazione
                shutil.move(file_path, new_file_path)
