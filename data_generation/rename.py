import os

# Specifica il percorso della cartella dei file da rinominare
cartella = "/homes/dcaputo/handwritten-math-recognition/data_generation/input_folder"

# Lista tutti i file nella cartella
files = os.listdir(cartella)

# Ordina i file in ordine alfabetico
files.sort()

# Inizializza una variabile per tenere traccia del numero crescente
numero_crescente = 1

# Itera sui file e rinominali
for file in files:
    # Crea il nuovo nome per il file
    nuovo_nome = str(numero_crescente)

    # Ottieni l'estensione del file
    estensione = os.path.splitext(file)[1]

    # Rinomina il file
    os.rename(os.path.join(cartella, file), os.path.join(cartella, nuovo_nome + estensione))

    # Incrementa il numero crescente per il prossimo file
    numero_crescente += 1


