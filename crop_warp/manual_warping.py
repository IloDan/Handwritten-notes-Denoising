import cv2
import numpy as np
import os

def specify_corners(image_path):
    # Carica l'immagine
    img = cv2.imread(image_path)

    # Crea una finestra per l'immagine originale
    cv2.namedWindow("Original Image")
    cv2.imshow("Original Image", img)

    # Specifica la dimensione massima per l'immagine visualizzata
    max_display_width = 700
    max_display_height = 700

    # Ridimensiona l'immagine se supera le dimensioni massime
    if img.shape[1] > max_display_width or img.shape[0] > max_display_height:
        scale_factor = min(max_display_width / img.shape[1], max_display_height / img.shape[0])
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Inizializza una lista vuota per i punti di destinazione
    dst_pts = []

    def mouse_callback(event, x, y, flags, param):
        # Callback per gestire i clic del mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            # Aggiungi il punto di destinazione alla lista
            dst_pts.append((x, y))
            # Disegna un cerchio sul punto di destinazione
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Original Image", img)

    # Collega la funzione di callback al clic del mouse
    cv2.setMouseCallback("Original Image", mouse_callback)

    print("Clicca su quattro punti nell'ordine: in alto a sinistra, in alto a destra, in basso a destra, in basso a sinistra.")
    
    while True:
        cv2.imshow("Original Image", img)
        key = cv2.waitKey(1) & 0xFF

        if len(dst_pts) == 4:
            break

    cv2.destroyAllWindows()

    # Punti di origine (angoli dell'immagine originale)
    src_pts = np.array([(0, 0), (img.shape[1]-1, 0), (img.shape[1]-1, img.shape[0]-1), (0, img.shape[0]-1)], dtype="float32")

    # Calcola la matrice di trasformazione
    M = cv2.getPerspectiveTransform(np.array(dst_pts, dtype="float32"), src_pts)

    # Crea un'immagine di output vuota con le dimensioni dell'area specificata
    max_x, max_y = np.max(dst_pts, axis=0)
    min_x, min_y = np.min(dst_pts, axis=0)
    output_width = int(max_x - min_x)
    output_height = int(max_y - min_y)
    output_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Calcola la correzione prospettica solo per l'area specificata e copiala nell'immagine di output
    corrected_region = cv2.warpPerspective(img, M, (output_width, output_height))
    cv2.imshow("Corrected Region", corrected_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output_img = corrected_region

    # Visualizza l'immagine corretta
    cv2.imshow("Corrected Image", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Esempio di utilizzo
# image_path = os.path.abspath('./input/ANALISI 1_page2.jpg')
# image_path = os.path.abspath('./input/test_1.jpg')
image_path = os.path.abspath('./input/Storta1.jpg')


specify_corners(image_path)
