L'algoritmo implementato è un motore di ricerca per il retrieval di immagini basato su contenuto (CBIR) che utilizza feature R-MAC (Regional Maximum Activation of Convolutions).

Il funzionamento in generale è il seguente:

1. Viene caricato un modello CNN pre-addestrato (VGG16 o VGG19) e rimosso lo strato di classificazione finale in modo da ottenere un estrattore di feature.

2. Per ogni immagine nel database viene calcolato il vettore di feature R-MAC ed salvato in una matrice db_feature_mat.

3. Per una immagine di query, si calcola il suo vettore R-MAC feature_vec. 

4. Si calcola la similarità tra feature_vec e ogni vettore in db_feature_mat usando il coseno della distanza.

5. Si ritornano le immagini del database con i punteggi di similarità più alti.


Entrando nel dettaglio della funzione rmac():

- Prende in input un tensore x di shape (N, C, W, H) corrispondente alla mappa di attivazione dell'ultimo strato convoluzionale.

- Viene calcolata la dimensione della finestra w come minimo tra larghezza W e altezza H.

- Per ogni livello di scala l (da min_scale_level a max_scale_level):

  - Si calcola la dimensione della finestra al livello l come w/(l+1)

  - Si calcola la stride della finestra per avere una sovrapposizione del 40% tra le finestre.

  - Si applica il pooling (max o avg) con kernel size e stride appena calcolati.

  - Dal risultato si ricavano le coordinate ijhw di ogni finestra a quel livello di scala.

  - Il risultato del pooling viene appiattito in un vettore ed inserito in una lista feature_vec.

- feature_vec contiene i vettori di tutti i livelli di scala concatenati. 

- Viene fatta la L2 normalization di feature_vec.

- Viene ritornato il vettore R-MAC feature_vec e le coordinate delle finestre regions_ijhw.

In questo modo l'operazione di R-MAC pooling permette di ottenere un vettore descrittore che aggrega informazioni locali e globali dell'immagine a diversi livelli di scala. Questo vettore può essere usato per confrontare immagini in maniera efficace per il retrieval.



> COMPUTE_BB_MAT()


Questa funzione serve a localizzare oggetti simili all'interno di un'immagine del database rispetto ad una patch di query contenente un singolo oggetto.

Il flusso di alto livello è:

1. Per ogni patch della query:
    - Estrai le feature regionali dell'intera immagine del database con i relativi bounding box associati (metodo get_im_feature_by_path)

    - Estrai un vettore di feature aggregato dalla patch di query (metodo compute_im_feature con MAC pooling)

    - Calcola la similarità tra ogni feature regionale dell'immagine e la feature della patch tramite prodotto scalare (matrice di similarità)

2. Applica una maschera alla matrice di similarità per sopprimere i punteggi elevati dei BB più grandi. L'idea è che BB grandi coprono porzioni significative dell'immagine e probabilmente contengono oggetti diversi, quindi non vogliamo recuperarli.

3. Trova l'indice del BB con similarità mascherata massima, che dovrebbe corrispondere alla posizione dell'oggetto simile nell'immagine di database.

4. Aggiunge questo BB alla lista da ritornare come risultato.

Quindi in sintesi, confronta le feature della patch contenente un singolo oggetto con tutte le regioni dell'immagine del database per trovare il BB che meglio localizza l'oggetto simile all'interno dell'immagine. La maschera sui BB grandi aiuta ad affinare la localizzazione.


#### Parametri sui quali posso agire:
- ovr = overlap ratio
- max scale lavel
- area tolerance


### Confronti che posso fare:
- VGG16 vs VGG19 vs U-Net
devo ricordarmi di dire che vgg lavora con immagini a 3 canali in ingresso, quindi quello che ho fatto è stato aprire le immagini con convert('RGB'). In questo modo il canale viene replicato 3 volte identico a se stesso
- Quando utilizzo Unet invece, il modello che abbiamo è stato addestrato indicando che in input ci devono essere immagini a singolo canale, quindi le apro con convert('L')