## Content Based Image Retrieval


### Background

**Preprocessing and CNN Activations.** Given an input image $I$ of size $W_I \times H_I$, the image is passed through a pre-trained CNN. The fully connected layers of the network are discarded, and the activations of a convolution layer are considered. These activations form a 3D tensor of dimensions $W \times H \times K$, where $K$ represents the number of feature channels, and $W$ and $H$ are the width and height of the feature maps, respectively.

**Maximum Activation of Convolutions (MAC).** The MAC representation calculates the maximum activation across each feature channel. This results in a vector of length $K$. Each component $f_i$ in the feature vector corresponds to the maximum activation of the $i$-th feature channel. The MAC representation is given by:
$$f = [f_1, ..., f_k, ..., f_K]^T,~~\text{with}~~f_k = \underset{x \in X_k}{\text{max}}~~x$$

**Regional Maximum Activation of Convolutions (R-MAC).** To capture regional information, the 3D tensor is divided into $R$ square regions sampled at different scales, with approximately 40% overlap between consecutives regions. The region size at the largest scale ($l=1$) is set as large as possible, i.e. its height and with are both equal to $\text{min}(W,H)$ of the 3D tensor. At every other scale $l$, the width and height of each region is $2\min(W,H)/(l+1)$.
The feature vector associated with each region can be expressed as
$$f_{R_i} = [f_{R_i,1}, ..., f_{R_i,k}, ..., f_{R_i,K}]^T,~~\text{with}~~f_{R_i,k} = \underset{x \in R_{i,k}}{\text{max}}~~x$$
The computed feature vectors are $l_2$-normalized, then summed into a single vector and $l_2$-normalized again.
$$F = \sum_{i=1}^N f_{R_i} = [\sum_{i=1}^N f_{R_i,1},...,\sum_{i=1}^N f_{R_i,k}, ..., \sum_{i=1}^N f_{R_i,K}]^T$$
The final dimension of $F$ is equal to the number of feature channels $K$.

### Image Retrieval Pipeline

1. Given a query image, it is first preprocessed and converted into a tensor.

2. The query image tensor is passed through the pre-trained CNN feature extractor to obtain the convolutional activation maps.

3. Regional maximum activation of convolutions (R-MAC) is applied on the activation maps to obtain a regional feature vector for each image region at multiple scales.

4. The regional feature vectors are $l_2$-normalized, summed and $l_2$-normalized again to obtain a single global image descriptor vector.

5. The query descriptor is compared against the precomputed database descriptors using cosine similarity to retrieve  the top $k$ most similar images.

#### Hyperparameters

- Backbone CNN: This refers to the pre-trained convolutional neural network used to extract features from the input image. The choice of backbone CNN can affect the performance of the image retrieval system.
- Overlap between regions: This is the amount of overlap between consecutive regions sampled from the 3D tensor. In this case, the overlap is set to approximately 40%.
- Number of scales: This refers to the number of scales at which the 3D tensor is sampled.

### Esperiments

### Evaluation Metrics

- Mean Average Precision (mAP)






---

### Object Retrieval Pipeline

1. Given a query image and a set of bounding boxes indicating objects of interest, the image regions outside the boxes are masked to zeros to obtain a masked query image.
2. The bounding box regions are extracted as separate patches containing individual objects.
3. The masked query image is passed through the pipeline to retrieve the top k database images similar to the whole masked scene.

4. For each retrieved database image, its regional descriptors and corresponding spatial coordinates are obtained using R-MAC pooling.

5. The descriptor of each query object patch is matched against the database regional descriptors to locate the most similar region based on cosine similarity.

6. The spatial coordinates of the best matching database region are returned as the predicted bounding box for that query object.

7. The predicted bounding boxes are aggregated across all retrieved images for each query object patch. The final object retrieval results consist of the top images with their relevance score and predicted bounding boxes.


---


Ecco le formule in LaTeX e un abbozzo di codice per calcolare il mAP:

\textbf{Formule}:

Precision@k:

$$Precision@k = \frac{TP@k}{k}$$

dove:

- $TP@k$ = numero di risultati positivi tra i primi k
- k = cutoff corrente 

Average Precision (AP):

$$AP = \frac{\sum_{k=1}^n Precision@k}{TP}$$

dove: 

- la sommatoria è per tutti i valori di k da 1 a n
- $TP$ = numero totale di risultati positivi per quella query

Mean Average Precision (mAP):

$$mAP = \frac{\sum_{q=1}^Q AP_q}{Q}$$

dove:

- la sommatoria è sulle Q query totali 
- $AP_q$ = AP per la query q

\textbf{Codice}:

```python
def compute_map(results, gt_labels, k):

  aps = []
  
  for q in range(num_queries):
  
    precisions = [] 
    
    for i in range(1,k+1):
    
      tp = # conteggio dei risultati positivi tra i primi i
      prec = tp / i
      precisions.append(prec)
      
    ap = sum(precisions) / num_positives
    
    aps.append(ap)

  map = sum(aps) / num_queries
  
  return map
```

Dove:

- results contiene i risultati del CBIR per ogni query 
- gt_labels contiene le label delle classi
- k è il numero di risultati su cui valutare

Spero questo chiarisca le formule e dia un'idea su come implementare il codice per calcolare il mAP! Fammi sapere se hai altre domande.