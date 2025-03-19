# Handwritten-notes-Denoising

This project provides tools to accelerate the collection of data needed to address mathematical equation detection and optical character recognition (OCR) in handwritten notes. Handwritten notes pose several difficulties for automatic extraction of text and equations, mainly because of the inherent noise created when notes are taken on lined or squared sheets, rather than on a blank sheet of paper. Our goal is to have a Convolutional Neural Network (CNN) trained to remove square grids from handwritten notes and to obtain a text-only, noise-free image.
The work was developed as part of the "Computer Vision and cognitive systems" examination at the University of Modena and Reggio Emilia.

> **_Paper_**: [Handwritten Notes Denoising](https://github.com/cleb98/Handwritten-notes-Denoising/blob/master/Handwritte-notes%20Denoising.pdf)

> **_Presentation_**: [Handwritten Notes Denoising](presentation.pdf)
## Aknowledgements

|AUTHORs|CONTACTs|GITHUBs|
|-|-|-|
|Olmo Baldoni|[325524@studenti.unimore.it](mailto:325524@studenti.unimore.it)|[olmobaldoni](https://github.com/olmobaldoni)|
|Cristian Bellucci|[322906@studenti.unimore.it](mailto:322906@studenti.unimore.it)|[cleb98](https://github.com/cleb98)|
|Danilo Caputo|[246019@studenti.unimore.it](mailto:246019@studenti.unimore.it)|[Ilodan](https://github.com/IloDan)|

---

## Usage

The data generation folder contains all the scripts to generate the dataset and compute the mean and variance.

The Unet folder contains the codes for training and inferring the model.

The crop-warp folder contains the scripts for image warping.

The retrieval folder contains the code for image retrieval.

## License

MIT
