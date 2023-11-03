from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from rmac import SearchEngine


# Define the image transformation pipeline
utils_transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Lambda(lambda x: torch.unsqueeze(x, 0))
                                      ])


def show_query_img(img_path: str, bbs: list = None):
    """

    ### Show the query image.

    #### Args:
        img_path (str): Path of the query image.
        bbs (list): List of bounding boxes of the query image.

    """

    img = Image.open(img_path).convert('RGB')
    img = utils_transform(img)
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)


    if bbs is not None:
        draw = ImageDraw.Draw(img)
        for bb in bbs:
            x, y, w, h = bb
            draw.rectangle([(x, y), (x + w, y + h)], outline='green', width=5)
            

    plt.figure()
    plt.imshow(img)
    plt.title("Query image", fontsize=14)
    plt.axis("off")
    plt.show()


def show_top_matches(results: list):
    """

    Show the top matches.

    Args:
        results (list): List of tuples (path, score, *bbs) of the top matches.

    """
    num_results = len(results)
    num_rows = (num_results + 4) // 5  # Calcolo del numero di righe

    plt.figure(figsize=(15, 3*num_rows))
    plt.suptitle("Top Matches", fontsize=16)  

    for i, (path, score, *bbs) in enumerate(results):

        img = Image.open(path).convert('RGB')
        img = utils_transform(img)
        img = img.squeeze(0)
        img = transforms.ToPILImage()(img)

        if bbs:
            draw = ImageDraw.Draw(img)
            for bb in bbs[0]:
                y, x, h, w = bb
                draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=5)

        plt.subplot(num_rows, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Path: {path}", fontsize=8)  
        plt.xlabel(f"Score: {score:.2f}", fontsize=8)  
        plt.xticks([])  
        plt.yticks([])

    plt.show()
