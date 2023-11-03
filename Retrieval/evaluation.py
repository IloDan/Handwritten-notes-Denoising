import os
from PIL import Image
from rmac import SearchEngine
from tqdm import tqdm
import csv

def compute_mAP(search_engine: SearchEngine,
                                    top_k_threshold: int = 3):
    """Compute the mean average precision (mAP) for a search engine.

    Args:
        search_engine (SearchEngine): The search engine to evaluate.
        top_k_threshold (int, optional): The number of top matches to consider when computing the mAP. Defaults to 3.

    Returns:
        float: The mAP score.
    """  
    # Read the labels from the CSV file
    with open(f"{search_engine.db_name}/labels.csv", mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csv_data = {row['path']: row['label'] for row in csv_reader}

    # Compute the mAP for each image in the database
    ap_scores = []
    for image_path in tqdm(search_engine.images):
        # Load the query image and its corresponding label
        query_image_path = os.path.join(search_engine.db_path, image_path)


        
        query_label = csv_data.get(query_image_path, 'label')

        # Compute the top matches for the query image

        if search_engine.backbone in ['vgg16', 'vgg19', 'densenet', 'kaiming_unet_0', 'kaiming_unet_1', 'kaiming_unet_2']:
            query_image = Image.open(query_image_path).convert('RGB')
        else:
            query_image = Image.open(query_image_path).convert('L')

      
        query_image = search_engine.transform(query_image).to(search_engine.device)
        results = search_engine.compute_top_matches(query_image, top_k=top_k_threshold)

        # Count the number of correct matches
        num_correct = 0
        for result in results:
            match_image_path, _ = result
            match_image_path = match_image_path.replace('\\', '/')
            match_label = csv_data.get(match_image_path, 'label')
            if match_label == query_label:
                num_correct += 1

        # Compute the accuracy for this image
        accuracy = num_correct / top_k_threshold
        ap_scores.append(accuracy)

    # Compute the mean AP score
    mAP = sum(ap_scores) / len(ap_scores)

    return mAP


if __name__ == "__main__":
    se = SearchEngine(db_path="db_2/data", backbone="vgg16", max_scale_level=6)
    mAP = compute_mAP(se, top_k_threshold=10)
    print(mAP)