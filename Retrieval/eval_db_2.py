from PIL import Image
from tabulate import tabulate
from rmac import SearchEngine
from evaluation import compute_mAP
from tqdm import tqdm

db_path = "db_2/data"
# model_list = ["vgg16", "vgg19", "densenet", "trained_unet", "kaiming_unet_0", "kaiming_unet_1", "kaiming_unet_2"]
model_list = ["kaiming_unet_1", "kaiming_unet_2"]
k_thresholds = [3, 5, 10, 20]

for model in tqdm(model_list):
    se = SearchEngine(db_path=db_path, backbone=model, max_scale_level=6)

    table_data = []

    for k in k_thresholds:
        mAP = compute_mAP(se, top_k_threshold=k)  
        table_data.append([k, mAP])

    table_headers = ["k_threshold", "mAP"]

    table = tabulate(table_data, headers=table_headers, tablefmt="rounded_grid")

    print(f"Model: {model}")
    print(table)

    with open(f'mAP_{model}_db_2.txt', 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(table)