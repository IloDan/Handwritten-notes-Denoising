from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import os
from .config import BATCH_SIZE, IMG_SIZE, ROOT, NUM_WORKERS, MEAN, STD

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Define the root directory of your dataset
root_dir = ROOT

# Define the paths for the images and masks
image_dir = os.path.join(root_dir, 'train_images')
mask_dir = os.path.join(root_dir, 'train_masks')

image_groups = os.listdir(image_dir)

image_paths = []
mask_paths = []
pbar = tqdm(total=len(image_groups), desc=f'Loading images and masks - 0%')
for group in image_groups:
    group_image_dir = os.path.join(image_dir, group)
    group_mask_file = group + '.png'
    
    image_files = [os.path.join(group_image_dir, img) for img in os.listdir(group_image_dir)]
    mask_file = os.path.join(mask_dir, group_mask_file)
    
    image_paths.extend(image_files)
    mask_paths.extend([mask_file] * len(image_files))
    pbar.update(1)
pbar.close()


# Split the dataset into training and testing subsets
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths, test_size=1/3, shuffle=True)

# Define the transformation for the images and masks
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean= MEAN, std= STD)  
])

# Create training and testing datasets
train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=transform)
test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform=transform)

# Define the dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Number of training examples:", len(train_dataset))
print("Number of testing examples:", len(test_dataset))