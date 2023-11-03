from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import os
import torch

IMG_SIZE = 1024

ROOT='/work/cvcs_2023_group11/dataset_more_grids/'
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
    
    def random_sample(self, index):
        image, mask = self.__getitem__(index)
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title('Image')
        plt.subplot(1, 2, 2)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Mask')
        plt.savefig('random_sample.png')


def load_data():
    # Define the paths for the images and masks
    image_dir = os.path.join(ROOT, 'train_images')
    mask_dir = os.path.join(ROOT, 'train_masks')

    image_groups = os.listdir(image_dir)

    image_paths = []
    mask_paths = []
    pbar = tqdm(total=len(image_groups), desc=f'Loading images and masks')
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
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.1, shuffle=True
    )

    return train_image_paths, test_image_paths, train_mask_paths, test_mask_paths


def create_dataloaders():
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = load_data()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
      #  transforms.Normalize(mean=0.9699, std=0.0478)
    ])

    # Create training and testing datasets
    train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=transform)
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform=transform)

    # Define the dataloaders for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    pbar = tqdm(total=len(loader), desc=f'Calculating mean and std')
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        pbar.update(1)
    pbar.close()

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def main():

    train_loader, _ = create_dataloaders()
    mean, std = get_mean_std(train_loader)
    print(f'Mean: {mean}, Std: {std}')


if __name__ == '__main__':
    main()