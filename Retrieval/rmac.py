import os
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from model import UNet

from PIL import Image

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SearchEngine():

    def __init__(self, db_path: str = 'db', backbone: str = 'vgg16', min_scale_level: int = 1, max_scale_level: int = 3):
        """

        Search Engine for Content-Based Image Retrieval (CBIR)

        Args:
            db_path: path of the database images
            backbone: backbone model to use for feature extraction
            min_scale_level: minimum scale level for R-MAC pooling
            max_scale_level: maximum scale level for R-MAC pooling

        """

        self.eps = 1e-6

        # Check if the 'db_path' argument is of type string
        assert isinstance(db_path, str), "db_path must be a string"

        # Check if the 'db_path' argument is a valid path
        assert os.path.exists(db_path), "db_path must be a valid path"

        # Set the database path
        self.db_path = db_path

        self.db_name = self.db_path.replace("/data", "")

        logging.info(f"Database path is: {self.db_name}")

        self.images = sorted(os.listdir(self.db_path))

        logging.info(f"Number of images in the database: {len(self.images)}")

        self.images_paths = [os.path.join(self.db_path, i) for i in self.images]

        
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logging.info(f'Using device: {self.device}')
                                

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: torch.unsqueeze(x, 0))
                                             ])
        
        # Check if the 'backbone' argument is of type string
        assert isinstance(backbone, str), "backbone must be a string"

        # Depending on the value of 'backbone', initialize the CNN model       
        if backbone == 'vgg16':
            self.cnn = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            del self.cnn.classifier
        elif backbone == 'vgg19':
            self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
            del self.cnn.classifier
        elif backbone == 'densenet':
            self.cnn = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            del self.cnn.classifier
        elif backbone == 'trained_unet':
            self.cnn = UNet(input_channels=1, num_classes=1)
            self.cnn.load_state_dict(torch.load('trained_unet.pth', map_location=torch.device('cpu')))
        elif backbone == 'kaiming_unet_0':
            self.cnn = UNet(input_channels=3, num_classes=2)
            self.cnn.load_state_dict(torch.load('kaiming_unet_0.pth', map_location=torch.device('cpu')))
        elif backbone == 'kaiming_unet_1':
            self.cnn = UNet(input_channels=3, num_classes=2)
            self.cnn.load_state_dict(torch.load('kaiming_unet_1.pth', map_location=torch.device('cpu')))
        elif backbone == 'kaiming_unet_2':
            self.cnn = UNet(input_channels=3, num_classes=2)
            self.cnn.load_state_dict(torch.load('kaiming_unet_2.pth', map_location=torch.device('cpu')))
        else:
            raise NotImplementedError("Only 'vgg16', 'vgg19', 'densenet', 'trained_unet' and 'kaiming_unet_0/1/2' are supported.")
        
        self.backbone = backbone
        
        logging.info(f"Backbone model is: {self.backbone}")
        
        # Iterate over all parameters of the CNN model
        for param in self.cnn.parameters():
            # Set requires_grad to False for each parameter
            # This disables training for these parameters
            param.requires_grad = False

        # Move the CNN model to the device
        self.cnn.to(self.device)

        # Set the CNN model to evaluation mode
        self.cnn.eval()

        # Set min scale level
        self.min_scale_level = min_scale_level

        # Set max scale level
        self.max_scale_level = max_scale_level

        logging.info(f"Min scale level is: {self.min_scale_level}")
        logging.info(f"Max scale level is: {self.max_scale_level}")

        # Initialize the database feature matrix
        self.db_feature_mat = self.compute_db_feature_mat()


    def rmac(self, x: torch.Tensor, pool_mode: str = 'max', ovr: float = 0.4, padding: int = 0): 
        """

        Regional Maximum Activation of Convolutions (R-MAC) pooling

        Args:
            x: input tensor of shape (N, C, W, H)
            pool_mode: pooling mode, either 'max' or 'avg'
            ovr: overlap ratio
            eps: epsilon value for l2-normalization
            padding: padding size for pooling
        Returns:
            feature_vec: feature vector of shape (N, C, R)
            regions_ijhw: list of window coordinates at each scale

        """

        N, C, H, W  = x.shape  

        w = np.minimum(W, H) # window size at scale 1

        regions_ijhw = [] # List of window coordinates at each scale
        feature_vec = [] # List of features at each scale

        for l in range(self.min_scale_level, self.max_scale_level + 1):

            window_size = int(np.floor(2 * w/ (l + 1))) # window size at scale l
            window_size = np.maximum(window_size, 2) 
            
            window_stride = int(np.floor((1 - ovr) * window_size)) #  window stride at scale l --> 40% overlap
            window_stride = np.maximum(window_stride, 1)

            if pool_mode == 'max':
                new_x = F.max_pool2d(x, kernel_size=window_size, stride=window_stride, padding=padding, ceil_mode=True)
            elif pool_mode == 'avg':
                new_x = F.avg_pool2d(x, kernel_size=window_size, stride=window_stride, padding=padding, ceil_mode=True)
            else:
                raise ValueError('Invalid pool_mode "%s"' % pool_mode)
            
            new_H, new_W = new_x.size()[2:] # new height and width of the feature map at scale l

            # Get the coordinates of each window at scale l
            for i in range(new_H):
                for j in range(new_W):
                    regions_ijhw.append([i * window_stride, j * window_stride, window_size, window_size])

            # Flatten the tensor and append it to the feature vector --> poiché la dimensione del tensore new_x dopo il pooling è sempre diversa, dobbiamo prima effetture un reshape e poi concatenare per ottenere un unico vettore contenente tutti i valori di pooling delle varie scale
            feature_vec.append(new_x.view(N, C, -1))

        # Concatenate the feature vectors at each scale
        feature_vec = torch.cat(feature_vec, dim=2) 

        regions_ijhw = np.array(regions_ijhw)

        feature_vec = feature_vec / (torch.linalg.norm(feature_vec, dim=1, keepdim=True) + self.eps)

        feature_vec = feature_vec.transpose(1, 2) # (N, C, R ) -> (N, R, C)

        return feature_vec, regions_ijhw
    
        
    def compute_img_feature_vec(self, img: torch.Tensor, pool_type: str = 'rmac'):
        """

        Extracts the R-MAC/MAC features from the activation map of the last convolutional layer of the network

        Args:
            img: input image tensor
        Returns:
            regions_ijhw: list of window coordinates
            feature_vec: feature vector 
            agg_feature_vec: aggregated feature vector 

        """

        assert img.dim() == 4, "img must be a 4D tensor"


        img_h, img_w = img.size()[2:] # H: img height, W: img witdh



        with torch.no_grad():
            if self.backbone in ['vgg16', 'vgg19', 'densenet']:
                actmap = self.cnn.features(img)
            else:
                activations = {}

                x = self.cnn.down_conv1(img)[0]
                activations['conv1'] = x

                x = self.cnn.down_conv2(x)[1]
                activations['conv2'] = x

                x = self.cnn.down_conv3(x)[1]
                activations['conv3'] = x

                x = self.cnn.down_conv4(x)[1]
                activations['conv4'] = x

                x = self.cnn.double_conv(x)
                activations['bottleneck'] = x

                actmap = activations['bottleneck']

        actmap_h, actmap_w = actmap.size()[2:] # H: actmap height, W: actmap witdh

        # compute the ratio between the image and the activation map
        ratio_h = img_h / actmap_h
        ratio_w = img_w / actmap_w

        if pool_type == 'rmac':

            # compute the R-MAC pooling
            feature_vec, regions_ijhw = self.rmac(actmap)

            # back project the window coordinates to the original image space
            regions_ijhw = regions_ijhw * np.array([ratio_h, ratio_w, ratio_h, ratio_w])
            regions_ijhw = np.floor(regions_ijhw)

            feature_vec = feature_vec.squeeze(0) # (1, R, C) -> (R, C)

            # combine the collection of regional feature vectors into a single image vector by summing them and l2-normalizing in the end
            agg_feature_vec = torch.sum(feature_vec, dim=0) # (R, C) -> (C,)

            agg_feature_vec = agg_feature_vec / torch.linalg.norm(agg_feature_vec, dim=0, keepdim=True) + self.eps


            return {"regions_ijhw": regions_ijhw,
                    "feature_vec": feature_vec,
                    "agg_feature_vec": agg_feature_vec}
        
        elif pool_type == 'mac':
            
            # compute the MAC pooling
            feature_vec = nn.AdaptiveAvgPool2d((1, 1))(actmap).squeeze()

            # l2-normalize the feature vector
            feature_vec = feature_vec / torch.linalg.norm(feature_vec, dim=0, keepdim=True) + self.eps

            return {'regions_ijhw': np.array([[0, 0, img_h, img_w]]),
                    'feature_vec': feature_vec.unsqueeze(0),
                    'agg_feature_vec': feature_vec}

        else:
            raise NotImplementedError("Only 'rmac' and 'mac' are supported.")
    

    def compute_db_feature_mat(self):
        """

        Computes the aggregated feature matrix of the database images

        Args:
            None
        Returns:
            db_feature_mat: database aggregated feature matrix

        """

        

        db_feature_mat_path = f"{self.db_name}_feature_mat_{self.backbone}.pth"

        if os.path.exists(db_feature_mat_path):

            logging.info("Loading database feature matrix...")

            db_feature_mat = torch.load(db_feature_mat_path)
            
        else:

            logging.info("Computing database feature matrix...")

            images = sorted(os.listdir(self.db_path))

            db_feature_mat = []

            for i in tqdm(images):
                if self.backbone in ['vgg16', 'vgg19', 'densenet', 'kaiming_unet_0', 'kaiming_unet_1', 'kaiming_unet_2']:
                    img = Image.open(os.path.join(self.db_path, i)).convert('RGB')
                else:
                    img = Image.open(os.path.join(self.db_path, i)).convert('L')
                img = self.transform(img).to(self.device)
                img_dict = self.compute_img_feature_vec(img=img)
                db_feature_mat.append(img_dict["agg_feature_vec"])
            
            db_feature_mat = torch.stack(db_feature_mat)

            torch.save(db_feature_mat, db_feature_mat_path)
        
        return db_feature_mat
    

    def compute_top_matches(self, img: torch.Tensor, top_k: int = 10):
        """
        Computes the top-k matches between the query image and the database images

        Args:
            img: input image tensor
            db_agg_feature_mat: database aggregated feature matrix
            top_k: number of top matches to return
        Returns:
            results: list of tuples (image_path, score)
            
        """

        logging.info("Computing top-k matches...")

        img_dict = self.compute_img_feature_vec(img=img)
        sim = torch.nn.CosineSimilarity(dim=1)
        sim_out = sim(img_dict["agg_feature_vec"], self.db_feature_mat)
        scores, indexes = torch.topk(sim_out, k=top_k)

        results = []

        for i in range(len(indexes)):
            results.append((self.images_paths[indexes[i]], scores[i].item()))

        return results
    

    def masked_img(self, img: torch.Tensor, query_bbs: torch.Tensor, show_img: bool = False):
        '''

        Helper function for creating bounding box masked image and 
        patches containing single objects
        
        Args:
            img: CHW, rgb tensor image
            query_bbs: a (n, 4) tensor representing xywh bounding boxes
            show_img: if True, shows the original image, the masked image and the patches
        Return:
            masked_img: image with region outside bounding boxes masked by zeros
            patches: list of n CHW, rgb patches containing single object
        '''
        
        patches = []
        masked = torch.zeros_like(img)
        for bb in query_bbs:
            x_left = bb[0].long()
            x_right = (bb[0] + bb[2]).long()
            y_up = bb[1].long() 
            y_down = (bb[1] + bb[3]).long()
            patches.append(img[:, :, y_up:y_down, x_left:x_right])
            masked[:, :, y_up:y_down, x_left:x_right] = img[:, :, y_up:y_down, x_left:x_right]

        if show_img:
            
            # original image
            plt.figure()
            plt.imshow(img.squeeze(0).permute(1, 2, 0))
            plt.title('Original image')
            plt.axis("off")
            plt.show()

            # masked image
            plt.figure()
            plt.imshow(masked.squeeze(0).permute(1, 2, 0))
            plt.title('Masked image')
            plt.axis("off")
            plt.show()

            for patch in patches:
                plt.figure()
                plt.imshow(patch.squeeze(0).permute(1, 2, 0))
                plt.title('Patch')
                plt.axis("off")
                plt.show()

        return masked, patches
    
    
    def compute_bb_mat(self, patches: torch.Tensor, img_path: str, tolerance: float = 0.2):
        """

        Computes the bounding box matrix for the retrieved images 

        Args:
            patches: list of n CHW, rgb patches containing single object
            img_path: path of the retrieved image
        Returns:
            ret_bbs: a (n, 4) tensor representing xywh bounding boxes
                    
        """
        
        ret_bbs = []

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).to(self.device) 
        img_dict = self.compute_img_feature_vec(img=img, pool_type='rmac')

        sim = torch.nn.CosineSimilarity(dim=1) 

        bb_areas = img_dict['regions_ijhw'][:,2] * img_dict['regions_ijhw'][:,3]
        
        for patch in patches:
            patch_dict = self.compute_img_feature_vec(img=patch, pool_type='mac')

            # Calculates the similarity between each regional feature in the image and the patch feature via scalar product
            sim_out = sim(patch_dict['agg_feature_vec'], img_dict['feature_vec'])

            patch_areas = patch_dict['regions_ijhw'][:,2] * patch_dict['regions_ijhw'][:,3]

            # # Creates a mask for the similarity scores that are within the tolerance range
            # mask = (bb_areas >= patch_areas[0] * (1-tolerance)) & (bb_areas <= patch_areas[0] * (1+tolerance))

            min_range = patch_areas[0] - (tolerance * patch_areas[0])
            max_range = patch_areas[0] + (tolerance * patch_areas[0])

            mask =  [min_range <= bb_area <= max_range for bb_area in bb_areas]

            mask = torch.tensor(mask)

            # Apply the mask to the similarity scores
            sim_out = sim_out * mask


            if not torch.all(sim_out == 0):
                # Selects the region with the highest similarity score
                _, indexes = torch.max(sim_out, dim=0)
                bb = img_dict['regions_ijhw'][indexes.item()]
                ret_bbs.append(bb)

        ret_bbs = np.array(ret_bbs)

        return ret_bbs

    
    def retrieve_object(self, img: torch.Tensor, query_bbs: torch.Tensor, tolerance: float = 0.2):
        """
        
        Retrieves images in database containing similar objects, and locate them 

        Args:
            img: input image tensor
            query_bbs: a (n, 4) tensor representing xywh bounding boxes
        Returns:
            results: list of tuples (image_path, score, bounding_box)
        """

        masked, patches = self.masked_img(img=img, query_bbs=query_bbs)
        top_k_images = self.compute_top_matches(img=masked, top_k=10)

        logging.info("Computing bounding boxes for retrieved images...")

        results = []

        for img_path, score in tqdm(top_k_images):
            bb_mat = self.compute_bb_mat(patches=patches, img_path=img_path, tolerance=tolerance)
            results.append((img_path, score, bb_mat))

        # Remove tuples with no bounding boxes
        results = [i for i in results if i[2].size != 0]
            
        return results
    

