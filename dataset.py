from PIL import Image 
import os 
import numpy as np 
from torch.utils.data import Dataset 

class MapSatelliteDataset(Dataset):
    def __init__(self ,root_satellite , root_map , transform = None):
        self.root_satellite = root_satellite
        self.root_map = root_map
        self.transform = transform 

        self.satellite_images = os.listdir(root_satellite)
        self.map_images = os.listdir(root_map)
        self.length_dataset = max(len(self.satellite_images) , len(self.map_images)) # 1000 , 1500 
        self.satellite_len = len(self.satellite_images)
        self.map_len = len(self.map_images) 

    def __len__(self): 
        return self.length_dataset
    
    def __getitem__(self , index): 
        satellite_img = self.satellite_images[index % self.satellite_len] 
        map_img = self.map_images[index % self.map_len]

        zebra_path = os.path.join(self.root_satellite , satellite_img) 
        horse_path = os.path.join(self.root_map , map_img) 

        satellite_img = np.array(Image.open(zebra_path).convert('RGB')) 
        map_img = np.array(Image.open(horse_path).convert('RGB')) 

        if self.transform:
            augmentations = self.transform(image = satellite_img , image0= map_img) 
            satellite_img = augmentations['image'] 
            map_img = augmentations['image0'] 

        return satellite_img , map_img 

