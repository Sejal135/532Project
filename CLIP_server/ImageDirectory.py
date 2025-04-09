import os
import numpy as np

class ImageDirectory:
    def __init__(self, path: str):
        assert os.path.exists(path), f"Path {path} does not exist"
        self.path = path

    def __len__(self):
        return len(os.listdir(self.path))
    
    def get_image(self, image_name: str):
        image_path = os.path.join(self.path, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_name} not found in directory {self.path}")
        
        # Assume for now that images will be numpy arrays
        return np.load(image_path)