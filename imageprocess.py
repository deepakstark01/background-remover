#imageprocess.py
import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from models import ISNetDIS  
from data_loader_cache import normalize, im_reader, im_preprocess

class ImageSegmenter:
    def __init__(self, model_path="./saved_models/isnet.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize parameters
        self.hypar = {
            "model_path": os.path.dirname(model_path),
            "restore_model": os.path.basename(model_path),
            "interm_sup": False,
            "model_digit": "full",
            "seed": 0,
            "cache_size": [1024, 1024],
            "input_size": [1024, 1024],
            "crop_size": [1024, 1024],
            "model": ISNetDIS()
        }
        
        # Initialize transform
        self.transform = transforms.Compose([
            self.GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        ])
        
        # Build model
        self.net = self._build_model()

    class GOSNormalize(object):
        def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            self.mean = mean
            self.std = std

        def __call__(self, image):
            return normalize(image, self.mean, self.std)

    def _build_model(self):
        net = self.hypar["model"]

        if self.hypar["model_digit"] == "half":
            net.half()
            for layer in net.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()

        net.to(self.device)

        if self.hypar["restore_model"] != "":
            model_path = os.path.join(self.hypar["model_path"], self.hypar["restore_model"])
            net.load_state_dict(torch.load(model_path, map_location=self.device))
            
        net.eval()
        return net

    def _load_image(self, image_path):
        im = im_reader(image_path)
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[:, :, :3]
        im, im_shp = im_preprocess(im, self.hypar["cache_size"])
        im = torch.divide(im, 255.0)
        shape = torch.from_numpy(np.array(im_shp))
        return self.transform(im).unsqueeze(0), shape.unsqueeze(0)

    def _predict(self, inputs_val, shapes_val):
        """Predict mask from input"""
        if self.hypar["model_digit"] == "full":
            inputs_val = inputs_val.type(torch.FloatTensor)
        else:
            inputs_val = inputs_val.type(torch.HalfTensor)

        inputs_val_v = Variable(inputs_val, requires_grad=False).to(self.device)
        ds_val = self.net(inputs_val_v)[0]
        pred_val = ds_val[0][0, :, :, :]

        # Resize to original image size
        pred_val = torch.squeeze(F.upsample(
            torch.unsqueeze(pred_val, 0),
            (shapes_val[0][0], shapes_val[0][1]),
            mode='bilinear'
        ))

        # Normalize prediction
        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val - mi) / (ma - mi)

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)
    
    def process_image(self, image_path):
        """
        Process a single image and return both the RGBA image and mask
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: (PIL.Image, PIL.Image) - (RGBA image with transparent background, mask)
        """
        # Load and process image
        image_tensor, orig_size = self._load_image(image_path)
        mask = self._predict(image_tensor, orig_size)
        
        # Convert mask to PIL Image
        pil_mask = Image.fromarray(mask).convert('L')
        
        # Load original image and create RGBA version
        im_rgb = Image.open(image_path).convert("RGB")
        im_rgba = im_rgb.copy()
        im_rgba.putalpha(pil_mask)
        
        return im_rgba, pil_mask

# Example usage:
if __name__ == "__main__":
    # Initialize the segmenter
    segmenter = ImageSegmenter(model_path="./saved_models/isnet.pth")
    
    # Process an image
    input_image_path = "joke.png"
    rgba_image, mask = segmenter.process_image(input_image_path)
    
    # Save results
    rgba_image.save("output_rgba.png")
    mask.save("output_mask.png")