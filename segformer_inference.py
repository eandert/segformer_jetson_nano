# Standard library imports
import json

# Third party imports
from PIL import Image, ImageDraw, ImageFont
from skimage import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

# Application specific imports
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class SegformerInference:
    def __init__(self, model_name, config_path, preproc_config_path, as_float=False):
        with open(config_path) as f:
            self.config = json.load(f)
        with open(preproc_config_path) as f:
            self.preproc_config = json.load(f)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if as_float:
            self.model.float()
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.as_float = as_float
        self.id2label = self.config['id2label'] # Get the mapping from class ID to name
        self.model_input_size = [self.preproc_config["size"], self.preproc_config["size"]]

    def preprocess_image_cv2(self, image_raw, model_input_size):
        # Resize the image
        image_resized = cv2.resize(image_raw, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image_resized - mean) / std
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype=np.float32, order="C")
        torch_image = torch.from_numpy(image)
        return torch_image

    def postprocess_image(self, result: torch.Tensor, im_size: list) -> np.ndarray:
        result = result.squeeze(0)
        result = F.interpolate(result.unsqueeze(0), size=im_size, mode='bilinear', align_corners=False).squeeze(0)
        result = result.argmax(0)
        im_array = result.byte().cpu().numpy()
        return im_array

    def inference(self, image):
        result = self.model(image)
        return result

    def run_inference(self, im_path):
        original_image = io.imread(im_path)
        orig_im_size = original_image.shape[0:2]
        preproc_image = self.preprocess_image_cv2(original_image, self.model_input_size)
        preproc_image = preproc_image.to(self.device)
        result = self.inference(preproc_image)
        result_image = self.postprocess_image(result[0][0], orig_im_size)
        self.visualize_result(result_image, im_path, self.id2label, self.as_float)
        return result_image
    
    def visualize_result(self, result_image, im_path, id2label, as_float):
        # Define a color map that maps each class ID to a specific color
        color_map = plt.get_cmap('tab20', 35).colors  # Get 35 distinct colors

        # Convert the color map to RGB format
        color_map = [color[:3] for color in color_map]

        # Set the color of the unlabeled class to clear (black)
        color_map[0] = [0, 0, 0]

        # Store the classes we see in the image
        existing_classes = []

        # Colorize the segmentation result using the color map
        colored_result = np.zeros((result_image.shape[0], result_image.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(color_map):
            colored_result[result_image == class_id] = (np.array(color) * 255).astype(np.uint8)
            existing_classes.append((class_id, id2label[str(class_id)], color))

        # Convert the colored result to a PIL image
        seg_image = Image.fromarray(colored_result)

        # Open the original image
        orig_image = Image.open(im_path)

        # Resize the segmentation image to the original size
        seg_image = seg_image.resize(orig_image.size, Image.NEAREST)

        # Blend the original image and the segmentation result
        overlay_image = Image.blend(orig_image, seg_image, alpha=0.5)

        # Draw the legend
        draw = ImageDraw.Draw(overlay_image)
        font = ImageFont.load_default()
        for i, (class_id, class_name, color) in enumerate(existing_classes):
            draw.rectangle([0, i * 20, 20, (i + 1) * 20], fill=tuple((np.array(color) * 255).astype(int)))
            draw.text((22, i * 20), f"{class_name} (ID: {class_id})", fill="white", font=font)

        # Save the result
        if as_float:
            overlay_image.save("example_image_output_float.jpg")
        else:
            overlay_image.save("example_image_output.jpg")

if __name__ == "__main__":
    segformer = SegformerInference("nickmuchi/segformer-b4-finetuned-segments-sidewalk", "segformer-b4-finetuned-segments-sidewalk/config.json", "segformer-b4-finetuned-segments-sidewalk/preprocessor_config.json")
    segformer.run_inference("./example_image.jpg")