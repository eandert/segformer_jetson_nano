# Standard library imports
import glob
import json
import os
import time

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
    def __init__(self, model_name, config_path, preproc_config_path, use_cv2 = True, as_float=False):
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
        self.use_cv2 = use_cv2
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

    def run_inference(self, input_path, output_path):
        original_image = io.imread(input_path)
        orig_im_size = original_image.shape[0:2]
        if self.use_cv2:
            preproc_image = self.preprocess_image_cv2(original_image, self.model_input_size)
            preproc_image = preproc_image.to(self.device)
        else:
            preproc_image = self.processor(original_image, return_tensors="pt").pixel_values.to(self.device)
        result = self.inference(preproc_image)
        result_image = self.postprocess_image(result[0][0], orig_im_size)
        self.annotate_result(result_image, input_path, self.id2label, output_path)
        return result_image

    def annotate_result(self, result_image, input_path, id2label, output_path):
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
        orig_image = Image.open(input_path)

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

        # Save the result to the specified output path
        overlay_image.save(output_path)

if __name__ == "__main__":
    # List of configurations
    configurations = [
        ("nickmuchi/segformer-b4-finetuned-segments-sidewalk", "segformer_models/segformer-b4-finetuned-segments-sidewalk/config.json", "segformer_models/segformer-b4-finetuned-segments-sidewalk/preprocessor_config.json"),
        # Add more configurations as needed
    ]

    # Find all .jpg images within the input_images folder
    image_paths = glob.glob('input/*.jpg')
    total_time_per_configuration = []  # Store total time taken per configuration

    for model_name, config_path, preprocessor_config_path in configurations:
        configuration_name = model_name.split("/")[-1]  # Extract configuration name from the model name
        print(f"Starting inference with configuration: {configuration_name}")  # Print the name of the configuration

        start_time_config = time.time()  # Start time for this configuration
        segformer = SegformerInference(model_name, config_path, preprocessor_config_path)
        
        times_per_image = []  # Store times for each image processed with this configuration
        
        for image_path in image_paths:
            # Extract the filename from input folder and create a new filename for the output folder
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            new_file_name = f"{name_without_ext}_" + configuration_name + "_processed.jpg"
            new_file_path = os.path.join("output", new_file_name)

            start_time_image = time.time()  # Start time for this image
            segformer.run_inference(image_path, new_file_path)
            end_time_image = time.time()  # End time for this image
            
            time_taken_image = end_time_image - start_time_image
            times_per_image.append(time_taken_image)  # Store time taken for this image
            
            print(f"Inference completed for {image_path} using configuration {configuration_name} in {time_taken_image:.2f} seconds")
        
        end_time_config = time.time()  # End time for this configuration
        total_time_config = end_time_config - start_time_config
        total_time_per_configuration.append(total_time_config)  # Store total time taken for this configuration
        
        avg_time_per_image = sum(times_per_image) / len(times_per_image) if times_per_image else 0
        print(f"Average time per image for configuration {configuration_name}: {avg_time_per_image:.2f} seconds")
        print(f"Total time for configuration {configuration_name}: {total_time_config:.2f} seconds")

    # Calculate and print the overall average time per configuration
    overall_avg_time_per_config = sum(total_time_per_configuration) / len(total_time_per_configuration) if total_time_per_configuration else 0
    print(f"Overall average time per configuration: {overall_avg_time_per_config:.2f} seconds")