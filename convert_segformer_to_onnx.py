# Standard library imports
from PIL import Image

# Third-party libraries
import onnx
import torch
from skimage import io
from torchvision import transforms
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class ONNXExporter:
    def __init__(self, model_name, as_fp32=False):
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if as_fp32:
            self.model.float()
        self.model.eval()

    def export_onnx_model(self, onnx_model_path, model_input_image_path):
        print("Exporting the model to ONNX format...")

        # prepare input
        orig_im = Image.open(model_input_image_path)
        image = self.processor(orig_im, return_tensors="pt").pixel_values.to(self.device)

        # Export the model to an ONNX file
        torch.onnx.export(self.model, image, onnx_model_path)

    @staticmethod
    def check_onnx_model(onnx_model_path):
        print("Checking ONNX model...")

        # Load the ONNX model
        model = onnx.load(onnx_model_path)

        # Check that the IR is well formed
        onnx.checker.check_model(model)

        # Uncomment to print a human readable representation of the graph
        # print(onnx.helper.printable_graph(model.graph))

        print("The ONNX model is well formed.")


if __name__ == "__main__":
    model_input_image_path = "example_image.jpg"
    onnx_path = "model.onnx"
    exporter = ONNXExporter("nickmuchi/segformer-b4-finetuned-segments-sidewalk", as_fp32=False)
    exporter.export_onnx_model(onnx_path, model_input_image_path)
    ONNXExporter.check_onnx_model(onnx_path)