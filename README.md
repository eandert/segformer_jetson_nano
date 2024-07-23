# Segformer Jetson Nano

This repository contains a quick and dirty C++ implementation of the Segformer model for the Jetson Nano.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)

## Installation

### ONNX Build Environment (No GPU Required)

To begin, clone this repository and navigate to it:

```
git clone https://github.com/eandert/segformer_jetson_nano
cd segformer_jetson_nano
```

Next, clone the repository for the desired segformer model from HuggingFace (or any other source). The implementation in this repository is set up to use the `nickmuchi/segformer-b4-finetuned-segments-sidewalk` model, so the following steps will guide you through that:

```
mkdir segformer_models
cd segformer_models
git clone https://huggingface.co/nickmuchi/segformer-b4-finetuned-segments-sidewalk
cd ../
```

This will clone the repository into the `segformer-b4-finetuned-segments-sidewalk` folder. All scripts in this repository will reference that folder structure. If you choose a different model, make sure to modify the name accordingly in the scripts. Now, let's install the requirements to build the ONNX file:

```
pip install -r python/onnx_build_requirements.txt
```

With the repositories cloned and the requirements installed, we can now verify that the inference is working correctly on the example image. This should create a file called `example_image_output.jpg` in the main directory, which should match the one included in the repository labeled `example_image_output.jpg`. If the output image does not match, there may be an issue and you should not proceed to create an ONNX file.

```
python3 python/segformer_inference.py
```

Assuming the image output matches, we can now create the ONNX file. You may see some warnings about CUDA, but that is okay as it will fall back to the CPU. At the end of this process, there should be no exceptions raised after it states "Checking the ONNX model..." and it should state "The ONNX model is well formed." if it has succeeded.

```
python3 convert_segformer_to_onnx.py
```

We will need the resulting `model.onnx` file after setting up the Jetson.

### Jetson Nano Deployment Environment

Exact installation steps for the Jetson Nano can vary depending on the state of your Jetson. However, in general, you will need the following libraries installed. It is recommended to use the installation scripts provided by the [jetson-inference](https://github.com/dusty-nv/jetson-inference) repository, as they simplify the installation process:

- CUDA
- TensorRT
- OpenCV
- jetson-utils
- jetson-inference

You will need to build the `jetson-inference` package from source, as it contains some tensor code that runs on CUDA. Follow the steps outlined [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md).

Now, clone this repository and navigate to it:

```
git clone https://github.com/eandert/segformer_jetson_nano
cd segformer_jetson_nano
```

Next, clone the repository for the desired segformer model from HuggingFace (or any other source). The implementation in this repository is set up to use the `nickmuchi/segformer-b4-finetuned-segments-sidewalk` model, so the following steps will guide you through that:

```
git clone https://huggingface.co/nickmuchi/segformer-b4-finetuned-segments-sidewalk
```

You will also need to install the following requirements to build the `.engine` file. Although a `requirements.txt` file is included, note that you typically cannot install TensorRT from pip and may need to follow NVIDIA's guidelines. It is possible that TensorRT was installed as part of the `jetson-inference` library mentioned earlier.

```
pip install -r engine_build_requirements.txt
```

We need to also go in and do some surgery on the `jetson-inference` library, unfortunately as it has a bug that doesn't support BGR but our model need that or the colors will be off and with incorrect boundaries (issue submitted here [Issue #1867](https://github.com/dusty-nv/jetson-inference/issues/1867)). It's a simple change but then you will need to follow the steps [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) again to build the project all over. Basically, you will keep getting an input error from the function because even though the template function is built to handle `isBGR` the type is for some reason not in the if statement. So do the following and then rebuild and re-install the `jetson-inference` project:

```
cd jetson-inference/c/
nano tensorConver.cu
Replace line 226 ""if( format == IMAGE_RGB8 )" with "if( format == IMAGE_RGB8 || format == IMAGE_BGR8 )"
```

With all the requirements installed, it's time to build the engine file. Copy the `model.onnx` file that you built earlier to the Jetson and run the following command. You may see some warnings about casting `int64` to `int32`, but this should not be an issue. Note that generating this engine on a different device with a GPU may not work due to compatibility issues between TensorRT/CUDA versions. It is recommended to build on the Jetson Nano itself.

```
python3 convert_onnx_to_tensorrt.py
```

After a short wait, you should have a file called `model.engine` in the directory. If everything went smoothly, you can now build the actual C++ file. Please note that the code in this file is not well-organized and may require cleaning up in the future.

```
make
```

If the build was successful, you should see the `tensorrt_segformer_inference` file. If you encountered build errors, it may be necessary to troubleshoot library and driver issues.

## Usage 

After following all the installation steps above, you should be able to run the program with the included example images and obtain outputs similar to those produced by the Python version during installation. Keep in mind that these steps were specifically designed for the Nvidia Jetson Nano 4GB, and results may vary if used on different hardware. Once you have confirmed the expected output on the example image, you can try feeding in other images. The colors on the Jetson version are a bit less clean than the Python output, but the areas generally line up - likely the bilinear interpolation needs to be worked on a bit more to get a perfect match.

```
./tensorrt_segformer_inference example_image.jpg example_image_output.jpg segformer-b4-finetuned-segments-sidewalk/config.json
```

## Performance

The performance of the deployed model was evaluated on an Nvidia Jetson Nano 4GB. The average times for each step of the process over ten runs are as follows:

- Deserialization time: 7506.63 ms
- Preprocessing time: 50.07 ms
- Inference time: 2185.49 ms
- Postprocessing time: 150.22 ms

Deserialization of the model is a one-time cost that occurs when the program starts. It involves loading the pre-trained model from disk into memory, which takes a significant amount of time. However, once the model is loaded, it can be used for multiple inferences without needing to be deserialized again.

Preprocessing is the fastest step, as it only involves reading the image from memory and copying it to the GPU. All preprocessing steps, including normalization, resizing, and feature extraction, are performed on the GPU using CUDA. The result is then copied to the input buffer of the model. Preprocessing typically takes around 50 ms.

The majority of the time is spent on the inference step. Currently, there have been no optimizations applied to the ONNX model or the engine that TensorRT generates. There is lots of potential for optimization, such as quantization, exploring changing the model to fp16, or other techniques that are currently not being utilized. Additionally, choosing a model that runs at an input size of 224 instead of 512 could provide lower accuracy but faster inference time.

Finally, postprocessing takes almost three times as long as preprocessing. This is because the postprocessing is done with OpenCV on the CPU, as it was not ported to CUDA due to time constraints. It is possible to move these OpenCV functions to the GPU or take advantage of output tensor parsing, but this has not been implemented yet.

Please note that these times may vary depending on the specific hardware and software configuration of your system.

Project Link: [https://github.com/eandert/segformer_jetson_nano](https://github.com/eandert/segformer_jetson_nano)
