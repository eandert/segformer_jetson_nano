// Standard library headers
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Third-party library headers
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

// Jetson specific headers
#include <jetson-inference/tensorConvert.h>
#include <jetson-inference/tensorNet.h>
#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/imageFormat.h>

using namespace nvinfer1;

/**
    * @brief Custom logger class for TensorRT.
    * 
    * This class extends the `nvinfer1::ILogger` class and provides a custom implementation
    * for logging messages. It suppresses info-level messages and prints all other severity
    * levels to the standard output.
    */
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

/**
 * Generates a color map with the specified number of colors.
 *
 * @param numColors The number of colors to generate in the color map.
 * @return A vector of cv::Vec3b representing the generated color map.
 */
std::vector<cv::Vec3b> generateColorMap(int numColors) {
    std::vector<cv::Vec3b> colorMap;
    cv::RNG rng(72948); // Seed for random number generator

    for (int i = 0; i < numColors; ++i) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colorMap.push_back(cv::Vec3b(b, g, r)); // OpenCV uses BGR format
    }

    return colorMap;
}

/**
 * Parses a JSON file containing id-to-label mappings and returns a map of id-to-label pairs.
 *
 * @param filename The path to the JSON file.
 * @return A map of id-to-label pairs, where the key is an integer ID and the value is the corresponding label.
 */
std::map<int, std::string> parseId2Label(const std::string& filename) {
    // Open the JSON file
    std::ifstream file(filename);

    // Parse the JSON file into a json object
    nlohmann::json j;
    file >> j;

    // Access the id2label section
    nlohmann::json id2label = j["id2label"];

    // Create a map to store the id2label pairs
    std::map<int, std::string> id2labelMap;

    // Iterate over the id2label section and insert each key-value pair into the map
    for (nlohmann::json::iterator it = id2label.begin(); it != id2label.end(); ++it) {
        id2labelMap[std::stoi(it.key())] = it.value();
    }

    return id2labelMap;
}

cv::Mat hwc_to_cwh(cv::Mat& src) {
    std::vector<cv::Mat> channels(3);
    cv::split(src, channels);
    for (int i = 0; i < 3; i++) {
        cv::transpose(channels[i], channels[i]);
    }
    cv::Mat transposed;
    cv::merge(channels, transposed);  
    return transposed;
}

/**
 * Visualizes the segmentation result by generating a colored overlay image.
 *
 * @param final_output          Pointer to the array containing the final output probabilities for each pixel.
 * @param output_height         The height of the output segmentation map.
 * @param output_width          The width of the output segmentation map.
 * @param output_classes        The number of classes in the segmentation map.
 * @param im_path               The path to the input image.
 * @param output_image_path     The path to save the output visualization image.
 * @param id2label              A map that maps class IDs to class labels.
 * @param as_float              Flag indicating whether the final output probabilities are in float format.
 */
void visualize_result(float* final_output, int output_height, int output_width, int output_classes, std::string im_path, std::string output_image_path, std::map<int, std::string> id2label) {
    std::vector<cv::Vec3b> color_map = generateColorMap(35);
    color_map[0] = cv::Vec3b(0, 0, 0); // Set color for class 0 to black

    // Get the class that is most probable for each pixel in the output and convert CWH to HWC
    cv::Mat result_image = cv::Mat::zeros(output_height, output_width, CV_8UC1);
    for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
            float maxVal = -10000.0f;
            int maxIdx = 0;

            for (int c = 0; c < output_classes; c++) {
                int index = c * output_height * output_height + h * output_width + w;
                float val = final_output[index];

                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = c;
                }
            }

            result_image.at<uint8_t>(h, w) = maxIdx;
        }
    }

    cv::Mat colored_result = cv::Mat::zeros(result_image.size(), CV_8UC3);
    for (int class_id = 0; class_id < color_map.size(); ++class_id) {
        colored_result.setTo(color_map[class_id], result_image == class_id);
    }

    // Read the original image again, and apply the overlay. TODO(eandert): Use the original memory location.
    cv::Mat orig_image = cv::imread(im_path);
    cv::resize(colored_result, colored_result, orig_image.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat overlay_image;
    cv::addWeighted(orig_image, 0.5, colored_result, 0.5, 0, overlay_image);

    for (int i = 0; i < color_map.size(); ++i) {
        cv::rectangle(overlay_image, cv::Rect(0, i * 20, 20, 20), cv::Scalar(color_map[i]), -1);
        cv::putText(overlay_image, id2label[i] + " (ID: " + std::to_string(i) + ")", cv::Point(22, i * 20 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    cv::imwrite(output_image_path, overlay_image);
}

/**
 * @brief Entry point of the jetson segformer inference program. This is a large (and ugly) blob of code that performs the following steps:
 * TODO(eandert): Clean this up and turn it into C++ class. With tight time constraints you either get performant code, or clean code. This
 * is the former.
 *
 * It performs the following steps:
 * 1. Parses the command line arguments to ensure the correct number of arguments are provided.
 * 2. Sets up the necessary variables and parameters for input and output.
 * 3. Creates a TensorRT engine and deserializes it from the "model.engine" file.
 * 4. Allocates memory for the input and output buffers on the GPU.
 * 5. Loads the input image using OpenCV and converts it to RGB format.
 * 6. Preprocesses the image by normalizing and resizing it.
 * 7. Performs inference using the TensorRT engine.
 * 8. Copies the output data back to the CPU.
 * 9. Postprocesses the output data by visualizing the result.
 * 10. Cleans up allocated resources.
 *
 * @param argc The number of command line arguments.
 * @param argv An array of command line arguments.
 * @return 0 if the program executed successfully, -1 otherwise.
 */
int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <config.json>" << std::endl;
        return -1;
    }

    // Inputs
    int input_height = 512;
    int input_width = 512;
    int num_pixels = 3;
    int input_pixel_type = sizeof(uint8_t);
    int batch_size = 1;
    int preprocessed_input_size_bytes = input_height * input_width * num_pixels * sizeof(float);

    // Outputs
    int output_classes = 35; // Number of classes in the model
    int output_height = 128;
    int output_width = 128;
    int output_size = output_classes * output_height * output_width * sizeof(float);

    // Preprocessing
    float3 mean = make_float3(0.485f, 0.456f, 0.406f);
    float3 stdDev = make_float3(0.229f, 0.224f, 0.225f);
    float2 range = make_float2(0.0f, 1.0f);

    // Num bindings
    int num_bindings = 2;

    // Create CUDA events for profiling.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Make a CUDA stream
    cudaStream_t stream;
    cudaError_t cuda_err = cudaStreamCreate(&stream);

    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during cuda stream create: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Create a TensorRT engine.
    IRuntime* runtime = createInferRuntime(gLogger);
    std::ifstream engineFile("model.engine", std::ios::binary);
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)), std::istreambuf_iterator<char>());
    engineFile.close();
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    IExecutionContext* context = engine->createExecutionContext();

    // The model has 2 bindings.
    void* buffers[2];
    if(num_bindings != engine->getNbBindings()) {
        std::cerr << "Number of bindings in the engine file does not match." << std::endl;
        return -1;
    }
    
    for (int i = 0; i < num_bindings; ++i)
    {
        // Get the dimensions of the binding.
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        // Calculate the total number of elements in the binding.
        int numElements = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());

        // Get the data type of the binding.
        nvinfer1::DataType dataType = engine->getBindingDataType(i);
        // Calculate the size of each element in the binding.
        int elementSize = (dataType == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(__half);

        // Allocate memory for the binding.
        cudaMalloc(&buffers[i], batch_size * numElements * elementSize);
    }

    // Start timing.
    cudaEventRecord(start);

    // Load the input image using OpenCV in BGR color space.
    cv::Mat input_image_cv2 = cv::imread(argv[1], cv::IMREAD_COLOR);
    if(input_image_cv2.empty()) {
        std::cerr << "Error: could not load image." << std::endl;
        return -1;
    }
    int input_image_size_bytes = batch_size * input_height * input_width * num_pixels * input_pixel_type;

    // Resize the image to match the input size
    cv::Mat resized_image;
    cv::Size new_size(input_height, input_width);
    cv::resize(input_image_cv2, resized_image, new_size, 0, 0, cv::INTER_LINEAR);

    // Allocate memory for the input image on the GPU
    float* raw_image_cuda;
    cuda_err = cudaMalloc(&raw_image_cuda, input_image_size_bytes);

    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during malloc input: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Copy the preprocessed data to the GPU. TODO(eandert): Use zero copy here.
    cuda_err = cudaMemcpy(raw_image_cuda, resized_image.data, input_image_size_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during copy to input: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Allocate memory for the preprocessed image on the GPU
    float* preprocess_input_cuda;
    cuda_err = cudaMalloc(&preprocess_input_cuda, batch_size * input_height * input_width * num_pixels * sizeof(float));

    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during malloc of output: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Preprocess the image using the cudaTensorNormMeanRGB function.
    // TODO(eandert): Propose fix for the library as it seems to have a bug with BGR, meaning we had to modify the library itself to get this to work!
    cuda_err = cudaTensorNormMeanBGR(raw_image_cuda, IMAGE_BGR8, input_height, input_width,
                                     preprocess_input_cuda, input_height, input_width, range, mean, stdDev, stream);

    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during preprocessing: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Copy the preprocessed data to the GPU. TODO(eandert): Change to zero copy.
    cuda_err = cudaMemcpy(buffers[0], preprocess_input_cuda, batch_size * preprocessed_input_size_bytes, cudaMemcpyDeviceToDevice);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during copy to input: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Stop timing and print the elapsed time.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Preprocessing time: " << milliseconds << " ms" << std::endl;

    // Start timing.
    cudaEventRecord(start);

    // Perform inference.
    cudaDeviceSynchronize();
    context->execute(batch_size, buffers);
    cudaDeviceSynchronize();

    // Stop timing and print the elapsed time.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Inference time: " << milliseconds << " ms" << std::endl;

    // Start timing.
    cudaEventRecord(start);

    // Copy the output data back to the CPU.
    float* final_output = new float[output_size];
    cuda_err = cudaMemcpy(final_output, buffers[1], batch_size * output_size, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error during copy to input: " << cudaGetErrorString(cuda_err) << std::endl;
        return -1;
    }

    // Postprocess the output data. TODO(eandert): Change to zero copy and a CUDA kernal instead.
    std::map<int, std::string> id2label = parseId2Label(argv[3]);
    visualize_result(final_output, output_height, output_width, output_classes, argv[1], argv[2], id2label);

    // Stop timing and print the elapsed time.
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Postprocessing time: " << milliseconds << " ms" << std::endl;

    // Clean up.
    cudaDeviceSynchronize();
    std::cout << "Done. Cleaning up." << std::endl;
    for (int i = 0; i < num_bindings; ++i)
    {
        cudaFree(buffers[i]);
    }
    cudaFree(raw_image_cuda);
    cudaFree(preprocess_input_cuda);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
