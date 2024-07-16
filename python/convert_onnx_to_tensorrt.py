import tensorrt as trt

# Convert the ONNX model to TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open("model.onnx", 'rb') as model:
        if not parser.parse(model.read()):
            print ('Failed to parse the ONNX file')
            for error in range(parser.num_errors()):
                print (parser.get_error(error))
        else:
            # Specify the builder parameters (including the FP16 mode)
            builder.fp16_mode = False
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            # Build the engine
            print("Building engine file")
            engine = builder.build_cuda_engine(network)
            if engine is None:
                print('Failed to build the engine')
            else:
                print('Successfully built the engine')

                # Save the engine to a file
                with open("model.engine", "wb") as f:
                    f.write(engine.serialize())
