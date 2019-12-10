import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

from PIL import Image

class ModelData(object):
    MODEL_PATH = "resnet50-infer-5.uff"
    INPUT_NAME = "input"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "output"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32
    
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


# The UFF path is used for TensorFlow models. You can convert a frozen TensorFlow graph to UFF using the included convert-to-uff utility.
def build_engine_uff(model_file):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        builder.max_workspace_size = 1 << 30
        # We need to manually register the input and output nodes for UFF.
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        # Load the UFF model and parse it in order to populate the TensorRT network.
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)
    
    
def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


def main():
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    
    # Get images, models 
    test_image = '/tensorrt/data/resnet50/binoculars.jpeg'
    uff_model_file = '/tensorrt/data/resnet50/resnet50-infer-5.uff' 

    # Build a TensorRT engine.
    with build_engine_uff(uff_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            test_case = load_normalized_test_case(test_image, h_input)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            print(np.argmax(h_output))






if __name__ == '__main__':
    
    main()
