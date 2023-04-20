# python3 generate_model.py --help 
# python3 generate_model.py --network ResNet50 VGG16 -ch 3 4 -sh 224 112 -sw 112 224
# python3 generate_model.py --network all -ch 3 3 3 -sh 224 512 720 -sw 224 512 720

import tensorflow as tf
import numpy as np
from pathlib import Path
import types
import argparse

MODELS = {}
for model_name in dir(tf.keras.applications):
    model_class = "tf.keras.applications.{}".format(model_name)
    model_class = eval(model_class)
    if not isinstance(model_class, types.FunctionType):
        continue    
    MODELS[model_name] = model_class
AVAILABLE_NETWORKS = list(MODELS.keys())

_choices = ['all'] + AVAILABLE_NETWORKS
parser = argparse.ArgumentParser(description='TFlite generator (with dummy weights) for networks available on TensorFlow')
parser.add_argument('--network', nargs="+", default='all', choices=_choices, help='Specify the network to generate tflite for')
parser.add_argument('-ch','--channels', nargs="+", type=int, default=[3], help='Number of input channels', required=False)
parser.add_argument('-sh','--shape_h', nargs="+", type=int, default=[224], help='Network input shape height', required=False)
parser.add_argument('-sw','--shape_w', nargs="+", type=int, default=[224], help='Network input shape width', required=False)
parser.add_argument('-o', '--output_dir_path', type=str, default="./test_models_dummy_weights", help='output dir path', required=False)
parser.add_argument('--include_top', default=False, action='store_true', help='Add head', required=False)

def convert2tflite(model, tensors, path_output):
    def representative_data_gen():
        for input_value in tensors:
            yield [input_value]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    path_output = Path(path_output)
    path_output.parent.mkdir(exist_ok=True, parents=True)
    path_output.write_bytes(tflite_model)
    return path_output

class TFliteModel:
    # https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
    '''
    Wrapper class on tflite interpreter
    '''
    def __init__(self, tflite_model_file):
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=str(tflite_model_file))
        self.input_tensor_idxs = []
        for input_tensor_info in self.tflite_interpreter.get_input_details():
            self.input_tensor_idxs.append(input_tensor_info["index"])
        self.output_tensor_idxs = []
        for output_tensor_info in self.tflite_interpreter.get_output_details():
            self.output_tensor_idxs.append(output_tensor_info["index"])
        self.tflite_interpreter.allocate_tensors()

    def __call__(self, input_tensors):
        for input_tensor_idx, input_tensor in zip(self.input_tensor_idxs, input_tensors):
            self.tflite_interpreter.set_tensor(input_tensor_idx, input_tensor)
        self.tflite_interpreter.invoke()
        output_tensors = []
        for output_tensor_idx in self.output_tensor_idxs:
            output_tensor = self.tflite_interpreter.get_tensor(
                output_tensor_idx)
            output_tensors.append(output_tensor)
        return output_tensors

def set_random_weights(model):
    for layer in model.layers:
        if "Batch" in str(type(layer)):
            continue
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        nw_weights = []
        for weight in weights:
            # nw_weight = (np.random.random(weight.shape)-.5)
            # nw_weight = np.random.random(weight.shape)
            nw_weight = weight
            nw_weights.append(nw_weight)
        layer.set_weights(nw_weights)


if __name__ == '__main__':
    args = parser.parse_args()
    assert len(args.shape_h) == len(args.shape_w), "Invalid input specification"
    assert len(args.channels) == len(args.shape_w), "Invalid input specification"

    INPUT_SHAPES = list(zip(args.shape_h, args.shape_w, args.channels))

    summary = ""
    networks = args.network
    if "all" in args.network:
        networks = AVAILABLE_NETWORKS

    for model_name in networks:
        for input_shape in INPUT_SHAPES:
            path_output = "{output_dir}/test_model_{model_name}/test_model_{model_name}_{h}X{w}_dummy.tflite".format(output_dir=args.output_dir_path, model_name=model_name, h=input_shape[0], w=input_shape[1])
            path_output = Path(path_output)
            path_output.parent.mkdir(parents=True, exist_ok=True)

            if path_output.is_file():
                print("Tflite file for {} is already present at {}".format(model_name, path_output))
                continue

            print("Generating {} tflite file at {}".format(model_name, path_output))
            model = MODELS[model_name](input_shape = input_shape, include_top=args.include_top, weights=None)
            set_random_weights(model)

            # model.save(filepath="./output/{0}_{1}.h5".format(model.name,model.input_shape))
            # model.summary()

            get_input_sample_float = lambda input_shape : np.random.random(input_shape).astype(np.float32)[np.newaxis, ...]
            tensors = [get_input_sample_float(input_shape), get_input_sample_float(input_shape), get_input_sample_float(input_shape)]
            convert2tflite(model, tensors, path_output)
            print("TFltie file generated at {}".format(path_output))
