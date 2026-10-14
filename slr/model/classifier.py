import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from slr.paths import DEFAULT_MODEL_PATH, resolve_path

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        if model_path is None:
            resolved_path = str(DEFAULT_MODEL_PATH)
        else:
            resolved_path = str(resolve_path(model_path))

        self.interpreter = Interpreter(
            model_path=resolved_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def predict_with_confidence(self, landmark_list, threshold=0.40):
        input_details_tensor_index = self.input_details[0]['index']

        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32)
        )

        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        squeezed = np.squeeze(result)
        max_conf = float(np.max(squeezed))

        if max_conf >= threshold:
            result_index = int(np.argmax(squeezed))
            return result_index, max_conf
        else:
            return 25, max_conf

    def __call__(self, landmark_list):
        result_index, _ = self.predict_with_confidence(landmark_list, threshold=0.50)
        return result_index