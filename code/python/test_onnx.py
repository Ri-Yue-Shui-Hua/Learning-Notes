import onnxruntime as ort
import numpy as np


if __name__ == '__main__':
    session = ort.InferenceSession("test_model.onnx")
    x = np.random.randint(0, 10, (3,2))
    pad = np.random.randint(0, 10, (1,4))
    val = np.random.randint(0, 10)
    print('inital value: \n', x, "\n", pad,"\n", val)
    input = {"X": x, "pads": pad, "value": val}

    output = session.run("Y", input)
    print(output)
