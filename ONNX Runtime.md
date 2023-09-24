

## Use ONNX Runtime with the platform of your choice

[ONNX Runtime | Home](https://onnxruntime.ai/index.html#getStartedTable)

Select the configuration you want to use and run the corresponding installation script.
ONNX Runtime supports a variety of hardware and architectures to fit any need.



![image-20230924112644224](ONNX Runtime.assets/image-20230924112644224.png)



![image-20230924112719715](ONNX Runtime.assets/image-20230924112719715.png)

## Use ONNX Runtime with your favorite language

[ONNX Runtime | Home](https://onnxruntime.ai/)



### python

```python
import onnxruntime as ort

# Load the model and create InferenceSession
model_path = "path/to/your/onnx/model"
session = ort.InferenceSession(model_path)

# Load and preprocess the input image inputTensor
...

# Run inference
outputs = session.run(None, {"input": inputTensor})
print(outputs)
```



### c#

```c#
using Microsoft.ML.OnnxRuntime;

// Load the model and create InferenceSession
string model_path = "path/to/your/onnx/model";
var session = new InferenceSession(model_path);

// Load and preprocess the input image to inputTensor
...

// Run inference
var outputs = session.Run(inputTensor).ToList();
Console.WriteLine(outputs[0].AsTensor()[0]);
```



### JavaScript

```javascript
import * as ort from "onnxruntime-web";

// Load the model and create InferenceSession
const modelPath = "path/to/your/onnx/model";
const session = await ort.InferenceSession.create(modelPath);

// Load and preprocess the input image to inputTensor
...

// Run inference
const outputs = await session.run({ input: inputTensor });
console.log(outputs);
```



### Java

```java
import ai.onnxruntime.*;

// Load the model and create InferenceSession
String modelPath = "path/to/your/onnx/model";
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(modelPath);

// Load and preprocess the input image inputTensor
...

// Run inference
OrtSession.Result outputs = session.run(inputTensor);
System.out.println(outputs.get(0).getTensor().getFloatBuffer().get(0));
```

### C++



```c++
#include "onnxruntime_cxx_api.h"

// Load the model and create InferenceSession
Ort::Env env;
std::string model_path = "path/to/your/onnx/model";
Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });

// Load and preprocess the input image to 
// inputTensor, inputNames, and outputNames
...

// Run inference
std::vector outputTensors =
 session.Run(Ort::RunOptions{nullptr}, 
 			inputNames.data(), 
			&inputTensor, 
			inputNames.size(), 
			outputNames.data(), 
			outputNames.size());

const float* outputDataPtr = outputTensors[0].GetTensorMutableData();
std::cout << outputDataPtr[0] << std::endl;
```





























