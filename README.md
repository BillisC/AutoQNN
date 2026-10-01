# AutoQNN
QNN inference for autonomous driving applications using QCS6490, or similar.

## Milestones
- [x] Basic QNN inference
- [ ] GStreamer integration
- [ ] Multi-camera support
- [ ] Output features passing

## Project structure
```
docs/ <-- Documentation
external/ <-- external libraries
src/
    core/ <-- Core wrapper classes
    log/  <-- Logging functions (from Qualcomm)
    qnn/  <-- QNN types (from Qualcomm)
    utils/ <-- Tensor / buffer management tools etc.
```

A pre-compiled EfficientNet-lite model is included as a static library for inference.

The camera supplies packed RGB frames at 224 × 224. The bundled model declares
an input shape of `[1, 224, 3, 224]` and applies a `[0, 3, 1, 2]` transpose;
the application reorders camera data with `[0, 2, 3, 1]` before copying it.
Input bytes represent `(pixel - 127) / 128`. The output is a `[1, 1000]`
unsigned 8-bit fixed-point Softmax tensor with scale `1/256` and offset `0`,
returned as floats. Startup checks enforce these shapes, datatypes,
quantization parameters, and the 1000-label count. A replacement model must
have its own verified preprocessing and IO contract.

Model outputs are available only after successful execution using filled
inputs. Filling inputs, reloading the model, or a failed execution invalidates
the previous output. Multi-graph execution runs each graph with its supplied
inputs; dependencies between graphs must be managed explicitly by the caller.

## Setup
1. Clone the repo locally.
2. Setup QNN SDK (https://docs.qualcomm.com/nav/home/general_setup.html?product=1601111740010412).
3. Get ARM's cross-compiler from Qualcomm SDKs or ARM.
4. Configure QCS6490 board with the latest Ubuntu Server or Qualcomm Linux, using Qualcomm Launcher (https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_Launcher).
5. Enable SSH access, generate key pairs, and copy them to the board:
```bash
ssh-keygen -t ecdsa -b 521
ssh-copy-id -i ~/.ssh/key user@boardip
```

## Build
1. Configure project with:
```bash
cmake -B build/ --toolchain aarch-linux-toolchain.cmake -DTARGET_KEY=".ssh/.." -DTARGET_USER="user" -DTARGET_IP="192.168.x.x" -DTARGET_DIR="/tmp"
```
2. Build project with:
```bash
cmake --build build/
```
3. Execute on board via SSH with:
```bash
cmake --build build/ --target target_run
```

## Inference results
| Model | Inference speed (ms) |
| ----- | -------------------- |
| EfficientNet-Lite4 |   4.2   |
