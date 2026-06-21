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