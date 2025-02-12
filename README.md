This folder contains scripts for exporting the KModel to ONNX format. These are still experimental and subject to change.

It uses uv for dependency management. To get started, install uv, and then `uv sync`.


The project exports a CLI. For options, run `uv run kokoro-onnx --help`.


TODO:
- validate the device is still working for GPU on the torch model
- quantize (8,16, 4?)
    - determine location of the majority of the weights/runtimes (bert?)
    - perhaps adjust to output separate modules for each component, so that quantization can be done on a per-module basis
- generally optimize the ONNX model

