This folder contains scripts for exporting the KModel to ONNX format. These are still experimental and subject to change.

It uses uv for dependency management. To get started, install uv, and then `uv sync`.


The project exports a CLI. For options, run `uv run kokoro-onnx --help`.


TODO:
- validate the device is still working for GPU on the torch model
- quantize (8,16, 4?)
    - determine location of the majority of the weights/runtimes (bert?)
    - perhaps adjust to output separate modules for each component, so that quantization can be done on a per-module basis
- generally optimize the ONNX model


Process:

```bash
uv run kokoro-onnx export
# make sure it's working, eyeball the MSE score, it should be below 0.001 (usually around 0.0006)
uv run kokoro-onnx verify
# okay, now slim it
uv run --with onnxslim onnxslim kokoro.onnx kokoro_slimmed.onnx
mv kokoro_slimmed.onnx kokoro.onnx
# now quantize. First try it without many calibration samples to make sure everything is working
uv run kokoro-onnx quantize --samples 1
# really quantize it
uv run kokoro-onnx quantize
# verify it
uv run kokoro-onnx verify --onnx-path kokoro_quantized.onnx
# also, listen to the onnx_output.wav, and verify no distortion
```





```

