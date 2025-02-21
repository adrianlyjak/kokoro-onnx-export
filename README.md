This folder contains scripts for exporting the KModel to ONNX format.

It uses uv for dependency management. To get started, install uv, and then `uv sync`.

Currently, it's expected that the latest version of kokoro is installed in a sibling directory.

The project exports a CLI. For options, run `uv run kokoro-onnx --help`.

```bash
uv run kokoro-onnx --help                                                         
 Usage: kokoro-onnx [OPTIONS] COMMAND [ARGS]...                                 
                                                                                
 Kokoro ONNX tools CLI                                                          
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ count                Analyzes an ONNX model, counting nodes or parameters by │
│                      operation type or name prefix.                          │
│ verify               Verify ONNX model output against PyTorch model output.  │
│ export               Export the Kokoro model to ONNX format.                 │
│ trial-quantization   Run quantization trials on individual nodes to measure  │
│                      their impact on model quality. Results are saved to a   │
│                      CSV file with columns: name, op_type, mel_distance,     │
│                      params, size                                            │
│ estimate-size        Estimate model size after quantization/casting based on │
│                      trial results and thresholds.                           │
│ export-optimized     Export an optimized model using both FP16 and INT8      │
│                      quantization based on trial results.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

It's all a bit exploratory. To export a quantized model:

First, export float 32 model to `kokoro.onnx`
```bash
uv run kokoro-onnx export 
```

Then, run trial quantization to generate a `quantization-trials.csv` file. This analyzes each node in the model, over a certain size threshold, to see how much if compromises the model quality. This is all a little silly, since, from what I can tell, 1) my "loss" function is imperfect, 2) we end up quantizing most everything anyway. This trials with "dynamic" (weight only) quantization, but can also do static (weight and activation) quantization

```bash
uv run kokoro-onnx trial-quantization
```

Then, finally export a quantized model, from `kokoro.onnx`, using the trial data. Unfortunately, `(^/decoder/generator/conv_post/Conv)` needs to be manually excluded, since it's not detected as affecting loss negatively, even though it adds a ton of static to the model output.

```bash
uv run kokoro-onnx export-optimized --quant-threshold=2 --quant-exclude '(^/decoder/generator/conv_post/Conv)'
```

You can optionally then verify the model, and analyze its contents. The torch and onnx output will be saved to `torch_output.wav` and `onnx_output.wav`, respectively.
```bash
uv run kokoro-onnx verify --onnx-model kokoro_optimized.onnx --text "Hello, world!" --voice "af_heart"
```

You can also count the nodes in the model.
```bash
uv run kokoro-onnx count --onnx-model kokoro_optimized.onnx --size --count-by 'op+dtype'
```
