This folder contains scripts for exporting the KModel to ONNX format.

It uses uv for dependency management. To get started, install uv, and then `uv sync`.

Currently, it's expected that the latest version of kokoro is installed in a sibling directory.

The project exports a CLI. For options, run `uv run kokoro-onnx --help`.

```bash
uv run kokoro-onnx --help
```

```
 Usage: kokoro-onnx [OPTIONS] COMMAND [ARGS]...                                 
                                                                                
 Kokoro ONNX tools CLI                                                          
                                                                                
╭─ Options ────────────────────────────────────────────────────────────────────╮
| --install-completion          Install completion for the current shell.      |
| --show-completion             Show completion for the current shell, to copy |
|                               it or customize the installation.              |
| --help                        Show this message and exit.                    |
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
| count                Analyzes an ONNX model, counting nodes or parameters by |
|                      operation type or name prefix.                          |
| verify               Verify ONNX model output against PyTorch model output.  |
| export               Export the Kokoro model to ONNX format.                 |
| trial-quantization   Run quantization trials on individual nodes to measure  |
|                      their impact on model quality. Results are saved to a   |
|                      CSV file with columns: name, op_type, mel_distance,     |
|                      params, size                                            |
| estimate-size        Estimate model size after quantization/casting based on |
|                      trial results and thresholds.                           |
| export-optimized     Export an optimized model using both FP16 and INT8      |
|                      quantization based on trial results.                    |
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


## Quantization

Comparable quantization commands to onnx-community

##### `model_q8f16.onnx`

```bash
uv run kokoro-onnx export-optimized --quant-threshold=2 --quant-exclude '(^/decoder/generator/conv_post/Conv)' --quant-type QInt8
```

```
Optimizing model:
FP16 nodes: 7
Q nodes: 529

Converting nodes to FP16...

Quantizing nodes...

Model differences:

Added operations:
  Add: 151 nodes
  Cast: 250 nodes
  ConvInteger: 85 nodes
  DynamicQuantizeLSTM: 6 nodes
  DynamicQuantizeLinear: 136 nodes
  MatMulInteger: 147 nodes
  Mul: 464 nodes
  Reshape: 1 nodes

Removed operations:
  Conv: 85 nodes
  Gemm: 73 nodes
  LSTM: 6 nodes
  MatMul: 74 nodes

Total nodes:
  Original: 2371
  Modified: 3373
  Difference: +1002

Final model size: 82.39 MB
Size reduction: 73.5%
```

```bash
uv run kokoro-onnx count --size --count-by op+dtype --onnx-path kokoro_optimized.onnx --max-rows 15
```

| Group                        | Size (KB) | Percentage |   Parameters |
|------------------------------|-----------|------------|--------------|
| ConvInteger (UINT8)          | 53,385.33 |      64.2% | 54,666,581.0 |
| MatMulInteger (INT8)         | 12,066.58 |      14.5% | 12,356,177.0 |
| DynamicQuantizeLSTM (INT8)   | 10,496.02 |      12.6% |   10,747,928 |
| ConvTranspose (FP16)         |  5,905.27 |       7.1% |    3,023,496 |
| Gather (FP32)                |    702.00 |       0.8% |      179,712 |
| Add (FP32)                   |    222.28 |       0.3% |     56,904.0 |
| DynamicQuantizeLSTM (FP32)   |     96.09 |       0.1% |       24,600 |
| Reshape (FP32)               |     95.76 |       0.1% |       24,514 |
| Mul (FP32)                   |     72.71 |       0.1% |     18,614.0 |
| MatMul (FP16)                |     50.00 |       0.1% |       25,600 |
| Conv (FP16)                  |     38.54 |       0.0% |       19,734 |
| LayerNormalization (FP32)    |     37.00 |       0.0% |        9,472 |
| InstanceNormalization (FP32) |     27.53 |       0.0% |        7,048 |
| Slice (INT64)                |      9.21 |       0.0% |        1,179 |
| Reshape (INT64)              |      2.03 |       0.0% |          260 |
| ... and 81 more rows         |      4.68 |       0.0% |      1,065.0 |
| Total                        | 83,211.05 |       100% | 81,162,884.0 |

##### `model_quantized.onnx`

```bash
uv run kokoro-onnx export-optimized --quant-threshold=10 --fp16-threshold=-1 --quant-exclude '(^/decoder/generator/conv_post/Conv)' --quant-type QInt8
```

```
Optimizing model:
FP16 nodes: 0
Q nodes: 530

Quantizing nodes...
pre-Quantizing conv layers only to uint8, as they are not compatible with int8

Model differences:

Added operations:
  Add: 151 nodes
  Cast: 233 nodes
  ConvInteger: 85 nodes
  DynamicQuantizeLSTM: 6 nodes
  DynamicQuantizeLinear: 137 nodes
  MatMulInteger: 148 nodes
  Mul: 466 nodes
  Reshape: 1 nodes

Removed operations:
  Conv: 85 nodes
  Gemm: 73 nodes
  LSTM: 6 nodes
  MatMul: 75 nodes

Total nodes:
  Original: 2371
  Modified: 3359
  Difference: +988

Final model size: 88.21 MB
Size reduction: 71.6%
```

```bash
uv run kokoro-onnx count --size --count-by op+dtype --onnx-path kokoro_optimized.onnx --max-rows 15
```

| Group                        | Size (KB) | Percentage |   Parameters |
|------------------------------|-----------|------------|--------------|
| ConvInteger (UINT8)          | 53,385.33 |      59.9% | 54,666,581.0 |
| MatMulInteger (INT8)         | 12,091.58 |      13.6% | 12,381,778.0 |
| ConvTranspose (FP32)         | 11,812.25 |      13.3% |    3,023,936 |
| DynamicQuantizeLSTM (INT8)   | 10,496.02 |      11.8% |   10,747,928 |
| Gather (FP32)                |    702.00 |       0.8% |      179,712 |
| Add (FP32)                   |    222.28 |       0.2% |     56,904.0 |
| DynamicQuantizeLSTM (FP32)   |     96.09 |       0.1% |       24,600 |
| Reshape (FP32)               |     95.76 |       0.1% |       24,514 |
| Conv (FP32)                  |     78.84 |       0.1% |       20,182 |
| Mul (FP32)                   |     72.71 |       0.1% |     18,615.0 |
| LayerNormalization (FP32)    |     37.00 |       0.0% |        9,472 |
| InstanceNormalization (FP32) |     27.53 |       0.0% |        7,048 |
| Slice (INT64)                |      9.21 |       0.0% |        1,179 |
| Reshape (INT64)              |      2.03 |       0.0% |          260 |
| Gather (INT64)               |      0.43 |       0.0% |         55.0 |
| ... and 78 more rows         |      0.79 |       0.0% |        122.0 |
| Total                        | 89,129.86 |       100% | 81,162,886.0 |



##### `model_uint8f16.onnx`

```bash
uv run kokoro-onnx export-optimized --fp16-threshold=1  --quant-threshold=1 --quant-type=QUInt8 --quant-activation-type=QUInt8 --quant-static --samples 2 --quant-exclude '(^/decoder/generator/conv_post/Conv|/decoder/generator/resblocks)'
```
```
Model differences:

Added operations:
  Cast: 370 nodes
  DequantizeLinear: 172 nodes
  QGemm: 37 nodes
  QLinearAdd: 74 nodes
  QLinearConv: 49 nodes
  QLinearMatMul: 38 nodes
  QLinearMul: 12 nodes
  QuantizeLinear: 111 nodes

Removed operations:
  Add: 74 nodes
  Conv: 49 nodes
  Gemm: 37 nodes
  MatMul: 38 nodes
  Mul: 12 nodes

Total nodes:
  Original: 2371
  Modified: 3024
  Difference: +653

Final model size: 107.64 MB
Size reduction: 65.3%
```

| Group                        |  Size (KB) | Percentage |   Parameters |
|------------------------------|------------|------------|--------------|
| QLinearConv (UINT8)          |  43,305.34 |      39.7% | 44,344,673.0 |
| LSTM (FP16)                  |  21,040.00 |      19.3% |   10,772,480 |
| Conv (FP16)                  |  20,212.04 |      18.5% |   10,348,566 |
| MatMul (FP16)                |   7,346.00 |       6.7% |    3,761,152 |
| ConvTranspose (FP16)         |   5,905.27 |       5.4% |    3,023,496 |
| QGemm (UINT8)                |   4,482.54 |       4.1% |  4,590,119.0 |
| Gemm (FP16)                  |   3,483.00 |       3.2% |    1,783,296 |
| QLinearMatMul (UINT8)        |   2,208.02 |       2.0% |  2,261,011.0 |
| Gather (FP32)                |     702.00 |       0.6% |      179,712 |
| QGemm (INT32)                |     140.08 |       0.1% |       35,860 |
| QLinearConv (INT32)          |      68.76 |       0.1% |       17,602 |
| LayerNormalization (FP32)    |      37.00 |       0.0% |        9,472 |
| InstanceNormalization (FP32) |      24.53 |       0.0% |        6,280 |
| Mul (FP32)                   |      24.06 |       0.0% |      6,160.0 |
| Mul (FP16)                   |      18.00 |       0.0% |        9,216 |
| ... and 99 more rows         |      27.76 |       0.0% |     14,039.0 |
| Total                        | 109,024.40 |       100% | 81,163,134.0 |

##### `model_uint8.onnx`

```bash
uv run kokoro-onnx export-optimized --fp16-threshold=-1  --quant-threshold=1 --quant-type=QUInt8 --quant-activation-type=QUInt8 --quant-static --samples 2 --quant-exclude '(^/decoder/generator/conv_post/Conv|/decoder/generator/resblocks)'
```

```
Model differences:

Added operations:
  DequantizeLinear: 172 nodes
  QGemm: 37 nodes
  QLinearAdd: 74 nodes
  QLinearConv: 49 nodes
  QLinearMatMul: 38 nodes
  QLinearMul: 12 nodes
  QuantizeLinear: 111 nodes

Removed operations:
  Add: 74 nodes
  Conv: 49 nodes
  Gemm: 37 nodes
  MatMul: 38 nodes
  Mul: 12 nodes

Total nodes:
  Original: 2371
  Modified: 2654
  Difference: +283

Final model size: 164.15 MB
Size reduction: 47.1%
```

```bash
uv run kokoro-onnx count --size --count-by op+dtype --onnx-path kokoro_optimized.onnx --max-rows 15
```

| Group                        |  Size (KB) | Percentage |   Parameters |
|------------------------------|------------|------------|--------------|
| QLinearConv (UINT8)          |  43,305.34 |      25.9% | 44,344,673.0 |
| LSTM (FP32)                  |  42,080.00 |      25.2% |   10,772,480 |
| Conv (FP32)                  |  40,425.84 |      24.2% |   10,349,014 |
| MatMul (FP32)                |  14,692.04 |       8.8% |    3,761,161 |
| ConvTranspose (FP32)         |  11,812.25 |       7.1% |    3,023,936 |
| Gemm (FP32)                  |   6,966.00 |       4.2% |    1,783,296 |
| QGemm (UINT8)                |   4,482.54 |       2.7% |  4,590,119.0 |
| QLinearMatMul (UINT8)        |   2,208.02 |       1.3% |  2,261,011.0 |
| Gather (FP32)                |     702.00 |       0.4% |      179,712 |
| QGemm (INT32)                |     140.08 |       0.1% |       35,860 |
| QLinearConv (INT32)          |      68.76 |       0.0% |       17,602 |
| Mul (FP32)                   |      60.06 |       0.0% |     15,376.0 |
| LayerNormalization (FP32)    |      37.00 |       0.0% |        9,472 |
| InstanceNormalization (FP32) |      27.53 |       0.0% |        7,048 |
| Slice (INT64)                |       9.21 |       0.0% |        1,179 |
| ... and 94 more rows         |      13.54 |       0.0% |     11,195.0 |
| Total                        | 167,030.21 |       100% | 81,163,134.0 |

##### `model_fp16.onnx`

```bash
uv run kokoro-onnx export-optimized --fp16-threshold=5 --quant-threshold=-1
```

```
Optimizing model:
FP16 nodes: 463
Q nodes: 0

Converting nodes to FP16...

Model differences:

Added operations:
  Cast: 632 nodes

Total nodes:
  Original: 2371
  Modified: 3003
  Difference: +632

Final model size: 156.17 MB
Size reduction: 49.7%
```

```bash
uv run kokoro-onnx count --size --count-by op+dtype --onnx-path kokoro_optimized.onnx --max-rows 15
```

| Group                        |  Size (KB) | Percentage |   Parameters |
|------------------------------|------------|------------|--------------|
| Conv (FP16)                  | 106,856.92 |      67.3% |   54,710,744 |
| LSTM (FP16)                  |  21,040.00 |      13.2% |   10,772,480 |
| Gemm (FP16)                  |  12,518.04 |       7.9% |    6,409,236 |
| MatMul (FP16)                |  11,762.00 |       7.4% |    6,022,144 |
| ConvTranspose (FP16)         |   5,905.27 |       3.7% |    3,023,496 |
| Gather (FP32)                |     702.00 |       0.4% |      179,712 |
| Mul (FP32)                   |      24.06 |       0.0% |      6,159.0 |
| Mul (FP16)                   |      24.00 |       0.0% |       12,288 |
| LayerNormalization (FP16)    |      18.50 |       0.0% |        9,472 |
| Add (FP16)                   |      14.00 |       0.0% |        7,168 |
| InstanceNormalization (FP16) |      13.77 |       0.0% |        7,048 |
| Slice (INT64)                |       9.21 |       0.0% |        1,179 |
| Conv (FP32)                  |       1.75 |       0.0% |          448 |
| ConvTranspose (FP32)         |       1.72 |       0.0% |          440 |
| Gather (INT64)               |       0.44 |       0.0% |         56.0 |
| ... and 78 more rows         |       1.19 |       0.0% |        200.0 |
| Total                        | 158,892.86 |       100% | 81,162,270.0 |
