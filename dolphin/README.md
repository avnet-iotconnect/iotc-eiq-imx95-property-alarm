# dolphin - YOLO11 on FRDM-IMX95, CPU vs Neutron NPU

Live object detection on a Logitech C920, proving the Neutron NPU path end-to-end on real
camera frames. Same code both ways; only the model + `--delegate` flag change.

## Files
- `detect.py` - the flow: capture -> preprocess -> inference -> postprocess, timed per stage.
- `camera.py` - GStreamer pipeline (`tee` -> waylandsink preview + appsink frames).
- `yolo.py` - the math: letterbox, quantize, decode YOLO11 output, NMS, box mapping.
- `run.sh` - launcher that sets the Wayland env for the preview.
- `deploy.sh` - host-side scp of code + models to the board.

## How YOLO works here (the short version)
YOLO11 takes one square RGB image and predicts, at 2100 anchor points, a box and 80 class scores.
We `letterbox` the frame into the square input (resize keeping aspect ratio, gray pad), quantize to
int8, and run the model. Then for each anchor we take the best class, drop low scores, and remove
overlapping duplicates with non-max suppression (NMS). Boxes are mapped back onto the full frame.

**Why the model output looks "raw".** YOLO's box decode (DFL softmax + anchor math) does not
survive int8 quantization - the coordinates collapse to zero. So the model is exported to emit its
**raw head tensors** (shape `1 x 144 x 2100`: 64 box-distribution channels + 80 class logits), and
that decode is done in float on the CPU in `yolo.py` (`decode_detections`). This is deliberate and
standard: the NPU runs the heavy convolutions (as `neutronOp`s), the CPU does the cheap decode.
See `work/export-neutron.py` for the matching export patch.

## Run (on the board, from ~/dolphin)
```
./run.sh --model yolo11n_int8.tflite                # CPU
./run.sh --model yolo11n_neutron.tflite --delegate  # Neutron NPU
```
Compare the `inference=` column between the two - that stage is what moves to the NPU. On the
`--delegate` run, look for the delegate's `N nodes delegated out of M ... K partitions` line:
that is the proof the NPU (not a silent CPU fallback) ran the model.
