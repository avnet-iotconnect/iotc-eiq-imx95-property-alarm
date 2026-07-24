# elephant - face recognition + anti-theft alarm on FRDM-IMX95

The second pilot. Takes dolphin's live loop (camera -> YOLO11n on Neutron -> tracking -> overlay) and
adds **who is this person** and a **simple alarm**: each `person` YOLO track is run through YuNet + SFace
on the CPU to recognize registered users, and the alarm trips when someone unrecognized is on screen.

It also road-tests a composition-root design: `main.py` builds every module and hands them to one
`AntiTheftApp` (`app.py`); the app calls out to them and nothing calls back in.

## The pieces (and which future module each seeds)
| File          | Role                                                                  | Lineage  |
|---------------|-----------------------------------------------------------------------|----------|
| `main.py`     | composition root + frame loop + `command.txt` polling                 | main     |
| `app.py`      | `AntiTheftApp`: alarm state machine, `on_frame` / `on_command`        | app      |
| `detector.py` | `Detector`: YOLO interpreter wrapper, frame -> detections             | ml       |
| `yolo.py`     | YOLO pre/post math (letterbox, decode, NMS) - unchanged from dolphin  | ml       |
| `face.py`     | `FaceRecognizer`: YuNet detect + SFace embed on person crops          | ml/face  |
| `tracking.py` | `Tracker` + `Track` (now carries `identity` + `embedding`)            | db       |
| `registry.py` | persistent face DB (name <-> embedding), presence queries             | db       |
| `camera.py`   | GStreamer capture - unchanged from dolphin                            | video    |
| `overlay.py`  | boxes, name labels, alarm-state HUD + big "ALARM" banner              | video    |

## Face recognition (the model choice)
Two OpenCV models, **both bundled inside the board's `cv2`** - no pip installs, which is the whole reason
they were picked over ArcFace/FaceNet for this pilot:
- **YuNet** (`cv2.FaceDetectorYN`) finds a face + 5 landmarks inside each person crop.
- **SFace** (`cv2.FaceRecognizerSF`) aligns on those landmarks and embeds the face to 128 floats.

Match = cosine similarity vs each registered user, threshold ~0.363 (OpenCV's SFace default). Both run on
the CPU, off the *serial* Neutron budget YOLO owns - there's ample CPU headroom at 30 fps.
Moving embedding to an int8 ArcFace on the NPU is the documented later lever, not needed here.

## Registration + control (command.txt)
The app polls `command.txt` each frame; write one line, it runs once, the file is cleared:
```
echo 'register user Joe' > command.txt   # bind Joe to the largest visible face right now
echo 'disarm'            > command.txt   # arm | disarm the alarm
echo 'arm'               > command.txt
```
Registrations persist to `faces.json`, so known users survive a restart.

## Alarm (this pilot's slice)
Starts **armed**. Each frame: armed + a person on screen + nobody recognized => **ALARM** (red banner,
centered). Recognize a registered user and it drops back to armed (yellow). `disarm` = green, no alarm.
Guarded-object / theft-timing states are a later pilot - this one proves the face path.

## Run (on the board, from ~/elephant)
```
python3 diag-face.py face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx  # verify cv2
./run.sh --model yolo11n_neutron.tflite --delegate   # YOLO on NPU, faces on CPU
./run.sh --model yolo11n_int8.tflite                 # full-CPU baseline
```

## Host setup (once)
```
bash scripts/face-models-fetch.sh   # download YuNet + SFace ONNX into work/models
bash elephant/deploy.sh             # scp code + models (yolo tflite + 2 onnx) to the board
```
