# About

This demo is Smart Camera with property alarm capabilities running on NXP FRDM-IMX95 with optional Ara240 hardware.

FRDM-IMX95 features 6 Arm CPU cores and a Nueutron NPU which accelerate the AI/ML workloads.
Optional Ara240 can power the large generative models.

The demo is based in part on the NXPs's 
[NXP® eIQ® GenAI Flow Demonstrator Package](https://github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator/tree/release/v3.0) 
version 3.0.


# Features

A USB camera attached to the FRDM-IMX95 board is used to detect and recognize people, faces and objects.

# Hardware

- [NXP FRDM-IMX95](https://www.avnet.com/americas/product/nxp/frdm-imx95/evolve-122131125/)
- (optional) [NXP Ara240](https://www.avnet.com/americas/product/gateworks/gw16168/evolve-216891109?searchTerm=Ara240) (optional)
- A USB Camera, [Logitech C920x HD Pro PC Webcam](https://www.amazon.com/Logitech-C920x-Pro-HD-Webcam/dp/B085TFF7M1) recommended.

Other optional devices to enhance the experience:
- An HDMI monitor can be connected to view the camera feed.

### AI/ML Features

- Voice processing is triggered by NXP  
[VIT wake word](https://www.nxp.com/design/design-center/software/embedded-software/voice-intelligent-technology-wake-word-and-voice-command-engines) 
technology - `Hey NXP`. 
- The board will process Speech-To-Text commands (Whisper, Moonshine)  
- The board will provide Text-To-Speech feedback to the user (VITS).
- A locally running VLM (SmolVLM 500M) can be used to describe the scene and answer questions about the scene.
- Ara240 can be used to power a local agent using Qwen 2.5 7B model for natural language interaction.
- The camera can be monitored using an HDMI display or remotely via /IOTCONNECT using WebRTC.

The NXP models are licensed for use on NXP hardware only and will run for one hour.

### /IOTCONNECT Features

- WebRTC remote camera monitoring.
- Cloud commands can be sent to the local agent, register new users, lock/unlock objects, describe the camera's scene and manage alert event logs. 
- Telemetry reporting at regular intervals to the cloud.
- File upload of periodic screenshots when an alert is triggered or a new user is registered.
- File download of an online database user face images to the local device for recognition.


### Enhanced Features

- TTS [phoneme mapping](src/config/vocabulary.json) for pronouncing names and words correctly.
- STT mapping of similarly sounding words [to known commands](mamba/applib/commands.py).
- Face detection threshold - The camera waits until it has a "clear shot" of the face to register the user.
- YOLO track jitter smoothing - The camera will smooth the bounding box of a tracked object to reduce jitter.

# Functionality

- The camera continuously processes the scene and detects faces and objects.
- Registered users can be added by adding their face to the local database.
  - People are tracked with bounding box with their name as the label and recognized using the camera feed.
  - A face needs to be recognized only once, and it will be associated with a person being tracked 
even if their face is no longer visible.
- An alert can be generated when an unrecognized person enters the camera view or an object is moved or removed from the scene.
  - During the alert, "*REC" is shown on the screen and the camera regularly snapshots and uploaded to the cloud for later review.
  - An event log is tracked locally and stays on the screen until the user clears the log.
  - The user can generate a full event log report to the cloud.
- The alarm can be armed or disarmed.
  - If the alarm is armed, an unrecognized person entering camera view will trigger an alert.
  - A few seconds of grace to let the user's face be recognized before triggering an alert.
- Objects can be locked on the screen and protected from theft whether the alarm is armed or disarmed.
  - If an object is locked it will be tracked on the screen, and if it is moved or removed from the scene, an alert will be generated.

# Board Setup

Download L6.18.2-1.0.0_MX95from
[NXP Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX) 
web page. The easiest way is to scroll all the way down to the *Downloads* section
and Enter `L6.18.2-1.0.0_MX95` into the search box.

Follow the
[Getting Started with FRDM-IMX95](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95:GS-FRDM-IMX95?section=get-software)
guide to flash the software.

## ARA2 SDK

**Do this first**. This SDK will also resize the installed image 
so that the rest of the project and large models can fit. 
This deb will also install the `uv` tool, which will be handy later,
as well as optimum_ara wheel, which we will use later for large models on the Ara240.

More info at [https://github.com/nxp-imx/rt-sdk-ara2](https://github.com/nxp-imx/rt-sdk-ara2)

Download the **Ara2 Runtime SDK 2.0.4** (not newer):
https://www.nxp.com/design/design-center/software/embedded-software/ara-software-development-kit:ARA-SDK

Copy the sdk to the board using SCP. For example:
```bash
scp ~/Downloads/rt-sdk-ara2_2.0.4.deb  root@192.168.38.203:
```

SSH to the board and Install it:
```bash
dpkg -i ./rt-sdk-ara2_2.0.4.deb
```

soft reboot the board by issuing the `reboot` command on the terminal.

To check that everything is working, run the following command on the board:
```
chip_info.sh
```

Look for the firmware_version(raw) field in the output:
```
| firmware_version(raw)       | 131072     |
| firmware_version            | 2.0.0      |
```

If the version is not **131072**, update it:
```
program_flash.sh
```
Reboot the board again after firmware update to ensure the new version is properly loaded and initialized.
Do not reset the board during this process.

When flashing is done, you should see something similar to this in the output in console logs:
```
[I:20260729:20:29:41:539968] [] [kinara_main_1057][StatsDPublishService] registered this collector:  default_lb
[I:20260729:20:29:41:541363] [] [task_request_queue_processing_task_runner0_1061][ChipInfoRequest] BOOT SUCCESS
[I:20260729:20:29:41:541403] [] [task_request_queue_processing_task_runner0_1061][ChipInfoRequest] HIF IS ENABLED
[I:20260729:20:29:41:649807] [ChipInfo] [kinara_main_1057][ChipInfo] chip info for the devices connected
```

### Diagnostics

To view detailed service logs:
```
journalctl -u rt-sdk-ara2.service
```
To verify the proxy is running:
```
ps -eaf | grep proxy_ara240
```

## eIQ Neutron SDK (with model converter)

On your PC download the eIQ Neutron SDK **version 3.1.3** from the

[eIQ® Toolkit for End-to-End Model Development and Deployment](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT#downloads)
web page.

Download the zip into the root of this repo. 

Version 3.1.3 does not match what is on the `L6.18.2-1.0.0_MX95` image and will need to be deployed on the board before running the installer.

For local development, you can clone this repo first and unzip the SDK contents into work-refs/ (see [scripts/package-models.sh](scripts/package-models.sh).

# /IOTCONNECT Setup

### Create an /IOTCONNECT Account
An /IOTCONNECT account with an AWS backend is required.  If you need to create an account, a free trial subscription is available.
The free subscription may be obtained directly from [iotconnect.io](https://iotconnect.io) or through the AWS Marketplace.

* Option #1 **(Recommended)**   
/IOTCONNECT via [AWS Marketplace](https://github.com/avnet-iotconnect/avnet-iotconnect.github.io/blob/main/documentation/iotconnect/subscription/iotconnect_aws_marketplace.md) - 60 day trial; AWS account creation required

* Option #2  
/IOTCONNECT via [iotconnect.io](https://subscription.iotconnect.io/subscribe?cloud=aws) - 30 day trial; no credit card required

> [!NOTE]
> Be sure to check any SPAM folder for the temporary password after registering.

Login to the platform by navigating to [console.iotconnect.io](https://console.iotconnect.io)

### /IOTCONNECT Device Template Setup

An /IOTCONNECT *Device Template* will need to be created or imported.
* Download the premade [device-template.json](files/device-template.json) 
(Open the link then click the *Download Raw File* icon on the right).
* Import the template into your /IOTCONNECT instance:  [Importing a Device Template](https://github.com/avnet-iotconnect/avnet-iotconnect.github.io/blob/main/documentation/iotconnect/import_device_template.md) guide  
> **Note:**  
> For more information on [Template Management](https://docs.iotconnect.io/iotconnect/concepts/cloud-template/) 
> please see the [/IOTCONNECT Documentation](https://iotconnect.io) website.


## Deploying the Demo

```bash
IMX95=root@192.168.38.203 # your board's IP
ssh $IMX95 mkdir -p pa
cd lion/ # or another pilot dir
# Download this file from the /IOTCONNECT portal and copy it to the board.
scp ../work/iotcDeviceConfig.json $TGT:pa/
# the app will look for these two fixed names for cert and private key
# either copy them from the /IOTCONNECT portal or create them on the board (see below)
scp ../work/*-crt.pem $IMX95:pa/device-cert.pem
scp ../work/*-key.pem $IMX95:pa/device-pkey.pem
scp eiq-neutron-sdk-linux-3.1.3.zip $IMX95:pa/  # the installer below will use this zip to install the matching neutron SDK on the board
```
SSH to the device (ssh $IMX95):

If you did not copy the device-cert.pem and device-pkey.pem files, you can create them with the following commands:

```bash
cd ~/pa
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 3600 -keyout device-pkey.pem -out device-cert.pem -subj "/CN=YourDeviceName"
# copy and paste this device cert when creating the device in /IOTCONNECT portal
cat device-cert.pem
```


```bash
cd ~/pa
# download the source package:
wget -O iotc-property-alarm-src.tgz https://downloads.iotconnect.io/partners/nxp/packages/iotc-property-alarm-src-v1.1.0.tgz
tar zxf iotc-property-alarm-src.tgz
./install.sh      # once: eIQ payload, espeak-ng, venv
./run.sh          # run the demo
```

The LLM based agent behind the `agent` command is a separate install and a separate process. 
The demo can run without it if you don't have the Ara240 board:

```bash
cd ~/pa/connector
./install.sh # run once
./run.sh # The model will take some time to load. Follow the on-screen instructions monitor the progress.
```

# Tips, FAQ and Troubleshooting

- When you have no sound device plugged in, "Hey NXP" trigger and command responses may appear to be delayed.
Use the `--no-tts` flag to bypass TTS processing and improve visual feedback time.

- The demo will default to the USB audio. If you don't want audio to be detected, run the demo with the 3.5mm jack sound device `./run.sh --mic micfilaudio` with no microphone plugged in.

- If a demo abruptly power cycles the board, the most likely issue is power delivery.
Plug the board into a high power USB port, using the cable supplied with the board or 
to be 100% sure, plug in a phone USB charger cable into the power USB-C port. 

- The USB camera may appear as /dev/video4 or /dev/video52 etc. when plugged in.
The best solution is power the board (power button or re-plug USB power)
**while the USB camera is plugged in**. If a board is running, issue a `poweroff` to cleanly shutdown. 

- FPS is lower at times:
Teh FPS fluctuation on the screen is normal as we periodically read faces on the screen (every Nth frame). 
Additionally, for the first minute or so, loading the models will consume more system resources and slow FPS down.
Depending on the camera, in low lighting conditions camera exposure may affect FPS. 
- Camera may keep the shutter open for longer in order to absorb more light.

