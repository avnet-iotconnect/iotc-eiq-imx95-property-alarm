# About

This demo is Smart Camera with property alarm capabilities running on NXP FRDM-IMX95 with optional Ara240 hardware.

FRDM-IMX95 features 6 Arm CPU cores and a Neutron NPU which accelerate the AI/ML workloads.
Optional Ara240 can power the large generative models.

The demo is based in part on the NXP's 
[NXP® eIQ® GenAI Flow Demonstrator Package](https://github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator/tree/release/v3.0) 
version 3.0.


# Hardware

- [NXP FRDM-IMX95](https://www.avnet.com/americas/product/nxp/frdm-imx95/evolve-122131125/)
- (optional) [NXP Ara240](https://www.avnet.com/americas/product/gateworks/gw16168/evolve-216891109?searchTerm=Ara240)
- A USB Camera, [Logitech C920x HD Pro PC Webcam](https://www.amazon.com/Logitech-C920x-Pro-HD-Webcam/dp/B085TFF7M1) recommended.

### Other Optional Devices

- An HDMI monitor can be connected to view the camera feed.
- 3.5mm headphone jack can be connected to a phone microphone + headphones.
- For best STT performance, we recommend a USB directional microphone like Fifine AM8. PC speakers can then be connected to the 3.5mm headphone jack.
- The board has a bluetooth receiver. It can be used to connect a bluetooth headphones and/or speaker for TTS feedback.

# Features

A USB camera attached to the FRDM-IMX95 board is used to detect and recognize people, faces and objects. Alerts and intrusions are monitored and reported to /IOTCONNECT.


### AI/ML Features

- YOLO11n running on Neutron NPU detects and tracks people and objects in the camera feed.
- YuNet (on CPU) and SFace (on Neutron) detect and recognize faces in the camera feed.
- Voice processing is triggered by NXP  
[VIT wake word](https://www.nxp.com/design/design-center/software/embedded-software/voice-intelligent-technology-wake-word-and-voice-command-engines) 
technology - `Hey NXP`. 
- The board will process Speech-To-Text commands (moonshine-base on Neutron).
- The board will provide Text-To-Speech feedback to the user (VITS).
- A locally running VLM (SmolVLM 256M) can be used to describe the scene and answer questions about the scene.
- Ara240 can be used to power a local agent using the *Qwen 2.5 7B Instruct* model for natural language interaction:
  - Tell the agent what you need done, even multiple commands in one sentence like: "send me the alert log and arm the alarm"
  - Also, in mostly any language: "löschen sie das ereignisprotokoll" 

The NXP models are licensed for use on NXP hardware only and will run for one hour.

### /IOTCONNECT Features

- WebRTC remote camera monitoring.
- Cloud commands can be sent to the local agent, register new users, lock/unlock objects, describe the camera's scene and manage alert event logs. 
- Telemetry reporting what the camera sees regular intervals to the cloud.
- File upload of periodic screenshots when an alert is triggered or a new user is registered.
- File download of an online database user face images to the local device for automated user registration.

### Enhanced Features

- TTS [phoneme mapping](src/config/vocabulary.json) for pronouncing names and words correctly.
- STT mapping of similarly sounding words [to known commands](src/applib/commands.py). Like STT often interprets "unregister" as "I'm register".
- Face detection threshold - The camera waits until it has a "clear shot" of the face to register the user.
- YOLO track jitter smoothing - The camera will [smooth the bounding](src/applib/tracking.py) box of a tracked object to reduce jitter.

# Application Behavior

- The camera continuously processes the scene and detects faces and objects.
- The video streaming will be available on /IOTCONNECT in the **Video Streaming** device tab. 
- Registered users can be added by adding their face to the local database.
  - People are tracked with bounding box with their name as the label and recognized using the camera feed.
  - A face needs to be recognized only once, and it will be associated with a person being tracked 
even if their face is no longer visible.
  - Once a new user is registered, the device will also automatically upload a snapshot of the user to the S3 bucket.
- An alert can be generated when an unrecognized person enters the camera view or an object is moved or removed from the scene.
  - During the alert, "REC" is shown on the screen and the camera regularly snapshots and uploaded to the cloud for later review.
  - An event log is tracked locally and stays on the screen until the user clears the log.
  - The user can generate a full event log report to the cloud.
- The alarm can be armed or disarmed to indicate whether unauthorized and unattended people should trigger an alert.
  - If the alarm is armed, an unrecognized person entering camera view will trigger an alert, unless a registered user is also in the scene.
  - A few seconds of grace to let the user's face be recognized before triggering an alert.
- Objects can be locked on the screen and protected from theft whether the alarm is armed or disarmed.
  - If an object is locked it will be tracked on the screen, and if it is moved or removed from the scene, an alert will be generated.
- Snapshots taken at as result of specific actions and can be seen in the **Telemetry Files** device tab in /IOTCONNECT.
  - Triggered by requesting a snapshot C2D/Voice/Agent command, taken during alerts
  - Automatically taken when a new user is registered.
  - While the system is in alert state, snapshots are taken every several seconds and uploaded to the cloud.
- One can upload user face images to the S3 bucket as device-uploads/*client_id*/faces/*Person Name*.jpg.
  - The device will periodically check the S3 bucket for new images and automatically register them to the local database.
  - Each of files in this directory will be applied only once. 
The downloaded file's S3 ETag is kept to avoid duplicate registration of the same image, for example if a user wishes to unregister or re-register using the camera.
- OSD shows alarm state, Camera FPS(Inference FPS) Inference ms, *REC*ording (alert) status, last alert, reports model loading status, success/status for actions etc.
- The demo restarts automatically to maintain the 60-minute eIQ license as well as AWS temporary credentials.

# Commands

The same commands are reachable three ways, and they all run the same code — the same handlers, the
same refusals. Whatever answers you get is spoken back, flashed on the screen, and sent to the cloud.

- **Voice** — say `Hey NXP`, wait for the blip, then you have about 3 seconds to start talking.
The wording below is what the parser anchors on; the sentence around it does not matter
("*please lock my laptop now*" works).
- **Cloud** — the *Commands* tab of the device in /IOTCONNECT. These names come from the
[device template](files/device-template.json), and arguments are typed into the command's parameter field.
- **Agent** — the `agent` cloud command takes a plain-English sentence and hands it to the LLM on the
Ara240, which then calls the commands below as its tools. Needs the Ara240 and [src/connector/](src/connector) running.

| What it does                        | Say it                    | Cloud command               | Ask the agent                           |
|-------------------------------------|---------------------------|-----------------------------|-----------------------------------------|
| Register the face on screen         | `register user Michael`   | `user-register` *Michael*   | *register another user Michael*         |
| Forget a registered user            | `unregister user Michael` | `user-unregister` *Michael* | *forget Michael*                        |
| Undo the last registration          | `unregister last user`    | —                           | —                                       |
| Arm the alarm                       | `arm the alarm`           | `alarm-arm`                 | *arm the alarm*                         |
| Disarm, and clear everything        | `disarm the alarm`        | `alarm-disarm`              | *please disarm*                         |
| Guard an object on screen           | `lock the laptop`         | `object-lock` *laptop*      | *guard the laptop*                      |
| Stop guarding it                    | `unlock the laptop`       | `object-unlock` *laptop*    | *stop guarding the laptop*              |
| Read out the event log              | `what happened`           | `alert-describe`            | *what have you caught?*                 |
| Clear the event log                 | `clear the alert`         | `alert-clear`               | *clear the alert*                       |
| Describe the scene (VLM)            | `describe the scene`      | `scene`                     | —                                       |
| Snapshot to the cloud               | `take a snapshot`         | `snapshot`                  | *take a screenshot*                     |
| Report the state and who is visible | —                         | —                           | *is the alarm on, and who can you see?* |
| Report the local date and time      | —                         | —                           | *what time is it?*                      |
| Ask a question in plain English     | —                         | `agent` *your sentence*     | —                                       |
| Restart the demo                    | —                         | `restart`                   | —                                       |


# Board Setup

Download L6.18.2-1.0.0_MX95 from the
[NXP Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX) 
web page. The easiest way is to scroll all the way down to the *Downloads* section
and Enter `L6.18.2-1.0.0_MX95` into the search box.

Follow the
[Getting Started with FRDM-IMX95](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95:GS-FRDM-IMX95?section=get-software)
guide to flash the software.

### ARA2 SDK (install even if you don't have Ara240)

**Do this first**. This SDK will also resize the installed image 
so that the rest of the project and large models can fit. 
This deb will also install the `uv` tool, which will be handy later,
as well as optimum_ara wheel, which we will use later for large models on the Ara240.

More info at [https://github.com/nxp-imx/rt-sdk-ara2](https://github.com/nxp-imx/rt-sdk-ara2)

Download the **Ara2 Runtime SDK 2.0.4** (not newer):
https://www.nxp.com/design/design-center/software/embedded-software/ara-software-development-kit:ARA-SDK

On Windows, you can use [GitBash](https://git-scm.com/install/windows) or WSL to run the following commands.
Copy the sdk to the board using SCP and install the package using ssh:
```bash
IMX95=root@192.168.38.203 # your board's IP
scp ~/Downloads/rt-sdk-ara2_2.0.4.deb $IMX95:
ssh $IMX95 dpkg -i ./rt-sdk-ara2_2.0.4.deb
```

At this point, the board must be restarted by issuing the `reboot` command on the ssh terminal.
```bash
ssh $IMX95 dpkg -i ./rt-sdk-ara2_2.0.4.deb
```

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

### Download eIQ Neutron SDK (with model converter)

On your PC download the eIQ Neutron SDK **version 3.1.3** exactly from the
[eIQ® Toolkit for End-to-End Model Development and Deployment](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT#downloads)
web page.

Version 3.1.3 does not match what is on the `L6.18.2-1.0.0_MX95` image and will need to be deployed on the board before running the installer.

For local development, you can clone this repo first and unzip the SDK contents into work-refs/ (see [scripts/package-models.sh](scripts/package-models.sh)).

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

### /IOTCONNECT Device Setup

* Using the left navigation menu, select **Devices** and click the **Device** in this menu.
* Click **Add Device** button on top-right of the screen.
* Choose a Device Unique ID (Example: `imx95-01`)
* Pick an Entity to place the device into.
* Select the Device Template created in the previous step.
* Choose either the autogenerated certificate or upload your own.
  * If using the autogenerated certificate, click **Save and View**:
    * Download `iotcDeviceConfig.json` (paper and cog icon in top-right) and copy it to the board (below)
    * Download the credentials zip file via **Connection Info**, then Click the certificate icon
    * Unzip the credentials zip so that you can copy the files to the board (below)
  * If uploading your own certificate:
    * Create a certificate and private key on the board (see below)
    * Copy and paste the device cert (including the BEGIN and END lines) into the certificate field in the device creation form.
    * Click **Save and View**
    * Download `iotcDeviceConfig.json` (paper and cog icon in top-right) and copy it to the board (below)

## Deploying the Demo

```bash
IMX95=root@192.168.38.203 # your board's IP
ssh $IMX95 mkdir -p pa
# Download this file from the /IOTCONNECT portal and copy it to the board.
scp iotcDeviceConfig.json $IMX95:pa/
# the app will look for these two fixed names for cert and private key
# either copy them from the /IOTCONNECT portal or create them on the board (see below)
scp *-crt.pem $IMX95:pa/device-cert.pem
scp *-key.pem $IMX95:pa/device-pkey.pem
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


If cloning this repo, you can copy the application to the board using the following commands instead of downloading the source package.
```bash
IMX95=root@192.168.38.203 # your board's IP
scp -r src/* $IMX95:pa/
```
```bash
ssh $IMX95
cd ~/pa
```

If evaluating, you can simply download the application source package and run the installer on the board:
```bash
IMX95=root@192.168.38.203 # your board's IP
ssh $IMX95
cd ~/pa
# download the source package:
wget -O iotc-property-alarm-src.tgz https://downloads.iotconnect.io/partners/nxp/packages/iotc-property-alarm-src-v1.1.0.tgz
tar zxf iotc-property-alarm-src.tgz
./install.sh      # run once
./run.sh          # run the demo
```

The LLM based agent behind the `agent` command is a separate install and you need to execute run.sh every time you start up the board. 
The demo can run without it if you don't have the Ara240 board:

```bash
cd ~/pa/connector
./install.sh # run once
./run.sh # The model will take some time to load. Follow the on-screen instructions monitor the progress.
```

# Tips, FAQ Troubleshooting and Known Issues

- When you have no sound device plugged in, "Hey NXP" trigger and command responses may appear to be delayed.
Use the `--no-tts` flag to bypass TTS processing and improve visual feedback time.

- The demo will default to the USB audio. If you don't want audio to be detected, run the demo with the 3.5mm jack sound device `./run.sh --mic micfil` with no microphone plugged in.

- If a demo abruptly power cycles the board, the most likely issue is power delivery.
Plug the board into a high power USB port, using the cable supplied with the board or 
to be 100% sure, plug in a phone USB charger cable into the power USB-C port. 

- The USB camera may appear as /dev/video4 or /dev/video52 etc. when plugged in.
The best solution is power the board (power button or re-plug USB power)
**while the USB camera is plugged in**. If a board is running, issue a `poweroff` to cleanly shutdown. 

- FPS is lower at times:
The FPS fluctuation on the screen is normal as we periodically read faces on the screen (every Nth frame). 
Additionally, for the first minute or so, loading the models will consume more system resources and slow FPS down.
Depending on the camera, in low lighting conditions camera exposure may affect FPS.
Camera may keep the shutter open for longer in order to absorb more light.

- 3.5mm jack audio may not recognize the first syllable correctly. The audio driver seems to have a problem with a "pop" sound as it starts the sound recording.
You could use a USB microphone instead.


# License

Avnet code: [LICENSE.md](LICENSE.md)
NXP code: [LICENSE_NXP.txt](LICENSE_NXP.txt)
other licenses: [licenses/](licenses/)

The two cover different halves of what ends up on the board. Everything in this repo is Avnet's and
is MIT. NXP's eIQ GenAI Flow payload — the wake word, STT, TTS and VLM models and their runtime — is
downloaded by `install.sh` and never redistributed here; it is covered by LICENSE_NXP.txt, and
[licenses/LICENSE-SUMMARY.md](licenses/LICENSE-SUMMARY.md) is NXP's own component breakdown of it,
copied unedited.

