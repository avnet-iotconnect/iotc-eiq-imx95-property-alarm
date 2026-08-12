# nxp-ml

Board setup:
Download L6.18.2-1.0.0_MX95from
[NXP Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX) 
web page. The easiest way is to scroll all the way down to the *Downloads* section
and Enter `L6.18.2-1.0.0_MX95` into the search box.

Follow the
[Getting Started with FRDM-IMX95](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95:GS-FRDM-IMX95?section=get-software)
guide to flash the software

## ARA2 SDK

**Do this first**. This SDK will also resize the installed image 
so that the rest of the project and large models can fit. 
This deb will alsso install the `uv` tool, which will be handy later,
as well as optimum_ara weel, which we will use later for large models on the Ara240.

More info at [https://github.com/nxp-imx/rt-sdk-ara2](https://github.com/nxp-imx/rt-sdk-ara2)

Download the **Ara2 Runtime SDK 2.0.4** (not newer):
https://www.nxp.com/design/design-center/software/embedded-software/ara-software-development-kit:ARA-SDK

Copy the sdk to the board using SCP. For example:
```bash
scp ~/Downloads/rt-sdk-ara2_2.0.4.deb  root@192.168.38.203:
```

SSH to the board and Install it:
dpkg -i ./rt-sdk-ara2_2.0.4.deb

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

When flashing is done, you should see something similar to this in the output:
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


## eIQ Neutron SDK (model converter)

On your PC download the eIQ Neutron SDK **version 3.1.3** from the

[eIQ® Toolkit for End-to-End Model Development and Deployment](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT#downloads)
web page.

Download the zip into the root of this repo. 

Version 3.1.3 does not match what is on the `L6.18.2-1.0.0_MX95` image and will need to be deployed on the board before running the installer:


## Deploying a pilot (a note, not a deploy script)

The whole deployment is the `koala/` directory plus the converted models. Both are copied straight
to the board — nothing is staged inside the repo, so the working copy stays exactly what is
committed:

```bash
IMX95=root@192.168.38.203 # your board's IP
cd lion/ # or another pilot dir
scp ../work/iotcDeviceConfig.json $TGT:pa/
# the app will look for these two fixed names for cert and private key:
scp ../work/*-crt.pem $IMX95:pa/device-cert.pem
scp ../work/*-key.pem $IMX95:pa/device-pkey.pem
scp eiq-neutron-sdk-linux-3.1.3.zip $IMX95:pa/  # We need to install the matching neutron SDK on the board
```
SSH to the device (ssh $IMX95) and run the following commands:

```bash
mkdir -p pa && cd pa
# download the source package:
wget -O iotc-property-alarm-src.tgz https://downloads.iotconnect.io/partners/nxp/packages/iotc-property-alarm-src-v1.1.0.tgz
tar zxf iotc-property-alarm-src.tgz
./install.sh      # once: eIQ payload, espeak-ng, venv
./run.sh          # run the demo
```

The LLM behind the `ask` command is a separate install and a separate process, and the demo runs
without it:

```bash
cd ~/pa/connector && ./install.sh && ./run.sh
```

`install.sh` does not install the board itself — the BSP image, the rt-sdk-ara2 `.deb` and the
Ara-240 firmware above are prerequisites. In the finished project this section becomes "wget a
tarball and unpack it".

# Optional TODO Fetaures

## Name Tag Detection

(Not feasible due to low resolution)

PP-OCRv3 text detection + CRNN_EN text recognition, both from opencv_zoo, run through cv2.dnn — the board's OpenCV already ships them, so no pip, no onnxruntime, no VLM. Detection is cheap and recognition is not, so detect first and recognise only what you need, splitting each line at its word gaps before recognising. Measured on the board: detection ~247 ms at 640×480 (~49 ms at 256×192), recognition ~200 ms per line — so a naive full-badge pass costs ~1.85 s.

# Tips, FAQ and Troubleshooting

- The demo will default to the USB audio. If you don't want audio to be detected, run the demo with the 3.5mm jack sound device `./run.sh --mic micfilaudio` with no microphone plugged in.

- If a demo abruptly power cycles the board, the most likely issue is power delivery.
Plug the board into a high power USB port, using the cable supplied with the board or 
to be 100% sure, plug in a phone USB charger cable into the power USB-C port. 

- The USB camera may appear as /dev/video4 or /dev/video52 etc. when plugged in.
The best solution is power the board (power button or re-plug USB power)
**while the USB camera is plugged in**. If a board is running, issue a `poweroff` to cleanly shutdown. 

- FPS is lower at times:

Teh FPS fluctuation on the screen is normal as we periodically read faces on the screen (every Nth frame). Additionally, for the first minute or so, loading the models will consume more system resourrces and slow FPS down.

Depending on the camera, in low lighting conditions camera exposure may affect FPS. Camera may keep the shutter open for longer in order to absorb more light.

