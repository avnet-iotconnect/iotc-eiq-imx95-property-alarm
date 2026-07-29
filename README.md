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

**Do this first**. This SDK will also resize the installed image so that the rest of the project and large models can fit.

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

On your PC download the eIQ Neutron SDK **version 3.0.0** from the

[eIQ® Toolkit for End-to-End Model Development and Deployment](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT#downloads)
web page. 

Version 3.0.0 matches what is on the `L6.18.2-1.0.0_MX95` image.

Expand it into a directory parallel to this repo `../imx-eiq-neutron-sdk`

### Which SDK version? Ask the board.

The converter bakes a microcode blob into the `.tflite`, and the Neutron firmware on the board refuses
a blob it did not expect. So the SDK version is not a choice — it is dictated by whichever image you
flashed. Get it from the board:

```bash
strings /usr/lib/firmware/NeutronFirmware.elf | grep 'Firmware ver'
# Neutron Firmware ver:3.0.0-0X6d41ac8b started
```

and from the SDK on your PC:

```bash
LD_LIBRARY_PATH=../imx-eiq-neutron-sdk/lib ../imx-eiq-neutron-sdk/bin/neutron-converter --version
# eIQ neutron-converter version 3.0.0+0X6d41ac8b
```

**Both the `3.0.0` and the hex hash must match.** They differ only in the separator (`-` vs `+`). If
they do not match the firmware prints `Microcode version mismatch!` at inference time. Note that a
newer board image does not imply a newer Neutron: L6.18.2 ships 3.0.0, while the older L6.12 shipped
3.1.2. To check a downloaded zip before expanding it, the same string is inside it:

```bash
unzip -p eiq-neutron-sdk-linux-3.0.0.zip target/imx95/imx95/NeutronFirmware.elf | strings | grep 'Firmware ver'
```

Do not "fix" a mismatch by copying the SDK's `target/imx95/` runtime onto the board — match the PC
side to the board and reconvert the models.

## Converting a YOLO model for Neutron (a note, not a guide)
```bash
bash scripts/yolo-convert-neutron.sh
```

```bash
bash run.sh --model yolo11n_neutron.tflite --delegate
```

# Troubleshooting

- If a demo abruptly power cycles the board, the most likely issue is power delivery.
Plug the board into a high power USB port, using the cable supplied with the board or 
to be 100% sure, plug in a phone USB charger cable into the power USB-C port. 

- The USB camera may appear as /dev/video4 or /dev/video52 etc. when plugged in.
The best solution is power the board (power button or re-plug USB power)
**while the USB camera is plugged in**. If a board is running, issue a `poweroff` to cleanly shutdown. 

