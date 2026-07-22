# nxp-ml

Board setup:
Download L6.12.49-2.2.0_MX95 from
[NXP Embedded Linux for i.MX Applications Processors](https://www.nxp.com/design/design-center/software/embedded-software/i-mx-software/embedded-linux-for-i-mx-applications-processors:IMXLINUX) 
web page. The easiest way is to scroll all the way down to the *Downloads* section
and Enter `L6.12.49-2.2.0_MX95` into the search box.

Follow the
[Getting Started with FRDM-IMX95](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95:GS-FRDM-IMX95?section=get-software)
guide to flash the software

[!TODO]
> Write a guide to resize the partition and/or deal with disk space.


On your PC download the eIQ Neutron SDK 3.1.2 from the

[eIQ® Toolkit for End-to-End Model Development and Deployment](https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT#downloads)
web page.

Expand it into a directory parallel to this repo `../imx-eiq-neutron-sdk`

## Converting a YOLO model for Neutron (a note, not a guide)
```bash
bash scripts/yolo-convert-neutron.sh
```



```bash
bash run.sh --model yolo11n_neutron.tflite --delegate
```