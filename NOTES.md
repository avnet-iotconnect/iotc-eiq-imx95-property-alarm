/run.sh --model yolov8n_neutron.tflite --delegate

ultralytics
yolo export model=yolov8n.pt format=tflite int8=True imgsz=224
neutron-converter --input yolov8n_full_integer_quant.tflite \
  --output yolov8n_neutron.tflite --target imx95


One clarification on my earlier hand-wave: the pf09/pf53_soc/pf53_arm zones reading a flat 105 °C are the PMIC sensors (PF09/PF5300 power chips), not the CPU — their trips are 140/155 °C and the constant value is an uncalibrated/fixed reading. That's why I dismissed them before; the cooling-device states now confirm they aren't triggering anything.

So the way to check thermal throttling on this board, for next time:


cat /sys/class/thermal/cooling_device*/cur_state        # >0 = throttling active (the real tell)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq   # vs scaling_max_freq under load
# and compare a55-thermal temp to its trip_point_*_temp (passive=105C here)

