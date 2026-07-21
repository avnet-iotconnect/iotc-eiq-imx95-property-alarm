ultralytics
yolo export model=yolov8n.pt format=tflite int8=True imgsz=224
neutron-converter --input yolov8n_full_integer_quant.tflite \
  --output yolov8n_neutron.tflite --target imx95