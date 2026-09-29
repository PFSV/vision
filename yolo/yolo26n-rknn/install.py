from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.export(
    format="rknn",
    name="rk3588",
    quantize=8,
    data="coco8.yaml",
)
