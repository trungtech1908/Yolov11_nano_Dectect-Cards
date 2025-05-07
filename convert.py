from ultralytics import YOLO

# Đường dẫn file .pt
pt_path = "best.pt"
engine_path = "best.engine"


# Tải mô hình
model = YOLO(pt_path, task="detect")

# Xuất sang TensorRT Engine với FP32
try:
    model.export(
        format="engine",
        imgsz=416,
        half=False,
        device=0,
        workspace=4
    )
    print(f"Successfully exported {pt_path} to {engine_path} with FP32")
except Exception as e:
    print(f"Error during export: {str(e)}")

