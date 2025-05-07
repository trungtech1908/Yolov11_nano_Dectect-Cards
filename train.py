from ultralytics import YOLO
import os

if __name__ == '__main__':
    checkpoint_path = "runs/detect/playing_cards_yolo11/weights/last.pt"
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}")
    else:
        print("No checkpoint found, starting from pretrained model")
        checkpoint_path = "yolo11n.pt"  # Nếu không có checkpoint, dùng mô hình pretrained

    # Tải mô hình
    model = YOLO(checkpoint_path)

    # Tiếp tục huấn luyện
    model.train(
        data="data.yaml",
        epochs=200,
        imgsz=640,
        batch=8,
        name="playing_cards_yolo11",
        device=0,
        workers=4,
        exist_ok=True
    )