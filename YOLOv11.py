from ultralytics import YOLO
import cv2
import time


try:
    import tensorrt
    print(f"TensorRT: {tensorrt.__version__}")
    use_tensorrt = True
except ImportError:
    print("TensorRT not installed, using PyTorch")
    use_tensorrt = False

# Đường dẫn đến mô hình
model_path = "D:/Project_AI/Segmantation/best.pt" if not use_tensorrt else "D:/Project_AI/Segmantation/best.engine"

# Tải mô hình
model = YOLO(model_path, task="detect")  # Chỉ định task=detect

# Danh sách lớp
CLASSES = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C',
           '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H',
           '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']


# Xử lý video
def process_video(video_path=None, conf_thres=0.05, imgsz=416, half=False):  # Sử dụng FP32
    if video_path:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 1280 or height > 720:
            width, height = 1280, 720
    else:
        cap = cv2.VideoCapture(0)  # Webcam
        width, height = 640, 480

    out = cv2.VideoWriter("D:/Project_AI/Segmantation/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize khung
        if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
            frame = cv2.resize(frame, (width, height))

        # Chạy suy luận
        results = model.predict(frame, conf=conf_thres, iou=0.4, imgsz=imgsz, half=half, device=0)
        detections = results[0].boxes
        print(f"Number of detections: {len(detections)}")  # Debug

        # Vẽ bounding box
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{CLASSES[class_id]}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tính và hiển thị FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Result", frame)  # Comment để tăng FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Chạy trên video
video_path = r"D:/Project_AI/Segmantation/Cards.mp4"
process_video(video_path, conf_thres=0.05, imgsz=416, half=False)  # Sử dụng FP32
