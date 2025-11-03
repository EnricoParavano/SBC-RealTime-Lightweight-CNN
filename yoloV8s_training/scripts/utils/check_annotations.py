from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(data="coco.yaml", epochs=1, imgsz=640, batch=1, device=0, cache=False)

if __name__ == '__main__':
    main()