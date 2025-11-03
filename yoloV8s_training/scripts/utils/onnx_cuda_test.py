import onnxruntime as ort

print("Esecuzione test ONNX Runtime con CUDA...")

# Lista dei provider disponibili
available_providers = ort.get_available_providers()
print(f"Provider disponibili: {available_providers}")

try:
    # Prova a creare una sessione su GPU
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        "runs/detect/benchmark_yolov8s_full/weights/best.onnx",
        sess_options,
        providers=["CUDAExecutionProvider"]
    )
    print("ONNX Runtime ha caricato correttamente il modello con CUDA!")
except Exception as e:
    print("Errore nel caricare ONNX Runtime con CUDA:")
    print(e)

print("\nTest completato.")
