import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from tqdm import tqdm

# SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEFAULT PATHS AND CONFIG
DEFAULT_MODEL_PATH = r"C:\\ModelliLiteTF\EfficientDet-D0.tflite"
DEFAULT_ANNOTATION_PATH = r"C:\\instances_val2017.json"
DEFAULT_IMAGE_DIR = r"C:\\coco2017\\images\\val2017"
DEFAULT_NUM_IMAGES = 5000
DEFAULT_NUM_SAVE = 100
DEFAULT_SCORE_THRESHOLD = 0.15
DEFAULT_INPUT_SIZE = 320
DEFAULT_CATEGORY_OFFSET = 1


class EfficientDetInference:
    def __init__(self, model_path: str, input_size: int) -> None:
        self.input_size = input_size
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        logger.info("=== INPUT DETAILS ===")
        for i, detail in enumerate(self.input_details):
            logger.info(
                "Input %s: %s, shape: %s, dtype: %s",
                i,
                detail["name"],
                detail["shape"],
                detail["dtype"],
            )

        logger.info("=== OUTPUT DETAILS ===")
        for i, detail in enumerate(self.output_details):
            logger.info(
                "Output %s: %s, shape: %s, dtype: %s",
                i,
                detail["name"],
                detail["shape"],
                detail["dtype"],
            )

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Ridimensiona l'immagine mantenendo l'aspect ratio e applicando padding."""
        target_size = self.input_size
        h, w = image.shape[:2]

        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)

        return input_tensor, scale, x_offset, y_offset

    def postprocess_detections(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
        num_detections: np.ndarray,
        original_w: int,
        original_h: int,
        scale: float,
        x_offset: int,
        y_offset: int,
        score_threshold: float,
    ) -> List[Dict[str, float]]:
        """Applica soglia e riconduce le bounding box alle dimensioni originali."""
        valid_detections: List[Dict[str, float]] = []

        for i in range(int(num_detections)):
            if scores[i] < score_threshold:
                continue

            y1, x1, y2, x2 = boxes[i]

            x1_corrected = (x1 * self.input_size - x_offset) / scale
            y1_corrected = (y1 * self.input_size - y_offset) / scale
            x2_corrected = (x2 * self.input_size - x_offset) / scale
            y2_corrected = (y2 * self.input_size - y_offset) / scale

            x1_corrected = max(0, min(x1_corrected, original_w))
            y1_corrected = max(0, min(y1_corrected, original_h))
            x2_corrected = max(0, min(x2_corrected, original_w))
            y2_corrected = max(0, min(y2_corrected, original_h))

            x = int(x1_corrected)
            y = int(y1_corrected)
            width = int(x2_corrected - x1_corrected)
            height = int(y2_corrected - y1_corrected)

            if width <= 0 or height <= 0 or width > original_w or height > original_h:
                continue

            valid_detections.append(
                {
                    "bbox": [x, y, width, height],
                    "class_id": int(classes[i]),
                    "score": float(scores[i]),
                }
            )

        return valid_detections

    def predict(self, image: np.ndarray, score_threshold: float) -> List[Dict[str, float]]:
        original_h, original_w = image.shape[:2]

        input_tensor, scale, x_offset, y_offset = self.preprocess_image(image)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        num_detections = self.interpreter.get_tensor(self.output_details[3]["index"])[0]

        return self.postprocess_detections(
            boxes,
            classes,
            scores,
            num_detections,
            original_w,
            original_h,
            scale,
            x_offset,
            y_offset,
            score_threshold,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue l'inferenza con efficientDet-D0 INT8 su COCO e salva predizioni e statistiche."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Percorso al modello TFLite.")
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATION_PATH,
        help="Percorso al file JSON con le annotazioni COCO (instances_val2017).",
    )
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGE_DIR,
        help="Cartella contenente le immagini COCO di validazione.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Numero di immagini da processare (<=0 per tutte).",
    )
    parser.add_argument(
        "--save-images",
        type=int,
        default=DEFAULT_NUM_SAVE,
        help="Numero di immagini annotate da salvare per analisi qualitativa.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Soglia minima sul punteggio di confidenza delle detection.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help="Dimensione di input (lato) utilizzata dal modello efficientDet.",
    )
    parser.add_argument(
        "--category-offset",
        type=int,
        default=DEFAULT_CATEGORY_OFFSET,
        help="Offset da sommare all'ID di classe restituito dal modello per allinearlo a COCO.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Cartella base in cui salvare predictions/ e reports/ (default: struttura repo).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Percorso del JSON di debug (default: predictions/predictions_debug.json).",
    )
    parser.add_argument(
        "--output-coco-json",
        default=None,
        help="Percorso del JSON compatibile con COCO (default: predictions/predictions_coco_format.json).",
    )
    return parser.parse_args()


def main() -> dict:
    args = parse_args()

    base_dir = (
        Path(args.output_root).expanduser()
        if args.output_root
        else Path(__file__).resolve().parents[1]
    )
    predictions_dir = base_dir / "predictions"
    reports_dir = base_dir / "reports"
    qualitative_dir = reports_dir / "qualitative_samples"

    for directory in (predictions_dir, reports_dir, qualitative_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.output_json:
        output_json_path = Path(args.output_json).expanduser()
        if not output_json_path.is_absolute():
            output_json_path = base_dir / output_json_path
    else:
        output_json_path = predictions_dir / "predictions_debug.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_coco_json:
        coco_predictions_path = Path(args.output_coco_json).expanduser()
        if not coco_predictions_path.is_absolute():
            coco_predictions_path = base_dir / coco_predictions_path
    else:
        coco_predictions_path = predictions_dir / "predictions_coco_format.json"
    coco_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser()
    annotation_path = Path(args.annotations).expanduser()
    image_dir = Path(args.images).expanduser()
    score_threshold = args.score_threshold
    input_size = args.input_size
    category_offset = args.category_offset
    requested_images = args.num_images
    num_images_to_save = max(0, args.save_images)

    logger.info("Modello: %s", model_path)
    logger.info("Annotazioni: %s", annotation_path)
    logger.info("Directory immagini: %s", image_dir)

    model = EfficientDetInference(str(model_path), input_size)

    logger.info("Caricamento annotazioni COCO...")
    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    total_available = len(image_ids)

    if requested_images is None or requested_images <= 0:
        selected_ids = image_ids
    else:
        if requested_images > total_available:
            logger.warning(
                "Richieste %s immagini ma il dataset ne contiene %s. Verranno elaborate tutte le immagini disponibili.",
                requested_images,
                total_available,
            )
        selected_ids = image_ids[:requested_images]

    images = coco.loadImgs(selected_ids)
    target_images = len(images)

    logger.info(
        "Inizio elaborazione di %s immagini (limite richiesto: %s)",
        target_images,
        requested_images,
    )
    logger.info(
        "Salvataggio visualizzazioni per le prime %s immagini (se disponibili).",
        num_images_to_save,
    )

    cat_id_to_name = {
        cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())
    }
    logger.info("Categorie COCO caricate: %s", len(cat_id_to_name))
    logger.info("Prime 5 categorie: %s", list(cat_id_to_name.items())[:5])

    results: List[Dict] = []
    failed_images = 0
    saved_images_count = 0

    for img_idx, img in enumerate(
        tqdm(images, desc="Elaborazione immagini", total=target_images), start=1
    ):
        img_path = image_dir / img["file_name"]

        try:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Immagine non trovata: %s", img_path)
                failed_images += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = model.predict(image_rgb, score_threshold)

            save_visualization = saved_images_count < num_images_to_save

            for detection in detections:
                x, y, width, height = detection["bbox"]
                class_id = detection["class_id"]
                score = detection["score"]

                category_id = class_id + category_offset
                label = cat_id_to_name.get(category_id, f"ID {category_id}")

                if save_visualization:
                    cv2.rectangle(
                        image_rgb,
                        (x, y),
                        (x + width, y + height),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        image_rgb,
                        f"{label} {score:.2f}",
                        (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                results.append(
                    {
                        "image_id": img["id"],
                        "category_id": category_id,
                        "bbox": [x, y, width, height],
                        "score": score,
                        "label": label,
                        "file_name": img["file_name"],
                    }
                )

            if save_visualization:
                img_out_path = qualitative_dir / img["file_name"]
                img_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(img_out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                )
                saved_images_count += 1

            if img_idx % 500 == 0:
                logger.info(
                    "Elaborate %s/%s immagini. Detection totali: %s",
                    img_idx,
                    target_images,
                    len(results),
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Errore elaborando %s: %s", img["file_name"], exc, exc_info=True
            )
            failed_images += 1
            continue

    logger.info("Salvataggio risultati...")

    total_detections = len(results)
    unique_images = len({r["image_id"] for r in results})
    processed_images = target_images - failed_images
    avg_detections_per_image = total_detections / max(unique_images, 1)

    stats = {
        "total_images_requested": requested_images,
        "total_images_loaded": target_images,
        "total_images_processed": processed_images,
        "images_with_visualizations_saved": saved_images_count,
        "failed_images": failed_images,
        "total_detections": total_detections,
        "unique_images_with_detections": unique_images,
        "avg_detections_per_image": avg_detections_per_image,
        "score_threshold": score_threshold,
        "model_path": str(model_path),
        "annotation_path": str(annotation_path),
        "image_root": str(image_dir),
        "processing_config": {
            "input_size": input_size,
            "category_offset": category_offset,
            "save_visualizations_limit": num_images_to_save,
        },
    }

    output_data = {"statistics": stats, "detections": results}

    with output_json_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    coco_format_results = [
        {
            "image_id": r["image_id"],
            "category_id": r["category_id"],
            "bbox": r["bbox"],
            "score": r["score"],
        }
        for r in results
    ]

    with coco_predictions_path.open("w") as f:
        json.dump(coco_format_results, f)

    logger.info("Risultati dettagliati salvati in: %s", output_json_path)
    logger.info("Predizioni COCO salvate in: %s", coco_predictions_path)
    logger.info("Ora puoi eseguire il calcolo mAP su tutte le immagini processate")

    return output_data


if __name__ == "__main__":
    main()
