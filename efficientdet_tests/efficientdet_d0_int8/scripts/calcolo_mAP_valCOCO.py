import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH_PATH = r"C:**\\instances_val2017.json"
DEFAULT_PREDICTIONS_FILENAME = "predictions_coco_format.json"
DEFAULT_REPORT_FILENAME = "detailed_evaluation_report.json"
DEFAULT_TOP_K = 20


class COCOEvaluator:
    def __init__(self, gt_path: Path, pred_path: Path) -> None:
        logger.info("Caricamento ground truth da: %s", gt_path)
        self.coco_gt = COCO(str(gt_path))

        logger.info("Caricamento predizioni da: %s", pred_path)
        with pred_path.open("r") as f:
            self.predictions = json.load(f)

        valid_img_ids = set(self.coco_gt.getImgIds())
        original_len = len(self.predictions)
        self.predictions = [p for p in self.predictions if p["image_id"] in valid_img_ids]
        filtered = original_len - len(self.predictions)
        if filtered:
            logger.warning(
                "Filtrate %s predizioni con image_id non presente nelle annotazioni.",
                filtered,
            )

        logger.info("Predizioni caricate: %s", len(self.predictions))
        logger.info("Immagini nel ground truth: %s", len(valid_img_ids))

        self.coco_dt = self.coco_gt.loadRes(self.predictions)

        self.cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())
        }

    def evaluate_all_metrics(self) -> Tuple[Dict[str, float], COCOeval]:
        logger.info("Inizio valutazione COCO...")

        coco_eval = COCOeval(self.coco_gt, self.coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "mAP_0.5:0.95": coco_eval.stats[0],
            "mAP_0.5": coco_eval.stats[1],
            "mAP_0.75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "mAR_1": coco_eval.stats[6],
            "mAR_10": coco_eval.stats[7],
            "mAR_100": coco_eval.stats[8],
            "mAR_small": coco_eval.stats[9],
            "mAR_medium": coco_eval.stats[10],
            "mAR_large": coco_eval.stats[11],
        }

        return metrics, coco_eval

    def evaluate_per_category(self) -> Dict[str, Dict[str, float]]:
        logger.info("Calcolo mAP per categoria:")

        category_results: Dict[str, Dict[str, float]] = {}

        for cat_id in self.coco_gt.getCatIds():
            cat_name = self.cat_id_to_name[cat_id]

            coco_eval = COCOeval(self.coco_gt, self.coco_dt, "bbox")
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()

            if len(coco_eval.eval["precision"]) > 0:
                precision = coco_eval.eval["precision"]
                precision = precision[precision > -1]
                map_category = float(np.mean(precision)) if len(precision) > 0 else 0.0

                precision_50 = coco_eval.eval["precision"][0, :, :, 0, 2]
                precision_50 = precision_50[precision_50 > -1]
                map_50_category = float(np.mean(precision_50)) if len(precision_50) > 0 else 0.0
            else:
                map_category = 0.0
                map_50_category = 0.0

            gt_count = len(self.coco_gt.getAnnIds(catIds=[cat_id]))
            pred_count = len([p for p in self.predictions if p["category_id"] == cat_id])

            category_results[cat_name] = {
                "category_id": cat_id,
                "mAP_0.5:0.95": map_category,
                "mAP_0.5": map_50_category,
                "gt_annotations": gt_count,
                "predictions": pred_count,
            }

        return category_results

    def analyze_prediction_distribution(self) -> Dict[str, Dict[str, float]]:
        logger.info("Analisi distribuzione predizioni:")

        pred_by_category: Dict[str, int] = defaultdict(int)
        scores_by_category: Dict[str, List[float]] = defaultdict(list)

        for pred in self.predictions:
            cat_id = pred["category_id"]
            cat_name = self.cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            pred_by_category[cat_name] += 1
            scores_by_category[cat_name].append(pred["score"])

        gt_by_category: Dict[str, int] = {}
        for cat_id in self.coco_gt.getCatIds():
            cat_name = self.cat_id_to_name[cat_id]
            gt_by_category[cat_name] = len(self.coco_gt.getAnnIds(catIds=[cat_id]))

        score_stats: Dict[str, Dict[str, float]] = {}
        for cat_name, scores in scores_by_category.items():
            if scores:
                score_stats[cat_name] = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(np.min(scores)),
                    "max_score": float(np.max(scores)),
                    "median_score": float(np.median(scores)),
                }

        distribution_analysis = {
            "predictions_by_category": dict(pred_by_category),
            "ground_truth_by_category": gt_by_category,
            "score_statistics": score_stats,
            "total_predictions": len(self.predictions),
            "total_categories_detected": len(pred_by_category),
            "total_categories_in_gt": len(gt_by_category),
        }

        return distribution_analysis

    def create_visualizations(
        self,
        metrics: Dict[str, float],
        category_results: Dict[str, Dict[str, float]],
        distribution_analysis: Dict[str, Dict[str, float]],
        figures_dir: Path,
        top_k_categories: int,
    ) -> None:
        logger.info("Creazione visualizzazioni in %s", figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 8))
        main_metrics = {
            "mAP@0.5:0.95": metrics["mAP_0.5:0.95"],
            "mAP@0.5": metrics["mAP_0.5"],
            "mAP@0.75": metrics["mAP_0.75"],
            "mAR@100": metrics["mAR_100"],
        }

        bars = plt.bar(
            main_metrics.keys(),
            main_metrics.values(),
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        plt.title("Metriche COCO Principali", fontsize=16, fontweight="bold")
        plt.ylabel("Valore Metrica", fontsize=12)
        plt.ylim(0, 1)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "main_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

        cat_results_sorted = sorted(
            category_results.items(),
            key=lambda item: item[1]["mAP_0.5"],
            reverse=True,
        )
        top_k = max(1, top_k_categories)
        top_categories = cat_results_sorted[:top_k]

        plt.figure(figsize=(15, 10))
        categories = [item[0] for item in top_categories]
        map_values = [item[1]["mAP_0.5"] for item in top_categories]

        bars = plt.barh(range(len(categories)), map_values, color="skyblue")
        plt.yticks(range(len(categories)), categories)
        plt.xlabel("mAP@0.5", fontsize=12)
        plt.title("Top categorie per mAP@0.5", fontsize=16, fontweight="bold")
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "top_categories_map.png", dpi=300, bbox_inches="tight")
        plt.close()

        common_cats = (
            set(distribution_analysis["predictions_by_category"].keys())
            & set(distribution_analysis["ground_truth_by_category"].keys())
        )
        common_cats = sorted(common_cats)[:top_k]

        pred_counts = [
            distribution_analysis["predictions_by_category"].get(cat, 0)
            for cat in common_cats
        ]
        gt_counts = [
            distribution_analysis["ground_truth_by_category"].get(cat, 0)
            for cat in common_cats
        ]

        x = np.arange(len(common_cats))
        width = 0.35

        plt.figure(figsize=(15, 8))
        plt.bar(x - width / 2, pred_counts, width, label="Predizioni", alpha=0.8, color="orange")
        plt.bar(x + width / 2, gt_counts, width, label="Ground Truth", alpha=0.8, color="blue")

        plt.xlabel("Categorie", fontsize=12)
        plt.ylabel("Numero di Istanze", fontsize=12)
        plt.title("Distribuzione Predizioni vs Ground Truth", fontsize=16, fontweight="bold")
        plt.xticks(x, common_cats, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "prediction_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

        size_metrics = np.array(
            [
                [metrics["mAP_small"], metrics["mAR_small"]],
                [metrics["mAP_medium"], metrics["mAR_medium"]],
                [metrics["mAP_large"], metrics["mAR_large"]],
            ]
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            size_metrics,
            xticklabels=["mAP", "mAR"],
            yticklabels=["Small", "Medium", "Large"],
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        plt.title("Performance per Dimensione Oggetto", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / "size_performance_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcola le metriche COCO a partire da predizioni in formato COCO JSON."
    )
    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Percorso al file delle annotazioni COCO (instances_val2017).",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Percorso al file JSON con le predizioni (default: predictions/predictions_coco_format.json).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Cartella base per reports/ e figures/ (default: struttura del repository).",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Percorso completo del report JSON (default: reports/detailed_evaluation_report.json).",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Cartella in cui salvare i grafici (default: reports/figures).",
    )
    parser.add_argument(
        "--top-k-categories",
        type=int,
        default=DEFAULT_TOP_K,
        help="Numero di categorie da mostrare nei grafici (default: 20).",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disabilita la generazione dei grafici riassuntivi.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, candidate: Optional[str], default: Path) -> Path:
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path
    return default


def main() -> Dict:
    args = parse_args()

    base_dir = (
        Path(args.output_root).expanduser()
        if args.output_root
        else Path(__file__).resolve().parents[1]
    )
    predictions_dir = base_dir / "predictions"
    reports_dir = base_dir / "reports"

    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = resolve_path(base_dir, args.figures_dir, reports_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = resolve_path(
        base_dir, args.predictions, predictions_dir / DEFAULT_PREDICTIONS_FILENAME
    )
    report_path = resolve_path(
        base_dir, args.report_path, reports_dir / DEFAULT_REPORT_FILENAME
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth_path = Path(args.ground_truth).expanduser()
    if not ground_truth_path.is_absolute():
        ground_truth_path = base_dir / ground_truth_path

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth non trovato: {ground_truth_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"File di predizioni non trovato: {predictions_path}")

    logger.info("Ground truth: %s", ground_truth_path)
    logger.info("Predizioni: %s", predictions_path)
    logger.info("Report JSON: %s", report_path)
    logger.info("Directory figure: %s", figures_dir)

    evaluator = COCOEvaluator(ground_truth_path, predictions_path)

    metrics, coco_eval = evaluator.evaluate_all_metrics()

    category_results = evaluator.evaluate_per_category()

    distribution_analysis = evaluator.analyze_prediction_distribution()

    if args.no_visualizations:
        logger.info("Generazione grafici disabilitata (--no-visualizations).")
    else:
        evaluator.create_visualizations(
            metrics,
            category_results,
            distribution_analysis,
            figures_dir,
            args.top_k_categories,
        )

    detailed_report = {
        "overall_metrics": metrics,
        "category_results": category_results,
        "distribution_analysis": distribution_analysis,
        "evaluation_summary": {
            "best_categories": sorted(
                category_results.items(), key=lambda x: x[1]["mAP_0.5"], reverse=True
            )[:10],
            "worst_categories": sorted(
                category_results.items(), key=lambda x: x[1]["mAP_0.5"]
            )[:10],
            "categories_with_no_predictions": [
                cat
                for cat, data in category_results.items()
                if data["predictions"] == 0
            ],
            "categories_with_no_ground_truth": [
                cat
                for cat, data in category_results.items()
                if data["gt_annotations"] == 0
            ],
        },
        "inputs": {
            "ground_truth_path": str(ground_truth_path),
            "predictions_path": str(predictions_path),
            "report_path": str(report_path),
            "figures_dir": str(figures_dir),
            "top_k_categories": args.top_k_categories,
            "visualizations_enabled": not args.no_visualizations,
        },
    }

    with report_path.open("w") as f:
        json.dump(detailed_report, f, indent=2)

    logger.info("\n%s", "=" * 60)
    logger.info("SUMMARY FINALE")
    logger.info("%s\n", "=" * 60)
    logger.info("mAP@0.5:0.95: %.4f", metrics["mAP_0.5:0.95"])
    logger.info("mAP@0.5: %.4f", metrics["mAP_0.5"])
    logger.info("mAP@0.75: %.4f", metrics["mAP_0.75"])
    logger.info("mAR@100: %.4f", metrics["mAR_100"])

    logger.info("\nMigliori 5 categorie (mAP@0.5):")
    for i, (cat_name, cat_data) in enumerate(
        detailed_report["evaluation_summary"]["best_categories"][:5], start=1
    ):
        logger.info("  %s. %s: %.4f", i, cat_name, cat_data["mAP_0.5"])

    logger.info("\nPeggiori 5 categorie (mAP@0.5):")
    for i, (cat_name, cat_data) in enumerate(
        detailed_report["evaluation_summary"]["worst_categories"][:5], start=1
    ):
        logger.info("  %s. %s: %.4f", i, cat_name, cat_data["mAP_0.5"])

    logger.info("\nFile salvati:")
    logger.info("  - Report dettagliato: %s", report_path)
    if not args.no_visualizations:
        logger.info("  - Visualizzazioni: %s", figures_dir)

    return detailed_report


if __name__ == "__main__":
    main()
