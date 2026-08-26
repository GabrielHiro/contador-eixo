#!/usr/bin/env python3
"""Processa uma imagem de um veículo e devolve JSON, sem treinamento."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vehicle-model", default="models/yolo26s.onnx")
    parser.add_argument("--axle-model", default="models/axles.onnx")
    parser.add_argument("--vehicle-conf", type=float, default=0.10)
    parser.add_argument("--axle-conf", type=float, default=0.10)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Imagem não encontrada ou inválida: {args.image}")
    vehicle_result = YOLO(args.vehicle_model, task="detect")(image, conf=args.vehicle_conf, verbose=False)[0]
    candidates = [box for box in vehicle_result.boxes if int(box.cls[0]) == 7]
    if not candidates:
        candidates = list(vehicle_result.boxes)
    if not candidates:
        cv2.imwrite(str(args.output), image)
        print(json.dumps({"vehicles": 0, "axles": 0, "error": "Veículo não detectado"}))
        return 0

    vehicle_box = max(candidates, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = [int(round(float(value))) for value in vehicle_box.xyxy[0]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    margin_x, margin_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    cx2, cy2 = min(image.shape[1], x2 + margin_x), min(image.shape[0], y2 + margin_y)
    crop = image[cy1:cy2, cx1:cx2]
    axle_result = YOLO(args.axle_model, task="detect")(crop, conf=args.axle_conf, verbose=False)[0]
    axle_count = len(axle_result.boxes)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    vehicle_name = vehicle_result.names[int(vehicle_box.cls[0])]
    cv2.putText(image, f"{vehicle_name} {float(vehicle_box.conf[0]):.2f}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    for box in axle_result.boxes:
        ax1, ay1, ax2, ay2 = [int(round(float(value))) for value in box.xyxy[0]]
        cv2.rectangle(image, (ax1 + cx1, ay1 + cy1), (ax2 + cx1, ay2 + cy1), (0, 220, 255), 2)
        cv2.putText(image, f"axle {float(box.conf[0]):.2f}", (ax1 + cx1, max(15, ay1 + cy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.rectangle(image, (10, 10), (430, 75), (0, 0, 0), -1)
    cv2.putText(image, f"Truck | Axles detected: {axle_count}", (20, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    print(json.dumps({"vehicles": 1, "axles": axle_count, "vehicle": vehicle_name, "confidence": round(float(vehicle_box.conf[0]), 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
