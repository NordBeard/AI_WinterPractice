from ultralytics import YOLO
import datetime
import json
import os

model = YOLO("yolov8m.pt")

HISTORY_FILE = "history.json"
RESULT_FOLDER = "static/results"

os.makedirs(RESULT_FOLDER, exist_ok=True)


def detect_ships(image_path):
    results = model(image_path, save=True, project="static", name="results")

    ships = 0
    for r in results:
        for cls in r.boxes.cls:
            if model.names[int(cls)] in ["ship", "boat"]:
                ships += 1

    result_image = results[0].save_dir + "/" + os.path.basename(image_path)

    record = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_path,
        "result_image": result_image,
        "ships_detected": ships
    }

    history = load_history()
    history.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

    return ships, result_image


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


def get_statistics():
    history = load_history()
    total_requests = len(history)
    total_ships = sum(r["ships_detected"] for r in history)
    average = round(total_ships / total_requests, 2) if total_requests else 0

    return {
        "requests": total_requests,
        "ships": total_ships,
        "average": average
    }
