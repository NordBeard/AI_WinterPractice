
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import json
import os
import time
import io


from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


# НАСТРОЙКИ


APP_TITLE = "🚢 ShipVision"
HISTORY_FILE = "ship_detection_history.json"

CONF_THRESHOLD = 0.35
IMG_SIZE = 1280
IOU_THRESHOLD = 0.5



st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚢",
    layout="wide",
)

st.title(APP_TITLE)
st.caption("Распознавание судов (YOLOv8m)")


# ФАЙЛ ИСТОРИИ


if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# МОДЕЛЬ

@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def load_history():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# SIDEBAR

st.sidebar.title("Разделы")
page = st.sidebar.radio(
    "",
    ["🚢 Детекция", "📊 История", "📈 Статистика", "📄 Отчеты"]
)

# ДЕТЕКЦИЯ

if page == "🚢 Детекция":
    st.subheader("📷 Детекция судов")

    uploaded = st.file_uploader(
        "Загрузите изображение",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with st.spinner("🔍 Анализируем..."):
            start = time.time()
            results = model(
                img_np,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                iou=IOU_THRESHOLD,
            )
            elapsed = time.time() - start

            out_img = img_np.copy()
            ship_count = 0
            confs = []

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 8:  # boat
                        ship_count += 1
                        conf_val = float(box.conf[0])
                        confs.append(conf_val)

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 180, 255), 2)
                        cv2.putText(
                            out_img,
                            f"Ship {conf_val:.2f}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 180, 255),
                            2,
                        )

        with col2:
            st.image(out_img, use_container_width=True)
            st.success(f"Найдено судов: {ship_count}")
            if confs:
                st.info(f"Средняя точность: {np.mean(confs):.2%}")
            st.caption(f"Время: {elapsed:.2f} сек")

        save_history({
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded.name,
            "ship_count": ship_count,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "processing_time": elapsed,
        })

# ИСТОРИЯ

elif page == "📊 История":
    st.subheader("📊 История")

    history = load_history()
    if not history:
        st.info("История пуста")
    else:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

        if st.button("🧹 Очистить историю"):
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            st.rerun()

# СТАТИСТИКА

elif page == "📈 Статистика":
    st.subheader("📈 Статистика")

    history = load_history()
    if not history:
        st.info("Нет данных")
    else:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        c1, c2, c3 = st.columns(3)
        c1.metric("Изображений", len(df))
        c2.metric("Всего судов", int(df["ship_count"].sum()))
        c3.metric("Среднее", f"{df['ship_count'].mean():.2f}")

        st.line_chart(df.groupby("date")["ship_count"].sum())


# ОТЧЕТЫ (PDF + JSON)

elif page == "📄 Отчеты":
    st.subheader("📄 Отчеты")

    history = load_history()
    if not history:
        st.info("Нет данных для отчета")
    else:
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # --- JSON ---
        st.download_button(
            "⬇️ Скачать JSON",
            json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="shipvision_data.json",
            mime="application/json",
        )

        # --- PDF ---
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="Cyr",
            fontName="HeiseiMin-W3",
            fontSize=10,
            leading=12
        ))

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []

        elements.append(Paragraph("Отчет ShipVision по детекции судов", styles["Cyr"]))
        elements.append(Paragraph(f"Всего изображений: {len(df)}", styles["Cyr"]))
        elements.append(Paragraph(f"Всего судов: {int(df['ship_count'].sum())}", styles["Cyr"]))

        table_data = [["Дата", "Файл", "Судов", "Средняя точность"]]
        for _, row in df.iterrows():
            table_data.append([
                row["timestamp"].strftime("%d.%m.%Y %H:%M"),
                row["filename"],
                str(row["ship_count"]),
                f"{row['avg_confidence']:.2f}",
            ])

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "HeiseiMin-W3"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ]))

        elements.append(table)
        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "⬇️ Скачать PDF",
            buffer,
            file_name="shipvision_report.pdf",
            mime="application/pdf",
        )

st.markdown("---")
st.caption("ShipVision | YOLOv8m | COCO boat")
