import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from PIL import Image
from datetime import datetime
from PIL import ImageOps

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Face Recognition",
    page_icon="📸",
    layout="centered"
)

# =========================
# DARK MODE
# =========================

dark_mode = st.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    .stApp{
        background-color:#121212;
        color:white;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("""
<h1 style='text-align:center;color:#0D6EFD'>
🤖 AI FACE RECOGNITION
</h1>

<h4 style='text-align:center;color:gray'>
Nhận diện khuôn mặt bằng CNN
</h4>

<hr>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "best_model.keras"
)

# =========================
# CLASS NAMES
# =========================

class_names = [
    "HoangKyAnh",
    "Lê Quang Dũng",
    "Lê Tuấn Thành",
    "Lương Ngọc Thuận",
    "Nguyen Ngoc Bao",
    "Nguyễn Hoàng Quế Châu",
    "Nguyễn Phạm Hoàng An",
    "Nguyễn Thị Khánh Lê",
    "Nguyễn Thị Ngọc Tuyết",
    "Nguyễn Tiến Mạnh",
    "Nguyễn Việt Đức",
    "Nguyễn Đăng Vĩnh Phúc",
    "Ngô Quốc Trung",
    "Phạm Gia Thành Duy",
    "Phạm Hứa Nhật Minh",
    "Phạm Nguyễn Bảo Châu",
    "Phạm Phú Hoà",
    "Trần Hải Yến",
    "Vũ Quang Thái",
    "Đinh Hữu Khánh Anh",
    "Đoàn Hùng",
    "Đỗ An Phúc"
]

IMG_SIZE = (200, 200)

# =========================
# CAMERA
# =========================

st.subheader("📸 Chụp ảnh để nhận diện")

camera_image = st.camera_input(
    "Mở camera"
)

# =========================
# PREDICT
# =========================

if camera_image is not None:

    image = Image.open(
        camera_image
    ).convert("RGB")

    image = ImageOps.mirror(image)

    st.image(
        image,
        caption="Ảnh vừa chụp"
    )

    img = image.resize(
        IMG_SIZE
    )

    img_array = np.array(
        img
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    # =========================
    # UNKNOWN FACE
    # =========================

    if confidence < 70:

        st.error(
            "❌ Người không có trong hệ thống"
        )

        st.write(
            f"Độ tin cậy cao nhất chỉ: {confidence:.2f}%"
        )

        st.stop()

    predicted_name = class_names[
        predicted_index
    ]

    # =========================
    # RESULT CARD
    # =========================

    st.markdown(
        f"""
        <div style="
        background:linear-gradient(
        135deg,#0D6EFD,#4DA3FF);

        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        ">

        <h2>
        👤 {predicted_name}
        </h2>

        <h3>
        🎯 {confidence:.2f}%
        </h3>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.progress(
        confidence / 100
    )

    # =========================
    # ATTENDANCE
    # =========================

    attendance_file = (
        "attendance.csv"
    )

    current_time = (
        datetime.now()
        .strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    )

    new_record = pd.DataFrame({
        "Tên":[predicted_name],
        "Thời gian":[current_time]
    })

    if os.path.exists(
        attendance_file
    ):

        old = pd.read_csv(
            attendance_file
        )

        old = pd.concat(
            [old,new_record],
            ignore_index=True
        )

        old.to_csv(
            attendance_file,
            index=False
        )

    else:

        new_record.to_csv(
            attendance_file,
            index=False
        )

    st.success(
        "✅ Điểm danh thành công"
    )

    # =========================
    # TOP 3
    # =========================

    st.subheader(
        "📊 Top 3 dự đoán"
    )

    probs = prediction[0]

    top3 = np.argsort(
        probs
    )[::-1][:3]

    for idx in top3:

        st.write(
            f"👤 {class_names[idx]} — {probs[idx]*100:.2f}%"
        )

    # =========================
    # CHART
    # =========================

    chart_data = pd.DataFrame({

        "Tên":[
            class_names[i]
            for i in top3
        ],

        "Xác suất":[
            probs[i] * 100
            for i in top3
        ]
    })

    st.subheader(
        "📈 Biểu đồ xác suất"
    )

    st.bar_chart(
        chart_data.set_index(
            "Tên"
        )
    )

# =========================
# DOWNLOAD ATTENDANCE
# =========================

st.divider()

if os.path.exists(
    "attendance.csv"
):

    with open(
        "attendance.csv",
        "rb"
    ) as file:

        st.download_button(
            label="📥 Tải danh sách điểm danh",
            data=file,
            file_name="attendance.csv",
            mime="text/csv"
        )