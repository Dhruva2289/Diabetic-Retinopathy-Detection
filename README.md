# 🩺 Diabetic Retinopathy Detection Using CNN

This project is a Deep Learning web application that detects diabetic retinopathy from retina images using a Convolutional Neural Network (CNN).

---

# 🚀 Features

- Upload retina image
- Predict diabetic retinopathy severity
- CNN-based image classification
- Streamlit web application
- Beginner-friendly AI project

---

# 🧠 Technologies Used

- Python
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Streamlit
- CNN (Convolutional Neural Network)

---

# 📂 Project Structure

```text
aptos2019-blindness-detection/
│
├── train_images/
├── train.csv
├── train.py
├── app.py
├── model.h5
├── requirements.txt
└── README.md
```

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone <your-github-repo-link>
```

---

## 2. Open Project Folder

```bash
cd aptos2019-blindness-detection
```

---

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

---

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train Model

Run:

```bash
python train.py
```

This will generate:

```text
model.h5
```

---

# 🌐 Run Web App

```bash
python -m streamlit run app.py
```

---

# 🩺 Prediction Classes

| Label | Meaning |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

# 📊 Dataset

Dataset used:

APTOS 2019 Blindness Detection Dataset from Kaggle.

---

# ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

---

# 👨‍💻 Author

Dhruv Kumar jha