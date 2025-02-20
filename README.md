# 🏥 Pneumonia Detection from Chest X-Ray using CNNs

## 🚀 Project Overview
This project leverages **Convolutional Neural Networks (CNNs)** to detect **pneumonia** from chest X-ray images. The model has been built using **TensorFlow** and deployed using **Streamlit**, providing an easy-to-use web interface for healthcare professionals.


## 📂 Project Structure
```
PNEUMONIA_DETECTION_PROJECT/
│
├── app/
│   ├── saved_model/
│   │   └── model_VGG16.keras
│   ├── app.py
│   └── requirements.txt
│
├── Data/
│   └── [Training and validation images]
│
├── medical_env/  (Virtual environment for training)
│
├── models/
│   ├── model_cnn.keras
│   ├── model_DenseNet121.keras
│   ├── model_MobileNetV2_NonTrainable.keras
│   ├── model_MobileNetV2_Trainable.keras
│   └── model_VGG16.keras
│
├── Notebooks/
│   ├── model1.ipynb
│   ├── model2.ipynb
│   ├── model3.ipynb
│   ├── model4.ipynb
│   └── model5.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 🧠 Model Information
- **Architecture:** VGG16 (Fine-tuned for binary classification)
- **Input:** Chest X-ray images (RGB, resized to 224x224 pixels)
- **Output:** Binary classification (Normal vs Pneumonia)
- **Loss function:** Binary Cross-Entropy
- **Optimizer:** Adam

---

## 📊 Dataset
- Dataset used: **[NEU-DET Dataset for Defect Detection](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
- **Train:** 5216 images  
- **Test:** 624 images  
- **Validation:** 16 images

---

## 🛠️ Tech Stack
- **Backend:** TensorFlow, Keras
- **Frontend:** Streamlit
- **Languages:** Python
- **Deployment:** Streamlit (local deployment)

---

## ⚙️ Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

2️⃣ Create a virtual environment (recommended for deployment):
```bash
python -m venv deployment_env
source deployment_env/bin/activate  
```

3️⃣ Install dependencies:
```bash
pip install -r app/requirements.txt
```

4️⃣ Run the Streamlit app:
```bash
cd app
streamlit run app.py
```

---

## 🎯 How to Use
1. Upload a chest X-ray image (JPG/PNG format).
2. The model predicts whether the image shows signs of **Pneumonia** or is **Normal**.
3. View the result instantly!

---

## 📈 Model Performance
✅ **Accuracy:** 95%  
✅ **Precision:** 93%  
✅ **Recall:** 96%

---

## 🚀 Deployment
You can deploy this app using:
- **Heroku**
- **Streamlit Sharing**
- **AWS EC2 / GCP VM / Azure App Services**

---

## 🤖 Future Improvements
✨ Implement Grad-CAM to visualize CNN attention on the images.
✨ Train on a larger dataset for more generalizable predictions.
✨ Create a REST API for model inference.

---

## 📄 License

This project is open-source and available under the MIT License.
---

## ⭐ Acknowledgments
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Kaggle Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---


✨ **Happy Coding! 🚀**

