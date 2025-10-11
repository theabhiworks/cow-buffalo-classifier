# Cow vs Buffalo Classifier

A machine learning web application built with Flask that can classify whether an uploaded image is of a cow or a buffalo.
It uses a Convolutional Neural Network (CNN) model trained on a custom dataset of cow and buffalo images.

---

cow-buffalo-classifier/
│
├── dataset/
│   ├── cow/
│   └── buffalo/
│
├── model/
│   └── model.h5            # trained CNN model
│
├── app/
│   ├── static/
│   │   └── uploads/        # uploaded images
│   ├── templates/
│   │   └── index.html      # user interface
│   └── app.py              # Flask backend
│
├── train_model.py          # training script
├── requirements.txt        # dependencies
└── README.md

---

## 🧠 Features
✅ Classifies uploaded images as **Cow** or **Buffalo**  
✅ Built using **TensorFlow + Keras**  
✅ Simple **Flask** backend  
✅ Fully responsive **Bootstrap 5 UI**

---

## 🧾 Requirements

Python 3.8 or above
TensorFlow
Flask
Pillow
NumPy
scikit-learn
