# Cow vs Buffalo Classifier

A machine learning web application built with Flask that can classify whether an uploaded image is of a cow or a buffalo.
It uses a Convolutional Neural Network (CNN) model trained on a custom dataset of cow and buffalo images.

---

## 📁 Project Structure

cow-buffalo-classifier/
│
├── dataset/
│ ├── cow/
│ ├── buffalo/
│
├── model/
│ └── model.h5 # trained model (ignored in .gitignore)
│
├── app/
│ ├── static/
│ │ └── uploads/ # uploaded images from web
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── app.py
│
├── train_model.py
├── requirements.txt
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
