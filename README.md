# 🎧 AI Mood-Based Music Recommender

A deep learning–based music recommendation system that predicts the **user’s mood from audio features** and suggests songs accordingly.
Built using a **hybrid CNN-LSTM architecture**, this project bridges the gap between static playlists and real-time emotional context.

---

## 📌 Problem Statement

Traditional music platforms recommend songs based on:

* Genre
* Listening history

However, they fail to capture the **user’s current emotional state**, leading to irrelevant recommendations.

---

## 💡 Solution

This system:

* Classifies songs into mood categories using deep learning
* Uses audio features to infer emotional tone
* Provides **personalized, mood-based recommendations** via a Streamlit app

---

## 📊 Dataset

* Source: Spotify Audio Features dataset (Kaggle)
* Size: ~2000 songs
* Features: 9 base audio features

### 🎯 Mood Labeling Strategy

Mood labels were generated using **feature-based thresholding**:

* High Energy → Energy > 0.8
* Chill / Acoustic → Acousticness > 0.7
* Groovy → Danceability > 0.7
* Focus / Instrumental → Instrumentalness > 0.5

This rule-based labeling simulates real-world mood tagging and enables supervised learning.

---

## 🔄 Methodology & Pipeline

1. Data Collection (Kaggle dataset)
2. Mood Label Generation
3. Feature Engineering (17 features: 9 base + 8 engineered)
4. Preprocessing (StandardScaler, train-test split)
5. Model Training (CNN-LSTM)
6. Evaluation (Accuracy, F1-score, confusion matrix)
7. Deployment (Streamlit web app)

---

## 🧠 Model Architecture

Hybrid Deep Learning Model:

* **Conv1D layers** → extract spatial/audio patterns
* **LSTM layers** → capture temporal dependencies
* **Dense layers** → classification

**Flow:**
Input → Conv1D → Conv1D → MaxPool → LSTM → Dense → Softmax

* Optimizer: Adam
* Epochs: 200
* Parameters: ~135K

---

## 📈 Results

* ✅ Training Accuracy: **95%**
* ✅ Test Accuracy: **85%**

### Per-Class F1 Score:

* High Energy: 0.94
* Chill/Acoustic: 0.91
* Groovy: 0.86
* Mixed Vibe: 0.72
* Focus/Instrumental: 0.68

---

## 🌐 Web Application (Streamlit)

The system includes an interactive UI with 3 modules:

### 1️⃣ Mood Explorer

* Select mood
* Get ranked song recommendations

### 2️⃣ Feature Predictor

* Adjust audio sliders
* Real-time mood prediction

### 3️⃣ Song Inspector

* Compare labeled vs predicted mood
* Visual radar chart

---

## 🎮 Example

**Input Mood:** High Energy

**Recommended Songs:**

* Blinding Lights
* Save Your Tears
* Levitating

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Streamlit
* Pandas, NumPy, Plotly

---

## 📂 Project Structure

```
ai-mood-music-recommender/
│
├── model/
├── app.py
├── mood_labeling.py
├── requirements.txt
├── README.md
```

---

## 🚀 Future Improvements

* Spotify API integration
* Real-time emotion detection (camera/text)
* Mobile application version

---

## 👩‍💻 Author

Charmi Vasa
B.Tech CSE (AI-ML)
