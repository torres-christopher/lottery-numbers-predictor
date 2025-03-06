# Sportka Lottery Predictor

This repository contains three different machine learning models designed to predict Sportka lottery numbers using historical data. Each script offers a different approach to number prediction.

## 📜 Scripts Overview

| Script Name            | Model Type            | Description |
|------------------------|----------------------|-------------|
| `predictor-single.py`  | Logistic Regression  | Predicts one set of 6 numbers and 1 dodatkové číslo |
| `predictor-logistic.py`| Logistic Regression  | Uses a probability-based approach to select numbers |
| `predictor-multi.py`   | Multi-Predictor Logistic Regression | Uses a probability-based approach for multi-number predictions |

## 🎯 Features
- **Historical Data Analysis**: Uses past draws to find number trends.
- **Machine Learning Models**: Each script uses a different ML model.
- **Adjustable History Depth**: Choose how far back the analysis should go.
- **Probability-Based Selection**: Picks numbers based on historical frequency.
- **Accuracy Check**: Compares predictions to recent draws to evaluate performance.

## 🚀 Installation & Setup

### 1️⃣ **Install Dependencies**
Ensure you have **Python 3.x** installed, then install the required libraries:
```bash
pip install pandas numpy scikit-learn
```

### 2️⃣ **Prepare Your Data**
- Save your Sportka historical draw data as **sportka.csv** in the same directory.
- The dataset should have columns in the following format as per official [downloadable file](https://www.sazka.cz/loterie/historie-cisel?game=sportka):
  ```csv
  datum;rok;tyden;den;1_tah_1;2_tah_1;3_tah_1;4_tah_1;5_tah_1;6_tah_1;dodatkove_1;...
  5.3.2025;2025;10;3;44;32;11;3;17;33;24;39;49;41;31;33;23;3
  2.3.2025;2025;09;7;26;24;25;27;44;17;16;34;32;43;41;25;35;5
  ```

### 3️⃣ **Run a Prediction**
Execute one of the predictor scripts:
```bash
python predictor-single.py
```

Example output:
```
🎟️ Predicted Best Numbers for Next Draw: [3, 12, 21, 25, 37, 44]
🎯 Predicted Best Dodatkové Číslo: 23
✅ Accuracy Check: 2/6 numbers appeared in the last 10 draws
```

---

## 📌 Predictor Details

### **🔹 predictor-single.py**
- Uses **Logistic Regression** to predict one set of **6 numbers + 1 dodatkové číslo**.
- Trains on historical data and selects the most probable combination.

### **🔹 predictor-logistic.py**
- Uses **Logistic Regression with an Accuracy Check**.
- Predicts a **single set of numbers** and **compares them with the last 10 draws**.
- Displays how accurate past predictions were.

### **🔹 predictor-multi.py**
- Uses **Logistic Regression to predict multiple likely numbers**.
- Ranks numbers based on their probability of appearing.
- Includes **probability boost factors** (e.g., "Number 21 is 1.5x more likely than average").

---

## ⚙️ Configuration
Modify these parameters inside each script:
```python
HISTORY_DEPTH = 1000  # Number of past draws to analyze
ACCURACY_CHECK = 10   # Number of recent draws to compare predictions with
```

Try adjusting **HISTORY_DEPTH** (100, 300, 1000) to see how predictions change!

---

## 📊 Accuracy Evaluation
After generating predictions, the script checks **how often the predicted numbers appeared in past draws**:
```
✅ Accuracy Check:
🎟️ 3/6 of the predicted numbers appeared in the last 10 draws
🎯 ✅ The dodatkové číslo was correct
```

---

## ❗ Notes
- These predictors **use statistical analysis**, but **lotteries are random**. No method guarantees success.
- Test different history depths and compare results.
- Try different models (e.g., **Random Forest or Neural Networks**) for better accuracy.

📩 **Enjoy experimenting & good luck!** 🎰