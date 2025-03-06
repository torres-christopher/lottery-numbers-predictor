import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

# Select how far back to analyze and accuracy check
HISTORY_DEPTH = 500  
ACCURACY_CHECK = 10

# Define column names based on dataset structure
columns = [
    "datum", "rok", "tyden", "den", 
    "1_tah_1", "2_tah_1", "3_tah_1", "4_tah_1", "5_tah_1", "6_tah_1", "dodatkove_1",
    "1_tah_2", "2_tah_2", "3_tah_2", "4_tah_2", "5_tah_2", "6_tah_2", "dodatkove_2"
]

# Load dataset
df = pd.read_csv("sportka.csv", delimiter=";", names=columns, skiprows=1)
df.iloc[:, 4:11] = df.iloc[:, 4:11].astype(int)
df.iloc[:, 11:18] = df.iloc[:, 11:18].astype(int)

# Prepare dataset
def prepare_data(df, window_size=5):
    X, y_main, y_bonus = [], [], []
    for i in range(len(df) - window_size):
        features = df.iloc[i:i+window_size, 4:10].values.flatten()
        target_numbers = df.iloc[i+window_size, 4:10].values.tolist()
        target_bonus = df.iloc[i+window_size, [10, 17]].values.tolist()
        target_main = [1 if num in target_numbers else 0 for num in range(1, 50)]
        target_bonus = [1 if num in target_bonus else 0 for num in range(1, 50)]
        X.append(features)
        y_main.append(target_main)
        y_bonus.append(target_bonus)
    return np.array(X), np.array(y_main), np.array(y_bonus)

df_filtered = df.iloc[:HISTORY_DEPTH] if HISTORY_DEPTH > 0 else df
X, y_main, y_bonus = prepare_data(df_filtered, window_size=5)

# Train/Test Split
X_train, X_test, y_main_train, y_main_test = train_test_split(X, y_main, test_size=0.2, random_state=42)
_, _, y_bonus_train, y_bonus_test = train_test_split(X, y_bonus, test_size=0.2, random_state=42)

# Train Logistic Regression Models
lr_main = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_main.fit(X_train, y_main_train)

lr_bonus = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_bonus.fit(X_train, y_bonus_train)

# Predict probability of each number appearing in the next draw
y_main_pred_probs = lr_main.predict_proba(X_test[-1].reshape(1, -1))
y_bonus_pred_probs = lr_bonus.predict_proba(X_test[-1].reshape(1, -1))

# Select the top 6 most likely numbers
predicted_numbers = np.argsort(y_main_pred_probs[0])[-6:] + 1
predicted_numbers = [int(num) for num in predicted_numbers]  # Convert np.int64 to int

# Select the best dodatkové číslo
predicted_bonus_number = int(np.argsort(y_bonus_pred_probs[0])[-1] + 1)  # Convert to integer

# Accuracy Check: Compare predictions with the last draws
last_draws = df.iloc[:ACCURACY_CHECK, 4:10].values.flatten().tolist()  # Ensure flat list
last_bonus = df.iloc[:ACCURACY_CHECK, [10, 17]].values.flatten().tolist()  # Ensure flat list

correct_main = len(set(predicted_numbers) & set(last_draws))
correct_bonus = len(set([predicted_bonus_number]) & set(last_bonus))

# Display results
print("-" * 50)
print("\n🎟️ Predicted Best Numbers for Next Draw:", predicted_numbers)
print(f"🎯 Predicted Best Dodatkové Číslo: {predicted_bonus_number}")

# Accuracy evaluation
print(f"\n✅ Accuracy Check:")
print(f"🎟️ {correct_main}/6 of the predicted numbers appeared in the last {ACCURACY_CHECK} draws")
print(f"🎯 {'✅' if correct_bonus > 0 else '❌'} The dodatkové číslo was {'correct' if correct_bonus > 0 else 'incorrect'}\n")
print("-" * 50)
