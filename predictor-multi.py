import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

# Select how far back to analyze
HISTORY_DEPTH = 500  # Number of past draws to analyze
ACCURACY_CHECK = 10   # Number of recent draws to compare predictions with

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

# Prepare data for probability-based prediction
def prepare_probabilistic_data(df, window_size=5):
    X, y_main, y_bonus = [], [], []
    for i in range(len(df) - window_size):
        features = df.iloc[i:i+window_size, 4:10].values.flatten()
        target_numbers = df.iloc[i+window_size, 4:10].values.tolist()
        target_bonus = df.iloc[i+window_size, [10, 17]].values.tolist()

        # Convert target into binary labels (1 if a number appeared, 0 otherwise)
        target_main = [1 if num in target_numbers else 0 for num in range(1, 50)]
        target_bonus = [1 if num in target_bonus else 0 for num in range(1, 50)]
        
        X.append(features)
        y_main.append(target_main)
        y_bonus.append(target_bonus)
    
    return np.array(X), np.array(y_main), np.array(y_bonus)

# Filter out history outside of defined depth
df_filtered = df.iloc[:HISTORY_DEPTH] if HISTORY_DEPTH > 0 else df

# Prepare dataset
X, y_main, y_bonus = prepare_probabilistic_data(df_filtered, window_size=5)

# Train/Test Split
X_train, X_test, y_main_train, y_main_test = train_test_split(X, y_main, test_size=0.2, random_state=42)
_, _, y_bonus_train, y_bonus_test = train_test_split(X, y_bonus, test_size=0.2, random_state=42)

# Train Logistic Regression Models using One-vs-Rest
lr_main = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_main.fit(X_train, y_main_train)

lr_bonus = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_bonus.fit(X_train, y_bonus_train)

# Predict probability of each number appearing in the next draw
y_main_pred_probs = lr_main.predict_proba(X_test)
y_bonus_pred_probs = lr_bonus.predict_proba(X_test)

# Average probabilities across test samples
average_main_probs = np.mean(y_main_pred_probs, axis=0)
average_bonus_probs = np.mean(y_bonus_pred_probs, axis=0)

# Compute average probability for main numbers and "dodatkové číslo"
avg_main_prob = np.mean(average_main_probs)
avg_bonus_prob = np.mean(average_bonus_probs)

# Select the 6 most likely main numbers
predicted_main_numbers = np.argsort(average_main_probs)[-6:] + 1  # Adjust index to match number range (1-49)
predicted_bonus_number = np.argsort(average_bonus_probs)[-1] + 1  # Single best "dodatkové číslo"

# Compute probability boost factors
boost_factors_main = {num: round(average_main_probs[num-1] / avg_main_prob, 2) for num in predicted_main_numbers}
boost_factor_bonus = round(average_bonus_probs[predicted_bonus_number-1] / avg_bonus_prob, 2)

# Accuracy Check: Compare predictions with the last draws
last_draws = df.iloc[:ACCURACY_CHECK, 4:10].values.flatten().tolist()  # Ensure flat list
last_bonus = df.iloc[:ACCURACY_CHECK, [10, 17]].values.flatten().tolist()  # Ensure flat list

correct_main = len(set(predicted_main_numbers) & set(last_draws))
correct_bonus = len(set([predicted_bonus_number]) & set(last_bonus))

# Display results
print("-" * 50)
print("\n🎟️ Predicted Best Numbers for Next Draw:")
for num in sorted(predicted_main_numbers):
    print(f" - Number {num}: {boost_factors_main[num]}x more likely than average")

print(f"\n🎯 Predicted Best Dodatkové Číslo: {predicted_bonus_number}")
print(f" - Boost Factor: {boost_factor_bonus}x more likely than average")

# Accuracy evaluation
print(f"\n✅ Accuracy Check:")
print(f"🎟️ {correct_main}/6 of the predicted numbers appeared in the last {ACCURACY_CHECK} draws")
print(f"🎯 {'✅' if correct_bonus > 0 else '❌'} The dodatkové číslo was {'correct' if correct_bonus > 0 else 'incorrect'}\n")
print("-" * 50)
