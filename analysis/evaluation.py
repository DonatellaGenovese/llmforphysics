import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Filepath to the saved CSV file
csv_file = "/hdd3/dongen/Desktop/llmforphysics/analysis/cached_data/Llama-3.1-8B-Instruct/Susy/answers_Llama-3.1-8B-Instruct.csv"

# Load the CSV file
data = pd.read_csv(csv_file)

print(data.head(40))

# --- Counts ---
total_responses = len(data)
nan_responses = data['model_response'].isna().sum()
nan_responses_label_background = data[data['model_response'].isna() & (data['original_label'] == 'background')].shape[0]
nan_responses_label_signal = data[data['model_response'].isna() & (data['original_label'] == 'signal')].shape[0]

# Print basic stats
print("📊 Basic Response Stats:")
print(f"Total number of rows                  : {total_responses}")
print(f"Number of NaN predictions             : {nan_responses}")
print(f"  └── where original label is 'background': {nan_responses_label_background}")
print(f"  └── where original label is 'signal'    : {nan_responses_label_signal}")

# --- Clean data for metric calculations ---
filtered_data = data.dropna(subset=['model_response'])

# Track unexpected responses
expected_responses = ['background', 'signal']
unexpected_data = filtered_data[~filtered_data['model_response'].isin(expected_responses)]
num_unexpected = len(unexpected_data)

print("\n❗ Unexpected Responses (not 'signal' or 'background'):")
print(f"Number of unexpected responses: {num_unexpected}")
if num_unexpected > 0:
    print("Some examples:")
    print(unexpected_data['model_response'].value_counts().head(10))  # Top 10 unexpected responses

# Filter for valid responses only
valid_data = filtered_data[filtered_data['model_response'].isin(expected_responses)]

# Map 'signal'/'background' to 1/0 for metric calculations
true_labels = valid_data['original_label'].map({'background': 0, 'signal': 1})
predictions = valid_data['model_response'].map({'background': 0, 'signal': 1})

# --- Metrics ---
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# --- Print metrics ---
print("\n✅ Evaluation Metrics (valid 'signal'/'background' only):")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")