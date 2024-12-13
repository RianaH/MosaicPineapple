
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

# Replace 'metrics.xlsx' with the path to your Excel file
file_path = "/Users/rianahoagland/PycharmProjects/MosaicP/lib/final_report_metrics .xlsx"

# Read the Excel file using openpyxl engine
data = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')

# Updated model names (excluding the last one)
models = [
    'ChatGPT',
    'Llama 1B 0-shot',
    'Llama 1B 1-shot',
    'Llama 1B COT',
    'Llama 3B 0-shot',
    'Llama 3B 1-shot',
    'Llama 3B COT',
    'Llama 8B 0-shot',
    'Llama 8B 1-shot',
    'Llama 8B COT',
]

# Function to divide each element in a column (list) by the "totalSamples" value
def process_counts(row, column_name):
    counts = ast.literal_eval(row[column_name])  # Convert string representation of list to an actual list
    total_samples = row['totalSamples']
    return [x / total_samples for x in counts]

# Process "trueCounts", "modelCounts", and "correctCounts" columns
data['trueCount_STD'] = data.apply(lambda row: process_counts(row, 'trueCounts'), axis=1)
data['modelCount_STD'] = data.apply(lambda row: process_counts(row, 'modelCounts'), axis=1)
data['correctCount_STD'] = data.apply(lambda row: process_counts(row, 'correctCounts'), axis=1)

# Convert "modelCounts", "correctCounts", and "trueCounts" to percentages
data['modelCount_PERCENT'] = data['modelCount_STD'].apply(lambda x: [val * 100 for val in x])
data['correctCount_PERCENT'] = data['correctCount_STD'].apply(lambda x: [val * 100 for val in x])
data['trueCount_PERCENT'] = data['trueCount_STD'].apply(lambda x: [val * 100 for val in x])

# Calculate average trueCounts across categories (excluding last row)
average_true_counts = np.mean(data['trueCount_PERCENT'].tolist(), axis=0)

# Calculate percentages for "correct", "tooHigh", and "tooLow"
columns_to_plot = ['correct', 'tooHigh', 'tooLow']
for column in columns_to_plot:
    data[f'{column}_PERCENT'] = (data[column] / data['totalSamples']) * 100

# Add a new column for the percentage of time "withinLetter" was oneCorrect
data['withinLetter_PERCENT'] = (data['withinLetter'] / data['oneCorrect']) * 100

# Adjust the percentages to standardize for a 50/50 chance assumption
data['withinLetter_STANDARDIZED'] = (data['withinLetter_PERCENT'] / 60) * 50  # Scale observed percentage for uniform distribution

# Plot 1: ModelCounts as Percentages Across Categories with Average TrueCounts
x_labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
x = np.arange(len(x_labels)) * 1.5  # Increase spacing between categories
width = 0.1

plt.figure(figsize=(14, 7))
for i, model in enumerate(models):
    plt.bar(x + i * width, data['modelCount_PERCENT'].iloc[i], width, label=f'{models[i]}')
plt.plot(x, average_true_counts, 'k--', label='Averaged TrueCounts', linewidth=2)
plt.xlabel('Categories')
plt.ylabel('Percentage (%)')
plt.title('ModelCounts as Percentages Across Categories')
plt.xticks(x + width * (len(models) - 1) / 2, x_labels)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: CorrectCounts as Percentages Across Categories with Average TrueCounts
plt.figure(figsize=(14, 7))
for i, model in enumerate(models):
    plt.bar(x + i * width, data['correctCount_PERCENT'].iloc[i], width, label=f'{models[i]}')
plt.plot(x, average_true_counts, 'k--', label='Averaged TrueCounts', linewidth=2)
plt.xlabel('Categories')
plt.ylabel('Percentage (%)')
plt.title('CorrectCounts as Percentages Across Categories')
plt.xticks(x + width * (len(models) - 1) / 2, x_labels)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Prediction Accuracy: Correct, One Off, Two Off, Three Off, and Four Off
columns_to_standardize = ['correct', 'oneCorrect', 'twoCorrect', 'threeCorrect', 'fourCorrect']
new_labels = ['Correct', 'One Off', 'Two Off', 'Three Off', 'Four Off']
x = np.arange(len(models))
width = 0.15
plt.figure(figsize=(16, 8))
for i, (column, label) in enumerate(zip(columns_to_standardize, new_labels)):
    data[f'{column}_PERCENT'] = (data[column] / data['totalSamples']) * 100
    plt.bar(x + i * width, data[f'{column}_PERCENT'], width, label=label)
plt.xlabel('Models')
plt.ylabel('Percentage (%)')
plt.title('Prediction Accuracy: Correct, One Off, Two Off, Three Off, and Four Off')
plt.xticks(x + width * (len(columns_to_standardize) - 1) / 2, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Standardized Percentage of Within-Letter Guesses When One Off
plt.figure(figsize=(12, 6))
plt.bar(x, data['withinLetter_STANDARDIZED'], width=0.6, label='Standardized Percent Within Same Letter When One Off')
plt.xlabel('Models')
plt.ylabel('Standardized Percentage (%)')
plt.title('Standardized Percentage of Within-Letter Guesses When One Off (50/50 Adjusted)')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 5: Percentage of Correct, Too High, and Too Low Guesses for Each Model
plt.figure(figsize=(14, 7))
width = 0.25
for i, column in enumerate(columns_to_plot):
    plt.bar(x + i * width, data[f'{column}_PERCENT'], width, label=column.capitalize())
plt.xlabel('Models')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Correct, Too High, and Too Low Guesses for Each Model')
plt.xticks(x + width, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 6: Percentage of Correct Guesses for Each Model
plt.figure(figsize=(12, 6))
plt.bar(x, data['correct_PERCENT'], width=0.6, label='Percent Correct Guesses')
plt.xlabel('Models')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Correct Guesses for Each Model')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
