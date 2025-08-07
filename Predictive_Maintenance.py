import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from pycaret.classification import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Read data set
df = pd.read_csv("manufacturing_6G_dataset.csv")
df = df.copy()

print("first five rows:")
print(df.head())

# Describe dataset
print("\n"*2)
print("Describe the data set")
print(df.describe())

# Convert time stamp from string to datetime
print("\n"*2)
df["Timestamp"] = pd.to_datetime(df['Timestamp'], errors='coerce')
print('Converted Timestamp to datetime format (Unconvertible values will appear as NaT):')
print(df['Timestamp'].head())

#############################################
# 1. Data Preprocessing and cleaning
############################################

print("Data info:")
print(df.info())

#There are no missing values.
print("Missing values:")
print(df.isnull().sum())

#Convert categorical columns to category type if necessary
categorical_columns=["Operation_Mode","Efficiency_Status"]

for col in categorical_columns:
    df[col]=df[col].astype('category')

print("Data info:")
print(df.info())

#############################################
# 2. Explatory Data Analysis (EDA)
############################################

#Lets find numerical columns
num_cols=df.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col],kde=True,bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

#Analyze relationship between numerical values
# Corelation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df[num_cols].corr()

# Heat map
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f')

plt.title('Correlation Matrix of Numeric Features', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

#Let's show how many of each value there are in Efficiency_Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Efficiency_Status', hue='Efficiency_Status', palette='viridis', legend=False)
plt.title('Count Plot of Efficiency_Status')
plt.xlabel('Efficiency_Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#############################################
# 3. Feature Engineering
############################################
#Outlier Detection

def detect_outliers_iqr(dataframe, col_name):
    """
    Identifies outliers using the IQR method.
    """
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return dataframe[(dataframe[col_name] < lower_bound) | (dataframe[col_name] > upper_bound)]

#Let's check for outliers for each numerical variable.
for col in num_cols:
    outliers = detect_outliers_iqr(df, col)

for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}', fontsize=16, pad=20)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    outliers = detect_outliers_iqr(df, col)
    plt.suptitle(f'Boxplot of {col} ({len(outliers)} outliers)', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.show()


def handle_outliers_iqr(dataframe, col_name, method='remove'):
    """
    """
    Q1 = dataframe[col_name].quantile(0.25)
    Q3 = dataframe[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if method == 'remove':
        # Remove outliers
        return dataframe[(dataframe[col_name] >= lower_bound) & (dataframe[col_name] <= upper_bound)]
    elif method == 'cap':
        # Limit outliers
        dataframe[col_name] = np.where(dataframe[col_name] < lower_bound, lower_bound, dataframe[col_name])
        dataframe[col_name] = np.where(dataframe[col_name] > upper_bound, upper_bound, dataframe[col_name])
        return dataframe
    elif method == 'log_transform':
        # Log transformation
        if (dataframe[col_name] > 0).all():
            dataframe[col_name + '_log'] = np.log1p(dataframe[col_name])
        return dataframe


for col in ['Temperature_C', 'Vibration_Hz', 'Network_Latency_ms', 'Packet_Loss_%',
           'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
           'Predictive_Maintenance_Score', 'Error_Rate_%']:
    df = handle_outliers_iqr(df, col, method='cap')


#Time based feature extraction
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour


df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)


df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

print("New time-based features have been created.")
print(df[['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend']].head())

df = df.drop(['Month'], axis=1)

#Categorical encoding
# Label Encoding
le_operation = LabelEncoder()
le_efficiency = LabelEncoder()

df['Operation_Mode_encoded'] = le_operation.fit_transform(df['Operation_Mode'])
df['Efficiency_Status_encoded'] = le_efficiency.fit_transform(df['Efficiency_Status'])

print("Codding mappings:")
operation_mapping = dict(zip(le_operation.classes_, le_operation.transform(le_operation.classes_)))
efficiency_mapping = dict(zip(le_efficiency.classes_, le_efficiency.transform(le_efficiency.classes_)))

print(f"Operation_Mode: {operation_mapping}")
print(f"Efficiency_Status: {efficiency_mapping}")

target_original = df['Efficiency_Status_encoded'].copy()

#Feature Scaling
current_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

cols_to_scale = [col for col in current_num_cols if col not in ['Machine_ID', 'Efficiency_Status_encoded']]

print(f"Columns to be scaled: {len(cols_to_scale)} number")
print(f"Example columns: {cols_to_scale[:5]}...")

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("Scaling completed!")


#Feature Interaction

df_features = df.copy()

# Efficiency ratio
df_features['Power_Efficiency_Ratio'] = df_features['Power_Consumption_kW'] / (df_features['Production_Speed_units_per_hr'] + 1)

# Quality-Speed interaction
df_features['Quality_Speed_Interaction'] = df_features['Quality_Control_Defect_Rate_%'] * df_features['Production_Speed_units_per_hr']

# Temperature-Vibration combined stress indicator
df_features['Thermal_Mechanical_Stress'] = df_features['Temperature_C'] * df_features['Vibration_Hz']

# Network performance indicator
df_features['Network_Performance'] = df_features['Network_Latency_ms'] * df_features['Packet_Loss_%']

print("New interaction features have been created:")
print(['Power_Efficiency_Ratio', 'Quality_Speed_Interaction', 'Thermal_Mechanical_Stress', 'Network_Performance'])


# Finding high correlated features
def find_highly_correlated_features(dataframe, threshold=0.9):
    """
    This functions finds high correlated features
    """
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    corr_matrix = dataframe[num_cols].corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find high correlated pairs
    high_corr_pairs = []
    for col in upper_triangle.columns:
        high_corr_features = upper_triangle.index[upper_triangle[col] > threshold].tolist()
        if high_corr_features:
            for feature in high_corr_features:
                high_corr_pairs.append((col, feature, upper_triangle.loc[feature, col]))

    return high_corr_pairs

high_corr_features = find_highly_correlated_features(df_features, threshold=0.8)
print(f"High correlated pairs: {high_corr_features}")

feature_cols = [col for col in df.columns
                if col not in ['Efficiency_Status', 'Timestamp']]


target_column = 'Efficiency_Status_encoded'

exclude_columns = [
    'Efficiency_Status',
    'Efficiency_Status_encoded',
    'Timestamp',
    'Operation_Mode',
    'Operation_Mode_encoded',
    'Machine_ID'
]

feature_columns = [col for col in df.columns if col not in exclude_columns]

#Feature corelation control
high_corr_features = []
for col in feature_columns:
    if col in df.columns:
        corr = abs(df[col].corr(target_original))
        if corr > 0.95:
            high_corr_features.append(col)
            print(f"High corelation detected - {col}: {corr:.4f}")

if high_corr_features:
    print(f"Removed high corelated features: {high_corr_features}")
    feature_columns = [col for col in feature_columns if col not in high_corr_features]

print(f"Final feature count: {len(feature_columns)}")
print(f"Features: {feature_columns}")

import collections
duplicates = [item for item, count in collections.Counter(feature_columns).items() if count > 1]
if duplicates:
    print(f"Warning!: repetitive column names found in feature_columns: {duplicates}")

    feature_columns = list(dict.fromkeys(feature_columns))

if target_column in feature_columns:
    feature_columns.remove(target_column)
    print(f"'{target_column}' target variable removed from feature_columns list.")

print(f"\n📊 Final data summary:")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")


df = df.sort_values('Timestamp').reset_index(drop=True)
#Train-test split
X = df[feature_columns]
y = target_original

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y classes (first 10): {y.unique()[:10]}")

# Train-test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


print("\nTrain set time step:")
print(f"  Start: {df['Timestamp'].iloc[:split_index].min()}")
print(f"  End:     {df['Timestamp'].iloc[:split_index].max()}")

print("\nTest set time step:")
print(f"  Start: {df['Timestamp'].iloc[split_index:].min()}")
print(f"  End:     {df['Timestamp'].iloc[split_index:].max()}")


train_data_for_pycaret = X_train.copy()
train_data_for_pycaret[target_column] = y_train

exp_clf = setup(
    data=train_data_for_pycaret,
    target=target_column,
    session_id=42,
    normalize=True,
    transformation=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.85,
    data_split_stratify=False,
    data_split_shuffle=False,
    fold_shuffle=False,
    fold_strategy='timeseries',
    fold=3
)

best_models = compare_models(n_select=5, sort='Accuracy', cross_validation=True, fold=3)

best_model = best_models[0]

tuned_model = tune_model(best_model, n_iter=20, optimize='Accuracy')

final_model = finalize_model(tuned_model)

predictions = predict_model(final_model, data=X_test)

print("PyCaret Model Accuracy:", accuracy_score(y_test, predictions['prediction_label']))

"""
It is suspected that the data is not actual production data and may have been synthetically generated. 
No data leakage has been confirmed. The fact that some models have 100% accuracy raises the suspicion of overfitting. 
Therefore, models such as Logistic Regression or SVm, where accuracy values are acceptable, can be tried.

"""

# Use Logistic Regression for classification

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs',             # ✅
    class_weight='balanced',
    multi_class='multinomial'  # ✅
)


print("Training Logistic Regression model...")
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"\nLogistic Regression Accuracy: {lr_accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# ROC Curve ve AUC (for the binary classification)
if len(np.unique(y)) == 2:
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr[:, 1])
    auc_lr = auc(fpr_lr, tpr_lr)
    print(f"\nAUC Score: {auc_lr:.4f}")

    # draw ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_lr:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Feature Importance (Coefficients)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_model.coef_[0],
    'abs_coefficient': np.abs(lr_model.coef_[0])
})

feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]

plt.barh(top_features['feature'], top_features['coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Feature Importance (Top 15)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_efficiency.classes_,
            yticklabels=le_efficiency.classes_)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#Model Comparison
# PyCaret model accuracy
pycaret_accuracy = accuracy_score(y_test, predictions['prediction_label'])

comparison_results = pd.DataFrame({
    'Model': ['PyCaret Best Model', 'Logistic Regression'],
    'Accuracy': [pycaret_accuracy, lr_accuracy]
})

print("\nModel Comparison:")
print(comparison_results)

plt.figure(figsize=(10, 6))
plt.bar(comparison_results['Model'], comparison_results['Accuracy'],
        color=['skyblue', 'lightcoral'], alpha=0.8)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for i, v in enumerate(comparison_results['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

best_accuracy = max(pycaret_accuracy, lr_accuracy)
best_model_name = 'PyCaret Best Model' if pycaret_accuracy > lr_accuracy else 'Logistic Regression'

print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Best Accuracy: {best_accuracy:.4f}")

