#############################################################
# BLOCK 1: SETUP AND DATA LOADING
#############################################################

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from collections import Counter
import time

# For data preprocessing
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline

# For anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# For modeling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as XGBClassifier
from sklearn.neural_network import MLPClassifier

# For evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# For handling class imbalance
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# For model interpretability
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading dataset...")
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
df = pd.read_csv(data_url)

# Display basic information
print(f"Dataset loaded successfully with shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Basic information about the dataset
print("\nDataset Information:")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"Feature names: {', '.join(df.columns)}")

#############################################################
# BLOCK 2: EXPLORATORY DATA ANALYSIS (EDA)
#############################################################

print("\n\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values check:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Statistical summary
print("\nStatistical summary of numerical features:")
print(df.describe().T)

# Target variable analysis
print("\nTarget variable distribution:")
target_counts = df['Machine failure'].value_counts()
print(target_counts)
print(f"Failure rate: {target_counts[1] / len(df):.2%}")

# Visualize target distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Machine failure', data=df, palette=['#4878D0', '#EE854A'])
plt.title('Distribution of Machine Failures', fontsize=16)
plt.xlabel('Machine Failure (0=No, 1=Yes)', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# Analyze specific failure types
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
print("\nSpecific failure types:")
for col in failure_types:
    print(f"{col}: {df[col].sum()} failures")

# Visualize failure types
plt.figure(figsize=(12, 7))
failure_counts = df[failure_types].sum().sort_values(ascending=False)
ax = sns.barplot(x=failure_counts.index, y=failure_counts.values, palette='viridis')
plt.title('Frequency of Different Failure Types', fontsize=16)
plt.xlabel('Failure Type', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add count labels on top of bars
for i, v in enumerate(failure_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_features = [col for col in numerical_cols if col not in ['UDI', 'Machine failure'] + failure_types]
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features for analysis: {numerical_features}")
print(f"Categorical features: {categorical_cols}")

# Correlation analysis
plt.figure(figsize=(14, 12))
correlation_matrix = df[numerical_cols].corr()
mask = np.triu(correlation_matrix)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt='.2f', annot_kws={"size": 8})
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()

# Distribution of numerical features by target class
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    
    # Create a subplot with 1 row and 2 columns
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=feature, hue='Machine failure', bins=30, kde=True, 
                 palette=['#4878D0', '#EE854A'], element="step", common_norm=False)
    plt.title(f'Distribution of {feature}', fontsize=14)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Machine failure', y=feature, data=df, palette=['#4878D0', '#EE854A'])
    plt.title(f'Boxplot of {feature} by Machine Failure', fontsize=14)
    
    plt.tight_layout()
    plt.show()

# Feature relationships - scatter plots matrix for key features
key_features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
plt.figure(figsize=(15, 15))
scatter_plot = sns.pairplot(df, vars=key_features, hue='Machine failure', 
                            palette=['#4878D0', '#EE854A'], plot_kws={'alpha': 0.5})
plt.suptitle('Scatter Plot Matrix of Key Features', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

#############################################################
# BLOCK 3: DATA CLEANING & PREPROCESSING
#############################################################

print("\n\n" + "="*50)
print("DATA CLEANING & PREPROCESSING")
print("="*50)

# Create a copy of the dataframe for cleaning
df_clean = df.copy()

# 3.1 Anomaly Detection
print("\nPerforming anomaly detection...")

def detect_anomalies(df, features, contamination=0.05):
    """
    Detect anomalies using Isolation Forest and Local Outlier Factor
    
    Parameters:
    df: DataFrame containing the data
    features: List of features to use for anomaly detection
    contamination: Expected proportion of outliers in the data
    
    Returns:
    DataFrame with anomaly scores and binary flags
    """
    # Create a copy of the dataframe
    df_anomaly = df.copy()
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_anomaly['anomaly_score_if'] = iso_forest.fit_predict(scaled_features)
    df_anomaly['anomaly_if'] = df_anomaly['anomaly_score_if'].apply(lambda x: 1 if x == -1 else 0)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    df_anomaly['anomaly_score_lof'] = lof.fit_predict(scaled_features)
    df_anomaly['anomaly_lof'] = df_anomaly['anomaly_score_lof'].apply(lambda x: 1 if x == -1 else 0)
    
    # Combined anomaly flag (if either method flags it)
    df_anomaly['is_anomaly'] = df_anomaly[['anomaly_if', 'anomaly_lof']].max(axis=1)
    
    return df_anomaly

# Apply anomaly detection
df_anomaly = detect_anomalies(df_clean, numerical_features)

# Summarize anomalies
anomaly_count = df_anomaly['is_anomaly'].sum()
print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df_anomaly):.2%} of the data)")

# Visualize anomalies in 2D
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x='Air temperature [K]', y='Process temperature [K]', 
                hue='is_anomaly', data=df_anomaly, palette={0: 'blue', 1: 'red'})
plt.title('Anomalies: Temperature Features')

plt.subplot(2, 2, 2)
sns.scatterplot(x='Rotational speed [rpm]', y='Torque [Nm]', 
                hue='is_anomaly', data=df_anomaly, palette={0: 'blue', 1: 'red'})
plt.title('Anomalies: Mechanical Features')

plt.subplot(2, 2, 3)
sns.scatterplot(x='Tool wear [min]', y='Torque [Nm]', 
                hue='is_anomaly', data=df_anomaly, palette={0: 'blue', 1: 'red'})
plt.title('Anomalies: Wear vs Torque')

plt.subplot(2, 2, 4)
sns.countplot(x='Machine failure', hue='is_anomaly', data=df_anomaly, palette={0: 'blue', 1: 'red'})
plt.title('Anomalies by Machine Failure')

plt.tight_layout()
plt.show()

# Analyze relationship between anomalies and failures
anomaly_failure_relation = pd.crosstab(df_anomaly['is_anomaly'], df_anomaly['Machine failure'])
print("\nRelationship between anomalies and failures:")
print(anomaly_failure_relation)

# 3.2 White Noise Testing
print("\nPerforming white noise testing...")

def is_white_noise(series, significance_level=0.05):
    """
    Test if a time series is white noise using the Ljung-Box test
    
    Parameters:
    series: Time series to test
    significance_level: Significance level for the test
    
    Returns:
    Boolean indicating if the series is white noise
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Perform Ljung-Box test
    result = acorr_ljungbox(series, lags=[10], return_df=True)
    p_value = result['lb_pvalue'].iloc[0]
    
    # If p-value > significance level, the series is white noise
    return p_value > significance_level, p_value

# Test each numerical feature for white noise
white_noise_results = {}
for feature in numerical_features:
    is_wn, p_value = is_white_noise(df_clean[feature])
    white_noise_results[feature] = {'is_white_noise': is_wn, 'p_value': p_value}

print("\nWhite noise test results:")
for feature, result in white_noise_results.items():
    status = "IS" if result['is_white_noise'] else "IS NOT"
    print(f"{feature} {status} white noise (p-value: {result['p_value']:.4f})")

# 3.3 Outlier Handling with Multiple Methods
print("\nHandling outliers...")

def detect_and_handle_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detect and handle outliers using various methods
    
    Parameters:
    df: DataFrame containing the data
    columns: List of columns to check for outliers
    method: Method to use ('zscore', 'iqr', or 'percentile')
    threshold: Threshold for outlier detection
    
    Returns:
    DataFrame with outliers handled and summary of outliers
    """
    df_clean = df.copy()
    outliers_summary = {}
    
    for col in columns:
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df_clean[col]))
            outliers = z_scores > threshold
        elif method == 'iqr':
            # IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        elif method == 'percentile':
            # Percentile method
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        
        # Count outliers
        outliers_count = outliers.sum()
        outliers_summary[col] = outliers_count
        
        # Handle outliers (winsorize - cap at boundaries)
        if outliers_count > 0:
            if method == 'zscore':
                # For z-score, replace with median
                median_value = df_clean[col].median()
                df_clean.loc[outliers, col] = median_value
            else:
                # For IQR and percentile, cap at boundaries
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
    
    return df_clean, outliers_summary

# Apply outlier detection and handling using IQR method
df_clean, outliers_summary_iqr = detect_and_handle_outliers(
    df_clean, numerical_features, method='iqr', threshold=1.5)

print("\nOutliers detected and handled using IQR method:")
for col, count in outliers_summary_iqr.items():
    if count > 0:
        print(f"{col}: {count} outliers ({count/len(df_clean):.2%} of data)")

# 3.4 Feature Transformation for Normality
print("\nTransforming features for normality...")

# Check skewness before transformation
skewness_before = df_clean[numerical_features].skew()
print("\nSkewness before transformation:")
print(skewness_before)

# Apply Box-Cox or Yeo-Johnson transformation for highly skewed features
pt = PowerTransformer(method='yeo-johnson')
df_transformed = df_clean.copy()
skewed_features = [col for col in numerical_features if abs(skewness_before[col]) > 0.5]

if skewed_features:
    # Transform only skewed features
    df_transformed[skewed_features] = pt.fit_transform(df_clean[skewed_features])
    
    # Check skewness after transformation
    skewness_after = df_transformed[skewed_features].skew()
    print("\nSkewness after transformation:")
    print(skewness_after)
    
    # Visualize before and after transformation for one feature
    sample_feature = skewed_features[0]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df_clean[sample_feature], kde=True)
    plt.title(f'Before Transformation: {sample_feature}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df_transformed[sample_feature], kde=True)
    plt.title(f'After Transformation: {sample_feature}')
    
    plt.tight_layout()
    plt.show()
else:
    print("No highly skewed features found, skipping transformation.")

# Use the transformed dataset for further analysis
df_clean = df_transformed

#############################################################
# BLOCK 4: FEATURE ENGINEERING (UPDATED)
#############################################################

print("\n\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# 4.1 Create Interaction Features
print("\nCreating interaction features...")
# Temperature difference
df_clean['Temp_Diff'] = df_clean['Air temperature [K]'] - df_clean['Process temperature [K]']

# Mechanical interactions
df_clean['Torque_per_RPM'] = df_clean['Torque [Nm]'] / (df_clean['Rotational speed [rpm]'] + 1)  # Add 1 to avoid division by zero
df_clean['Power_Estimate'] = df_clean['Torque [Nm]'] * df_clean['Rotational speed [rpm]']

# Tool wear interactions
df_clean['Wear_per_RPM'] = df_clean['Tool wear [min]'] / (df_clean['Rotational speed [rpm]'] + 1)
df_clean['Wear_Torque_Interaction'] = df_clean['Tool wear [min]'] * df_clean['Torque [Nm]']

# Temperature interactions
df_clean['Temp_RPM_Interaction'] = df_clean['Process temperature [K]'] * df_clean['Rotational speed [rpm]']
df_clean['Temp_Torque_Interaction'] = df_clean['Process temperature [K]'] * df_clean['Torque [Nm]']

# Polynomial features for key variables
df_clean['RPM_Squared'] = df_clean['Rotational speed [rpm]']**2
df_clean['Torque_Squared'] = df_clean['Torque [Nm]']**2
df_clean['Tool_Wear_Squared'] = df_clean['Tool wear [min]']**2

# Create combined failure indicator
df_clean['Failure_Type_Count'] = df_clean[failure_types].sum(axis=1)

# Check for string columns that might need encoding
print("\nChecking data types after feature engineering:")
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        print(f"Column '{col}' contains string values. Sample values: {df_clean[col].unique()[:5]}")
        
        # For 'Product ID' column, we'll drop it since it's likely just an identifier
        if col == 'Product ID':
            print(f"Dropping column '{col}' as it's an identifier")
            df_clean = df_clean.drop(col, axis=1)
        else:
            # For other categorical columns, use one-hot encoding
            print(f"One-hot encoding column '{col}'")
            df_clean = pd.get_dummies(df_clean, columns=[col], prefix=col)

# 4.3 Principal Component Analysis
print("\nPerforming Principal Component Analysis...")
# Get all numerical features including engineered ones (excluding identifiers and target)
all_numerical_features = [col for col in df_clean.columns 
                         if df_clean[col].dtype in ['float64', 'int64'] 
                         and col not in ['UDI', 'Machine failure'] + failure_types]

print(f"Number of numerical features for PCA: {len(all_numerical_features)}")

# Standardize the data before PCA
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_clean[all_numerical_features])

# Apply PCA
pca = PCA(n_components=5)  # Start with 5 components
pca_result = pca.fit_transform(scaled_features)

# Add PCA components to dataframe
for i in range(pca_result.shape[1]):
    df_clean[f'PCA_{i+1}'] = pca_result[:, i]

# Visualize explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.title('Cumulative Explained Variance Ratio by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Explained variance by components: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 5 components: {sum(pca.explained_variance_ratio_):.4f}")

# Visualize first two PCA components by failure status
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA_1', y='PCA_2', hue='Machine failure', 
                data=df_clean, palette=['#4878D0', '#EE854A'], s=70, alpha=0.7)
plt.title('PCA: First Two Principal Components by Machine Failure', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(title='Machine Failure', title_fontsize=12, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4.4 Feature Selection
print("\nPerforming feature selection...")
# Prepare X and y for feature selection
X = df_clean.drop(['UDI', 'Machine failure'] + failure_types, axis=1)
y = df_clean['Machine failure']

# Ensure all features are numeric
print("\nChecking for non-numeric features:")
non_numeric_columns = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_columns:
    print(f"Found non-numeric columns: {non_numeric_columns}")
    print("Dropping these columns before feature selection")
    X = X.drop(columns=non_numeric_columns)
else:
    print("All features are numeric. Proceeding with feature selection.")

# Method 1: Univariate feature selection (ANOVA F-value)
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)
univariate_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_,
    'P-value': selector.pvalues_
}).sort_values('Score', ascending=False)

print("\nTop 10 features by univariate selection (ANOVA F-value):")
print(univariate_scores.head(10))

# Method 2: Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 features by Random Forest importance:")
print(feature_importances.head(10))

# Visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
plt.title('Feature Importance from Random Forest', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()

# Method 3: Recursive Feature Elimination
print("\nPerforming Recursive Feature Elimination...")
rfe = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=10)
rfe.fit(X, y)
rfe_support = pd.DataFrame({
    'Feature': X.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print("\nTop features by RFE:")
print(rfe_support[rfe_support['Selected']].sort_values('Ranking'))

# Combine results from different feature selection methods
selected_features = list(set(
    univariate_scores.head(10)['Feature'].tolist() + 
    feature_importances.head(10)['Feature'].tolist() + 
    rfe_support[rfe_support['Selected']]['Feature'].tolist()
))

print(f"\nCombined selected features ({len(selected_features)}):")
print(selected_features)
#############################################################
# BLOCK 5: DATA PREPARATION FOR MODELING
#############################################################

print("\n\n" + "="*50)
print("DATA PREPARATION FOR MODELING")
print("="*50)

# 5.1 Prepare the final dataset
# Add 'Machine failure' to selected features for the final dataset
df_final = df_clean[selected_features + ['Machine failure']]

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Selected features ({len(selected_features)}):")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# 5.2 Split the data into train and test sets
X = df_final.drop('Machine failure', axis=1)
y = df_final['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Check class distribution
print("\nClass distribution in training set:")
print(Counter(y_train))
print(f"Failure rate in training set: {sum(y_train)/len(y_train):.2%}")

# 5.3 Handle Class Imbalance
print("\nHandling class imbalance with multiple techniques...")

# 5.3.1 SMOTE - Synthetic Minority Over-sampling Technique
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {Counter(y_train_smote)}")

# 5.3.2 ADASYN - Adaptive Synthetic Sampling
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
print(f"Class distribution after ADASYN: {Counter(y_train_adasyn)}")

# 5.3.3 SMOTETomek - Combination of SMOTE and Tomek links
smote_tomek = SMOTETomek(random_state=42)
X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTETomek: {Counter(y_train_smote_tomek)}")

# 5.3.4 Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"Class distribution after Random Under-Sampling: {Counter(y_train_rus)}")

# 5.4 Standardize Features
# Create a scaler for standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply scaling to all resampled datasets
X_train_smote_scaled = scaler.transform(X_train_smote)
X_train_adasyn_scaled = scaler.transform(X_train_adasyn)
X_train_smote_tomek_scaled = scaler.transform(X_train_smote_tomek)
X_train_rus_scaled = scaler.transform(X_train_rus)

# For simplicity in the rest of the code, we'll primarily use SMOTE resampling
print("\nUsing SMOTE resampled data for primary model training")

#############################################################
# BLOCK 6: MODEL TRAINING AND EVALUATION
#############################################################

print("\n\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# 6.1 Define models to evaluate
print("\n\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# 6.1 Define models to evaluate
# Fix the XGBoost import
from xgboost import XGBClassifier  # Correct import

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),  # Now correctly instantiated
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
}

# 6.2 Cross-validation evaluation
print("\nPerforming cross-validation evaluation...")
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    start_time = time.time()
    cv_scores = cross_val_score(model, X_train_smote_scaled, y_train_smote, 
                               cv=cv, scoring='accuracy')
    training_time = time.time() - start_time
    
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'training_time': training_time
    }
    
    print(f"{name}: Mean CV Score = {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), "
          f"Training Time = {training_time:.2f}s")

# 6.3 Train models on the full training set and evaluate on test set
print("\nTraining models on full training set and evaluating on test set...")
model_results = {}

for name, model in models.items():
    # Train the model
    start_time = time.time()
    model.fit(X_train_smote_scaled, y_train_smote)
    training_time = time.time() - start_time
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # For ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        # For SVM without probability
        y_score = model.decision_function(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
    
    # Store results
    model_results[name] = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'training_time': training_time,
        'inference_time': inference_time,
        'model': model
    }
    
    # Print results
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Inference Time: {inference_time:.4f}s")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 6.4 Visualize ROC curves
plt.figure(figsize=(12, 8))
for name, results in model_results.items():
    plt.plot(results['fpr'], results['tpr'], label=f"{name} (AUC = {results['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curves for Different Models', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6.5 Compare model performances
# Create a comparison table
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [results['accuracy'] for results in model_results.values()],
    'ROC AUC': [results['roc_auc'] for results in model_results.values()],
    'Precision (Class 1)': [results['classification_report']['1']['precision'] for results in model_results.values()],
    'Recall (Class 1)': [results['classification_report']['1']['recall'] for results in model_results.values()],
    'F1-Score (Class 1)': [results['classification_report']['1']['f1-score'] for results in model_results.values()],
    'Training Time (s)': [results['training_time'] for results in model_results.values()],
    'Inference Time (s)': [results['inference_time'] for results in model_results.values()]
}).sort_values('F1-Score (Class 1)', ascending=False)

print("\nModel Comparison:")
print(comparison_df)

# Visualize model comparison
metrics = ['Accuracy', 'ROC AUC', 'Precision (Class 1)', 'Recall (Class 1)', 'F1-Score (Class 1)']
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    ax = sns.barplot(x='Model', y=metric, data=comparison_df)
    plt.title(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 6.6 Select the best model based on F1-Score for the minority class
best_model_name = comparison_df.iloc[0]['Model']
best_model = model_results[best_model_name]['model']
print(f"\nBest performing model: {best_model_name}")

#############################################################
# BLOCK 7: HYPERPARAMETER TUNING
#############################################################

print("\n\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# 7.1 Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

# Define parameter grid based on the best model
if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)]
    }
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'class_weight': [None, 'balanced']
    }
elif best_model_name == 'Neural Network':
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam', 'sgd']
    }

# Use RandomizedSearchCV for more efficient hyperparameter tuning
grid_search = RandomizedSearchCV(
    estimator=best_model, 
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # Optimize for F1 score
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit grid search
grid_search.fit(X_train_smote_scaled, y_train_smote)

# Get best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation F1 score: {best_score:.4f}")

# 7.2 Train a new model with the best parameters
print("\nTraining a new model with the best parameters...")
# Create a new model instance with the best parameters
if best_model_name == 'Logistic Regression':
    tuned_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
elif best_model_name == 'Random Forest':
    tuned_model = RandomForestClassifier(**best_params, random_state=42)
elif best_model_name == 'Gradient Boosting':
    tuned_model = GradientBoostingClassifier(**best_params, random_state=42)
elif best_model_name == 'XGBoost':
    tuned_model = XGBClassifier(**best_params, random_state=42)
elif best_model_name == 'SVM':
    tuned_model = SVC(**best_params, probability=True, random_state=42)
elif best_model_name == 'Neural Network':
    tuned_model = MLPClassifier(**best_params, random_state=42, max_iter=1000)

# Train the tuned model
tuned_model.fit(X_train_smote_scaled, y_train_smote)

# Evaluate the tuned model
y_pred_tuned = tuned_model.predict(X_test_scaled)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
report_tuned = classification_report(y_test, y_pred_tuned)

print(f"\n--- {best_model_name} (Tuned) Results ---")
print(f"Accuracy: {accuracy_tuned:.4f}")
print("\nConfusion Matrix:")
print(cm_tuned)
print("\nClassification Report:")
print(report_tuned)

# Compare ROC curves before and after tuning
if hasattr(tuned_model, "predict_proba"):
    y_prob_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)
    roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)
else:
    y_score_tuned = tuned_model.decision_function(X_test_scaled)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_score_tuned)
    roc_auc_tuned = roc_auc_score(y_test, y_score_tuned)

plt.figure(figsize=(10, 8))
plt.plot(model_results[best_model_name]['fpr'], model_results[best_model_name]['tpr'], 
         label=f"Before Tuning (AUC = {model_results[best_model_name]['roc_auc']:.4f})")
plt.plot(fpr_tuned, tpr_tuned, 
         label=f"After Tuning (AUC = {roc_auc_tuned:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title(f'ROC Curve Before and After Tuning: {best_model_name}', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#############################################################
# BLOCK 8: MODEL INTERPRETATION
#############################################################

print("\n\n" + "="*50)
print("MODEL INTERPRETATION")
print("="*50)

# 8.1 Feature Importance Analysis
print("\nAnalyzing feature importance...")

# Different methods of feature importance based on model type
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # Tree-based models have feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': tuned_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature importance from the model:")
    print(feature_importances)
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title(f'Feature Importance from {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
elif best_model_name == 'Logistic Regression':
    # For logistic regression, we can use the coefficients
    coef = tuned_model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coef,
        'Absolute': np.abs(coef)
    }).sort_values('Absolute', ascending=False)
    
    print("\nFeature coefficients from Logistic Regression:")
    print(feature_importance)
    
    # Visualize coefficients
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
    plt.title('Feature Coefficients from Logistic Regression', fontsize=16)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

# 8.2 SHAP Values for Model Interpretation
print("\nCalculating SHAP values for model interpretation...")

# Create a SHAP explainer based on the model type
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    explainer = shap.TreeExplainer(tuned_model)
    # Get SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)
    
    if isinstance(shap_values, list):
        # For multi-class, take the values for class 1 (failure)
        shap_values = shap_values[1]
        
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title(f'SHAP Summary Plot for {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Dependence plots for top features
    top_features = feature_importances.head(3)['Feature'].values
    for feature in top_features:
        plt.figure(figsize=(10, 7))
        feature_idx = list(X.columns).index(feature)
        shap.dependence_plot(feature_idx, shap_values, X_test_scaled, 
                            feature_names=X.columns, show=False)
        plt.title(f'SHAP Dependence Plot for {feature}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
elif best_model_name in ['XGBoost']:
    explainer = shap.TreeExplainer(tuned_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title(f'SHAP Summary Plot for {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
elif best_model_name in ['Logistic Regression', 'SVM', 'Neural Network']:
    # For these models, use KernelExplainer which is model-agnostic
    # Create a background dataset for the explainer using a sample of training data
    background = shap.sample(X_train_scaled, 100)
    explainer = shap.KernelExplainer(
        tuned_model.predict_proba if hasattr(tuned_model, "predict_proba") 
        else tuned_model.predict, 
        background
    )
    
    # Calculate SHAP values for a sample of test data (for efficiency)
    sample_indices = np.random.choice(X_test_scaled.shape[0], 100, replace=False)
    shap_values = explainer.shap_values(X_test_scaled[sample_indices])
    
    if isinstance(shap_values, list):
        # For multi-class, take the values for class 1 (failure)
        shap_values = shap_values[1]
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled[sample_indices], 
                     feature_names=X.columns, show=False)
    plt.title(f'SHAP Summary Plot for {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

# 8.3 Error Analysis
print("\nPerforming error analysis...")

# Get predictions and probabilities
y_pred = tuned_model.predict(X_test_scaled)
if hasattr(tuned_model, "predict_proba"):
    y_prob = tuned_model.predict_proba(X_test_scaled)[:, 1]
else:
    # For SVM without probability
    y_prob = tuned_model.decision_function(X_test_scaled)

# Identify misclassified samples
misclassified = y_test != y_pred
misclassified_indices = np.where(misclassified)[0]
misclassified_count = misclassified.sum()

# Count false positives and false negatives
false_positives = ((y_pred == 1) & (y_test == 0)).sum()
false_negatives = ((y_pred == 0) & (y_test == 1)).sum()

print(f"Total test samples: {len(y_test)}")
print(f"Correctly classified: {len(y_test) - misclassified_count} ({(1 - misclassified_count/len(y_test)):.2%})")
print(f"Misclassified: {misclassified_count} ({misclassified_count/len(y_test):.2%})")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")

# Analyze characteristics of misclassified samples
if misclassified_count > 0:
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_test_df['true_label'] = y_test.values
    X_test_df['predicted'] = y_pred
    X_test_df['probability'] = y_prob
    X_test_df['misclassified'] = misclassified
    
    # Compare feature distributions for correctly vs incorrectly classified samples
    for feature in X.columns[:5]:  # Limit to first 5 features for brevity
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='misclassified', y=feature, data=X_test_df)
        plt.title(f'Distribution of {feature} by Classification Result', fontsize=14)
        plt.xlabel('Misclassified (1=Yes, 0=No)', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # Analyze false positives vs false negatives
    fp_samples = X_test_df[(X_test_df['predicted'] == 1) & (X_test_df['true_label'] == 0)]
    fn_samples = X_test_df[(X_test_df['predicted'] == 0) & (X_test_df['true_label'] == 1)]
    
    print("\nFalse Positive characteristics:")
    print(fp_samples.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\nFalse Negative characteristics:")
    print(fn_samples.describe().T[['mean', 'std', 'min', 'max']])

#############################################################
# BLOCK 9: FINAL MODEL AND SUMMARY
#############################################################

print("\n\n" + "="*50)
print("FINAL MODEL AND SUMMARY")
print("="*50)

# 9.1 Final Model Selection
print(f"\nFinal selected model: {best_model_name} (Tuned)")

# Performance metrics of the final model
print("\nPerformance metrics on test data:")
print(f"Accuracy: {accuracy_tuned:.4f}")
print(f"ROC AUC: {roc_auc_tuned:.4f}")
print(f"Precision (Class 1): {classification_report(y_test, y_pred_tuned, output_dict=True)['1']['precision']:.4f}")
print(f"Recall (Class 1): {classification_report(y_test, y_pred_tuned, output_dict=True)['1']['recall']:.4f}")
print(f"F1-Score (Class 1): {classification_report(y_test, y_pred_tuned, output_dict=True)['1']['f1-score']:.4f}")

# 9.2 Summary of the project
print("\nProject Summary:")
print("1. Data exploration revealed an imbalanced dataset with around 3% failure rate")
print("2. Created engineered features including temperature differences, mechanical interactions, and more")
print("3. Applied PCA for dimensionality reduction")
print("4. Used SMOTE to address class imbalance")
print(f"5. Evaluated multiple models, with {best_model_name} performing best")
print("6. Tuned hyperparameters to optimize model performance")
print("7. Analyzed feature importance and model interpretability using SHAP values")
print("8. Performed error analysis to understand model limitations")

# 9.3 Recommendations
print("\nRecommendations:")
print("1. Monitor features identified as most important for early failure detection")
print("2. Implement regular model retraining as new data becomes available")
print("3. Consider cost-sensitive learning approaches for this imbalanced problem")
print("4. Explore ensemble methods combining multiple models for improved performance")
print("5. Collect additional data for failure cases to improve model robustness")

# 9.4 Save the final model (if needed)
# import joblib
# joblib.dump(tuned_model, 'equipment_failure_prediction_model.pkl')
# joblib.dump(scaler, 'feature_scaler.pkl')
# print("\nModel and scaler saved for future use")

print("\n" + "="*50)
print("END OF ANALYSIS")
print("="*50)
