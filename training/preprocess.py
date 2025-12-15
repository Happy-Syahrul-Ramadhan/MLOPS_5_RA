"""
Data Preprocessing Module for Customer Churn Prediction
Handles data loading, cleaning, encoding, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split


class ChurnDataPreprocessor:
    """
    Preprocessor untuk data Customer Churn.
    Melakukan cleaning, encoding, scaling, dan balancing data.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.numeric_cols = None
        self.one_hot_cols_multi = None
        self.one_hot_cols_binary = None
        self.label_encoding_cols = ['Contract']
        
    def load_data(self, filepath):
        """
        Load data dari CSV file
        
        Args:
            filepath (str): Path ke file CSV
            
        Returns:
            pd.DataFrame: DataFrame yang sudah di-load
        """
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """
        Membersihkan data: handle missing values, convert types, drop unnecessary columns
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with mean
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].mean())
        
        # Drop customerID (tidak berguna untuk prediksi)
        if 'customerID' in df_clean.columns:
            df_clean.drop('customerID', axis=1, inplace=True)
        
        print(f"✓ Data cleaned: {df_clean.isnull().sum().sum()} missing values remaining")
        return df_clean
    
    def encode_features(self, X):
        """
        Encode categorical features:
        - One-hot encoding untuk multi-class
        - Label encoding untuk Contract
        - Binary encoding untuk binary columns
        
        Args:
            X (pd.DataFrame): Feature dataframe
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X_encoded = X.copy()
        
        # Identifikasi kolom untuk encoding
        self.one_hot_cols_multi = [
            col for col in X_encoded.columns 
            if X_encoded[col].dtype == 'object' 
            and len(X_encoded[col].unique()) > 2 
            and col != 'Contract'
        ]
        
        self.one_hot_cols_binary = [
            col for col in X_encoded.columns 
            if X_encoded[col].dtype == 'object' 
            and len(X_encoded[col].unique()) == 2
        ]
        
        # One-hot encoding untuk multi-class
        X_encoded = pd.get_dummies(X_encoded, columns=self.one_hot_cols_multi, drop_first=False)
        
        # Label encoding untuk Contract
        if 'Contract' in X_encoded.columns:
            X_encoded['Contract'] = self.label_encoder.fit_transform(X_encoded['Contract'])
        
        # Binary encoding
        X_encoded = pd.get_dummies(X_encoded, columns=self.one_hot_cols_binary, drop_first=False)
        
        # Convert boolean to int
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'bool':
                X_encoded[col] = X_encoded[col].astype(int)
        
        print(f"✓ Features encoded: {X_encoded.shape[1]} total features")
        return X_encoded
    
    def scale_features(self, X):
        """
        Scale numerical features menggunakan MinMaxScaler
        
        Args:
            X (pd.DataFrame): Encoded features
            
        Returns:
            pd.DataFrame: Scaled features
        """
        X_scaled = X.copy()
        
        # Identifikasi numerical columns
        self.numeric_cols = X_scaled.select_dtypes(include=['int64', 'float64']).columns
        
        # Scale numerical columns
        X_scaled[self.numeric_cols] = self.scaler.fit_transform(X_scaled[self.numeric_cols])
        
        print(f"✓ {len(self.numeric_cols)} numerical features scaled")
        return X_scaled
    
    def encode_target(self, y):
        """
        Encode target variable: 'Yes'->1, 'No'->0
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            pd.Series: Encoded target
        """
        y_encoded = y.map({'No': 0, 'Yes': 1})
        print(f"✓ Target encoded: Class distribution {y_encoded.value_counts().to_dict()}")
        return y_encoded
    
    def balance_data(self, X, y, random_state=42):
        """
        Balance dataset menggunakan SMOTEENN
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        smote_enn = SMOTEENN(random_state=random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        print(f"✓ Data balanced with SMOTEENN:")
        print(f"  Before: {y.value_counts().to_dict()}")
        print(f"  After: {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, filepath, apply_smoteenn=True, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline:
        1. Load data
        2. Clean data
        3. Separate X and y
        4. Encode features and target
        5. Scale features
        6. Apply SMOTEENN (optional)
        7. Split train/test
        
        Args:
            filepath (str): Path ke data CSV
            apply_smoteenn (bool): Apakah apply SMOTEENN untuk balancing
            test_size (float): Test set proportion
            random_state (int): Random seed
            
        Returns:
            dict: Dictionary berisi X_train, X_test, y_train, y_test, feature_columns, dan preprocessor
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load dan clean data
        df = self.load_data(filepath)
        df = self.clean_data(df)
        
        # 2. Separate X dan y
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # 3. Encode target
        y = self.encode_target(y)
        
        # 4. Encode features
        X = self.encode_features(X)
        
        # 5. Scale features
        X = self.scale_features(X)
        
        # Save feature columns
        self.feature_columns = X.columns.tolist()
        
        # 6. Balance data (optional)
        if apply_smoteenn:
            X, y = self.balance_data(X, y, random_state)
        
        # 7. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Train/Test split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
        print("="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60 + "\n")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'preprocessor': self
        }
    
    def transform_new_data(self, X_new):
        """
        Transform data baru menggunakan preprocessor yang sudah di-fit
        Berguna untuk inference/prediction
        
        Args:
            X_new (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        X_transformed = X_new.copy()
        
        # Encode features (harus menggunakan encoder yang sama)
        X_transformed = self.encode_features(X_transformed)
        
        # Scale features
        X_transformed[self.numeric_cols] = self.scaler.transform(X_transformed[self.numeric_cols])
        
        # Ensure same feature columns
        # Add missing columns with 0
        for col in self.feature_columns:
            if col not in X_transformed.columns:
                X_transformed[col] = 0
        
        # Remove extra columns
        X_transformed = X_transformed[self.feature_columns]
        
        return X_transformed


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ChurnDataPreprocessor()
    
    # Update path sesuai dengan struktur Anda
    data_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    results = preprocessor.preprocess_pipeline(
        filepath=data_path,
        apply_smoteenn=True,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\n📊 Preprocessing Results:")
    print(f"   X_train shape: {results['X_train'].shape}")
    print(f"   X_test shape: {results['X_test'].shape}")
    print(f"   Number of features: {len(results['feature_columns'])}")
