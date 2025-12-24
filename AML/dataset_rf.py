import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
class AMLtoRF:
    def __init__(self, root: str, transform=None, pre_transform=None):
        self.root = root
        self.raw_dir = os.path.join(root, 'raw')
        self.processed_dir = os.path.join(root, 'processed')
        self.processed_data = None
        self.label_encoders = {}
        self.scaler = None
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._load_data()

    @property
    def raw_file_names(self) -> str:
        return ['HI-Small_Trans.csv']

    @property
    def processed_file_names(self) -> str:
        return ['rf_features.npz', 'rf_labels.npy', 'preprocessor.pkl']

    def _load_data(self):
        processed_paths = [
            os.path.join(self.processed_dir, fname) 
            for fname in self.processed_file_names
        ]
        if all(os.path.exists(path) for path in processed_paths):
            self._load_processed_data()
        else:
            self._process_data()

    def _load_processed_data(self):
        try:
            self.processed_data = (
                np.load(os.path.join(self.processed_dir, 'rf_features.npz'))['data'],
                np.load(os.path.join(self.processed_dir, 'rf_labels.npy'))
            )
            preprocessor = joblib.load(os.path.join(self.processed_dir, 'preprocessor.pkl'))
            self.label_encoders = preprocessor['label_encoders']
            self.scaler = preprocessor['scaler']
        except Exception as e:
            print(f"Failed to load processed data: {e}")
            self._process_data()

    def _process_data(self):
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]))
        X, y = self._prepare_features(df)

        print(f"raw data: {len(df)}")
        print("sample distribution:")
        print(df['Is Laundering'].value_counts())
        
        np.savez_compressed(os.path.join(self.processed_dir, 'rf_features.npz'), data=X)
        np.save(os.path.join(self.processed_dir, 'rf_labels.npy'), y)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }, os.path.join(self.processed_dir, 'preprocessor.pkl'))
        
        self.processed_data = (X, y)

    def _basic_preprocessing(self, df):
        df.fillna({
            'Amount Received': 0,
            'Amount Paid': 0,
            'Payment Format': 'unknown',
            'Payment Currency': 'unknown',
            'Receiving Currency': 'unknown'
        }, inplace=True)
        
        df = df[(df['Amount Received'] >= 0) & (df['Amount Paid'] >= 0)]
        return df

    def _feature_engineering(self, df):

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        
        df['amount_discrepancy'] = (df['Amount Received'] - df['Amount Paid']).abs()
        df['currency_mismatch'] = (df['Receiving Currency'] != df['Payment Currency']).astype(int)
        
        df['tx_frequency'] = df.groupby('Account')['Account'].transform('count')
        return df

    def _prepare_features(self, df):
        df = self._basic_preprocessing(df)

        df = self._feature_engineering(df)
    
        cat_cols = ['From Bank', 'To Bank', 'Payment Format', 
                   'Payment Currency', 'Receiving Currency']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        feature_cols = [c for c in df.columns if c not in [
            'Timestamp', 'Account', 'Account.1', 'Is Laundering'
        ]]
        X = df[feature_cols]
        y = df['Is Laundering'].values

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.scaler = StandardScaler()
        X[num_cols] = self.scaler.fit_transform(X[num_cols])
        
        return X.values, y

    def get_train_test_split(self, test_size=0.3, random_state=42):
        if self.processed_data is None:
            self._process_data()
            
        X, y = self.processed_data

        print(f"Original X shape: {X.shape}, y shape: {y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print(f"Training X shape: {X_train.shape}, y shape: {y_train.shape}")
        X_train_resampled, y_train_resampled = SMOTE(random_state=None).fit_resample(X_train, y_train)
        print(f"Resampled Training X shape: {X_train_resampled.shape}, y shape: {y_train_resampled.shape}")
        #X, y = SMOTE(random_state=None).fit_resample(X, y)
        return X_train_resampled, X_test, y_train_resampled, y_test
        #return train_test_split(X, y, test_size=test_size, random_state=None, stratify=y)