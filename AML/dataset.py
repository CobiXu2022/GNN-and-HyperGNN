
import os
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Optional, Callable

from torch_geometric.data import (
    Data,
    InMemoryDataset
)

pd.set_option('display.max_columns', None)


class AMLtoGraph(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, use_rf_features: bool = False):
        self.edge_window_size = edge_window_size
        self.use_rf_features = use_rf_features #newly added
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> str:
        return 'HI-Small_Trans.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df


    def preprocess(self, df):
        df = self.df_label_encoder(df,['Payment Format', 'Payment Currency', 'Receiving Currency'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp']-df['Timestamp'].min())/(df['Timestamp'].max()-df['Timestamp'].min())

        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
        df = df.sort_values(by=['Account'])
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']]
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
        currency_ls = sorted(df['Receiving Currency'].unique())

        print(f"After step preprocess: {len(df)} rows")  

        return df, receiving_df, paying_df, currency_ls

    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank']]
        rdf = df[['Account.1', 'To Bank']]
        suspicious = df[df['Is Laundering']==1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']]
        s2 = s2.rename({'Account.1': 'Account'}, axis=1)
        suspicious = pd.concat([s1, s2], join='outer')
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)
        df = pd.concat([ldf, rdf], join='outer')
        df = df.drop_duplicates()

        df['Is Laundering'] = 0
        df.set_index('Account', inplace=True)
        df.update(suspicious.set_index('Account'))
        df = df.reset_index()

        print(f"After step get all accounts: {len(df)} rows")  
        return df
    
    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df['Payment Currency'] == i]
            accounts['avg paid '+str(i)] = temp['Amount Paid'].groupby(temp['Account']).transform('mean')
        return accounts

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df['Receiving Currency'] == i]
            accounts['avg received '+str(i)] = temp['Amount Received'].groupby(temp['Account']).transform('mean')
        accounts = accounts.fillna(0)
        return accounts

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Account'], accounts['ID']))
        df['From'] = df['Account'].map(mapping_dict)
        df['To'] = df['Account.1'].map(mapping_dict)
        df = df.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)

        edge_index = torch.stack([torch.from_numpy(df['From'].values), torch.from_numpy(df['To'].values)], dim=0)

        df = df.drop(['Is Laundering', 'From', 'To'], axis=1)

        edge_attr = torch.from_numpy(df.values).to(torch.float)

        print(f"After step get edge df: {len(df)} rows") 
        return edge_attr, edge_index

    def get_node_attr(self, currency_ls, paying_df,receiving_df, accounts):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df['Is Laundering'].values).to(torch.float)
        print("Columns in node_df before dropping:", node_df.columns)
        print("First few rows of node_df:\n", node_df.head())

        if self.use_rf_features:
            rf_features = self.generate_rf_features(node_df.copy())
            node_df = pd.concat([
                node_df.drop(['Account', 'Is Laundering'], axis=1),
                pd.DataFrame(rf_features, columns=[f'rf_{i}' for i in range(rf_features.shape[1])])
            ], axis=1)
        else:
            node_df = node_df.drop(['Account', 'Is Laundering'], axis=1)

        node_df = self.df_label_encoder(node_df,['Bank'])
        node_df = torch.from_numpy(node_df.values).to(torch.float)

        return node_df, node_label

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)


        print(df.columns)
        #accounts, df = self.resample_data(accounts, df) #resample

        node_attr, node_label = self.get_node_attr(currency_ls, paying_df,receiving_df, accounts)
        edge_attr, edge_index = self.get_edge_df(accounts, df)
        #self._data = Data(x=node_attr, edge_index=edge_index, y=node_label, edge_attr=edge_attr) #resampled

        data = Data(x=node_attr,
                    edge_index=edge_index,
                    y=node_label,
                    edge_attr=edge_attr
                    )
        
        data_list = [data] 
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"After step process: {len(df)} rows") 
        

    def calculate_class_weights(self):

        if not hasattr(self, '_data'):
            self._data = torch.load(self.processed_paths[0])[0]
    
        labels = self._data.y
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 0).sum().item()
        total = len(labels)
    
        print("\n=== classes statistics ===")
        print(f"total: {total}")
        print(f"suspicious (positive): {pos_count} ({pos_count/total*100:.2f}%)")
        print(f"licit (negative): {neg_count} ({neg_count/total*100:.2f}%)")
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32)
    
        print(f"\n recommended pos_weight: {pos_weight.item():.2f}")
        return pos_weight

    def generate_rf_features(self, accounts_df):
        print("generate random forest features")
        X = accounts_df.drop(['Account', 'Is Laundering'], axis=1)
        y = accounts_df['Is Laundering']
        print(f"\noriginal distribution - positive: {y.sum()}, negative: {len(y)-y.sum()}")

        #sampler = RandomOverSampler(sampling_strategy=0.1, random_state=42) #duplicate
        #X_res, y_res = sampler.fit_resample(X, y)

        smote = SMOTE(sampling_strategy=0.1, random_state=42) #smote
        X_res, y_res = smote.fit_resample(X, y)
        print(f"resampled - positive: {y_res.sum()}, negative: {len(y_res)-y_res.sum()}")
        rf = RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_res, y_res)
        leaves = rf.apply(X)         
        probs = rf.predict_proba(X)[:, 1:]  
        
        return np.hstack([leaves, probs])  

    def resample_data(self, accounts, df):
 
        print("resampling activated")
        
        X = accounts.drop(['Is Laundering'], axis=1)
        y = accounts['Is Laundering']
        print(f"Before resampling: positive count = {y.sum()}, negative count = {len(y) - y.sum()}")
        print("df columns:", df.columns)

        sampler = RandomOverSampler(sampling_strategy=0.05, random_state=42) #duplicate
        X_res, y_res = sampler.fit_resample(X, y)


        
        #smote = SMOTE(sampling_strategy=0.1, random_state=42) #smote
        #X_res, y_res = smote.fit_resample(X, y)


        resampled_accounts = pd.DataFrame(X_res, columns=X.columns)
        resampled_accounts['Is Laundering'] = y_res

        
        pos_count = (resampled_accounts['Is Laundering'] == 1).sum()
        neg_count = (resampled_accounts['Is Laundering'] == 0).sum()  

        print(resampled_accounts.columns)   

        df = df[df['Account'].isin(resampled_accounts['Account'])]
        print(f"After resampling: positive count = {pos_count}, negative count = {neg_count}")

        return resampled_accounts, df
    

