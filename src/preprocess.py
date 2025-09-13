import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    yes_no = ['Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yes_no:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1,'No':0})
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace({'No phone service':'No'})
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    obj_cols = [c for c in obj_cols if c != 'churn']
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].fillna(False)
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].fillna('Unknown')
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    df = df.fillna(0)
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    obj_remaining = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_remaining:
        try:
            df[c] = pd.to_numeric(df[c], errors='ignore')
        except Exception:
            pass
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(int)
    return df

if __name__ == '__main__':
    df = load_data('data/processed.csv')
    df = preprocess(df)
    df.to_csv('data/cleaned.csv', index=False)
