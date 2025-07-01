def scale_features(df):
    from sklearn.preprocessing import StandardScaler
    df['scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    return df.drop(['Time', 'Amount'], axis=1)