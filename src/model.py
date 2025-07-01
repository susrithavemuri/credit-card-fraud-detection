def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
    grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_ensemble_model(X_train, y_train):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression
    model = BaggingClassifier(estimator=LogisticRegression(), n_estimators=10)
    model.fit(X_train, y_train)
    return model
