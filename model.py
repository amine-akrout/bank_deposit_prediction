import pandas as pd
from pprint import pprint
import shutil
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


import mlflow

df = pd.read_csv('./data/bank.csv')
df.head()

df.info()

y = df['deposit']
X = df.drop(columns='deposit', axis=1)
cat_features = list(set(X.columns) - set(X._get_numeric_data().columns))
X[cat_features] = X[cat_features].astype(str)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1234)


mlflow.catboost

def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

with mlflow.start_run(run_name="run") as run:

    params = {'loss_function':'Logloss', 
          'eval_metric':'AUC',
          'verbose': 200,
          'random_seed': 1234
         }
    cb = CatBoostClassifier(**params)
    cb.fit(X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_valid, y_valid), 
            use_best_model=True, 
            plot=False
            )
    train_mertics = cb.get_best_score()['learn']
    pprint(train_mertics)
    validation_mertics = cb.get_best_score()['validation']
    pprint(validation_mertics)
    # mlflow.log_metric('train_AUC', train_mertics['AUC'])
    mlflow.log_metric('train_loglooss', train_mertics['Logloss'])
    mlflow.log_metric('val_AUC', validation_mertics['AUC'])
    mlflow.log_metric('val_logloss', validation_mertics['Logloss'])
    mlflow.log_params(params)

    # Log catboost model
    mlflow.sklearn.log_model(cb, "catboost-model")

    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    pprint(params)
    pprint(metrics)
    pprint(tags)
    pprint(artifacts)
    pprint(run.info.run_id)

    model_path = 'mlruns/0/{}/artifacts/model'.format(run.info.run_id)
    pprint(model_path)

    # shutil.copytree(model_path, './model', dirs_exist_ok=True)

    