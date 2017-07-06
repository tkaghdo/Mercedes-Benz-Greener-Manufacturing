import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def doLinearRegression(train_df, test_df):
    # is there any rows with empty values
    print('Number of rows with any null values: {0}'.format(len(train_df[pd.isnull(train_df).any(axis=1)])))

    features = train_df.columns.values.tolist()

    features.remove('ID')
    features.remove('y')

    # analyze all features
    '''
    for i in features:
        print('Unique values for column {0}'.format(str(i)))
        print(train_df[str(i)].value_counts())
        print('--------------')
    '''

    # *** convert discrete values (columns X0 to X9) to categorical variables

    discrete_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
    for i in discrete_features:
        temp_df = pd.get_dummies(train_df[i], prefix=i)
        train_df = pd.concat([train_df, temp_df], axis=1)
        train_df = train_df.drop(i, axis=1)

    features = train_df.columns.values.tolist()
    features.remove('ID')
    features.remove('y')

    lr = LinearRegression()
    lr.fit(train_df[features], train_df['y'])

    print("*** Accuracy (Score) ***")
    print(lr.score(train_df[features], train_df['y']))

    # prepare the test data
    for i in discrete_features:
        temp_df = pd.get_dummies(test_df[i], prefix=i)
        test_df = pd.concat([test_df, temp_df], axis=1)
        test_df = test_df.drop(i, axis=1)

    test_dfxx = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(test_dfxx.head())
    features = test_dfxx.columns.values.tolist()
    features.remove('ID')
    features.remove('y')
    print(features)
    print(train_df.shape)
    print(test_dfxx.shape)
    print(test_df.shape)
    print(len(features))
    y_predictions = lr.predict(test_dfxx[features])
    print(len(y_predictions))
    test_dfxx['y_predict'] = y_predictions
    df = test_dfxx[['ID', 'y_predict']]
    print(df.head(100))

    df.to_csv('submission.csv', index=False)

def doRandomForest(train_df, test_df):

    discrete_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

    for name in discrete_features:
        col = pd.Categorical.from_array(train_df[name])
        train_df[name] = col.codes
    print(train_df.head())

    for name in discrete_features:
        col = pd.Categorical.from_array(test_df[name])
        test_df[name] = col.codes
    print(test_df.head())

    features = train_df.columns.values.tolist()
    features.remove('ID')
    features.remove('y')
    print(features)


    clf = RandomForestRegressor(n_estimators=1000, random_state=1, oob_score=True, bootstrap=True)
    clf.fit(train_df[features], train_df['y'])

    predictions = clf.predict(test_df[features])
    #predictions = clf.predict(train_df[features])
    #print(r2_score(train_df['y'], predictions))
    #print(predictions[0:10])


    test_df['y'] = predictions
    submission_df = test_df[['ID', 'y']]
    print(len(submission_df))
    print(submission_df.head())
    submission_df.to_csv('rf_submission.csv', index=False)

def doGradientBoostingRegression(train_df, test_df):
    discrete_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

    for name in discrete_features:
        col = pd.Categorical.from_array(train_df[name])
        train_df[name] = col.codes
    print(train_df.head())

    for name in discrete_features:
        col = pd.Categorical.from_array(test_df[name])
        test_df[name] = col.codes
    print(test_df.head())

    features = train_df.columns.values.tolist()
    features.remove('ID')
    features.remove('y')
    print(features)
    params = {'n_estimators': 1000, 'max_depth': 17,'min_samples_split': 2,
              'learning_rate': 0.05, 'loss': 'ls', 'random_state': 1}
    clf = GradientBoostingRegressor(**params)

    clf.fit(train_df[features], train_df['y'])
    #predictions = clf.predict(train_df[features])
    predictions = clf.predict(test_df[features])
    #mse = mean_squared_error(train_df['y'], predictions)
    #print("MSE: %.4f" % mse)


    test_df['y'] = predictions
    submission_df = test_df[['ID', 'y']]
    print(len(submission_df))
    print(submission_df.head())
    submission_df.to_csv('gbr_submission.csv', index=False)

def doRandomForest2(train_df, test_df):
    # is there any rows with empty values
    print('Number of rows with any null values: {0}'.format(len(train_df[pd.isnull(train_df).any(axis=1)])))

    features = train_df.columns.values.tolist()

    features.remove('ID')
    features.remove('y')

    # *** convert discrete values (columns X0 to X9) to categorical variables

    discrete_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
    for i in discrete_features:
        temp_df = pd.get_dummies(train_df[i], prefix=i)
        train_df = pd.concat([train_df, temp_df], axis=1)
        train_df = train_df.drop(i, axis=1)

    features = train_df.columns.values.tolist()
    features.remove('ID')
    features.remove('y')

    lr = RandomForestRegressor(n_estimators=500, random_state=1)
    lr.fit(train_df[features], train_df['y'])

    print("*** Accuracy (Score) ***")
    print(lr.score(train_df[features], train_df['y']))

    # prepare the test data
    for i in discrete_features:
        temp_df = pd.get_dummies(test_df[i], prefix=i)
        test_df = pd.concat([test_df, temp_df], axis=1)
        test_df = test_df.drop(i, axis=1)

    test_dfxx = test_df.reindex(columns=train_df.columns, fill_value=0)

    print(test_dfxx.head())
    features = test_dfxx.columns.values.tolist()
    features.remove('ID')
    features.remove('y')
    print(features)
    print(train_df.shape)
    print(test_dfxx.shape)
    print(test_df.shape)
    print(len(features))
    y_predictions = lr.predict(test_dfxx[features])
    print(len(y_predictions))
    test_dfxx['y_predict'] = y_predictions
    df = test_dfxx[['ID', 'y_predict']]
    print(df.head(100))

    df.to_csv('submission.csv', index=False)


def main():
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')

        #train_df = pd.read_csv('data/train_hour.csv')
        #test_df = pd.read_csv('data/test_hour.csv')
    except Exception as e:
        print(e)
        sys.exit(1)

    # doLinearRegression(train_df, test_df)

    #doRandomForest(train_df, test_df)

    doRandomForest2(train_df, test_df)

    #doGradientBoostingRegression(train_df, test_df)

if __name__ == '__main__':
    sys.exit(0 if main() else 1)