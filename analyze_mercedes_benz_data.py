import sys
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
    except Exception as e:
        print(e)
        sys.exit(1)

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

    # create binary values for each of the 'X0' values. the columns will be prefixed by 'X0'
    #X0_df = pd.get_dummies(train_df['X0'], prefix='X0')
    #train_df = pd.concat([train_df, X0_df], axis=1)
    #train_df = train_df.drop('X0', axis=1)
    #print(train_df.head())

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




if __name__ == '__main__':
    sys.exit(0 if main() else 1)