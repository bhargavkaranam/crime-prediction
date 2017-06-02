import pandas as pd

from sklearn.ensemble import RandomForestClassifier

def load_training_data():
    # loading training data
    data = pd.read_csv('train.csv',nrows=10000)
    #data = data.head(20000)

    X_tr = data
    y_tr = data["Category"]
    X_tr = X_tr.drop("Category", axis=1)
    
    # drop columns not in test data for now
    X_tr = X_tr.drop("Descript", axis=1) 
    X_tr = X_tr.drop("Resolution", axis=1)
        
    return X_tr, y_tr
    
def load_testing_data():    
    # loading test data
    data = pd.read_csv('test.csv', nrows=10000, index_col="Id")
    X_test = data
    return X_test

def process_X(X_in):
    X = X_in
    
    # build categorical features
    hours = pd.get_dummies(X.Dates.map(lambda x: pd.to_datetime(x).hour), prefix="hour")
    months = pd.get_dummies(X.Dates.map(lambda x: pd.to_datetime(x).month), prefix="month")
    years = pd.get_dummies(X.Dates.map(lambda x: pd.to_datetime(x).year), prefix="year")
    district = pd.get_dummies(X["PdDistrict"])
    day_of_week = pd.get_dummies(X["DayOfWeek"])
    
    # string them all together
    X = pd.concat([X, hours, months, years, district, day_of_week], axis=1)
    
    X = X.drop("PdDistrict", axis=1)
    X = X.drop("Address", axis=1)
    X = X.drop("Dates", axis=1)
    X = X.drop("DayOfWeek", axis=1)
    
    return X

def write_results(output):
    # save the result
    output.to_csv("results.csv", index_label="Id")
    
def main():
    print('Loading data...')
    # loading training data
    X_tr, y_tr = load_training_data()
    
    # loading test data
    X_test = load_testing_data()
    
    # process X
    print("Processing datasets...")
    X_tr = process_X(X_tr)
    X_test = process_X(X_test)
        
    print("Fitting classifier...")   
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_tr, y_tr)
    
    print('Predicting...')
    y_test = pd.DataFrame(clf.predict_proba(X_test), index=X_test.index, columns=clf.classes_)
    
    print("Writing results to results.csv...")
    # save the result
    write_results(y_test)


if __name__ == '__main__':
    main()