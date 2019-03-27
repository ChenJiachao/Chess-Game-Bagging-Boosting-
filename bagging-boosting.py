import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


chess = pd.read_csv('./krkopt_data.txt', header=None)
chess.columns = ['wkf', 'wkr', 'wrf', 'wrr', 'bkf', 'bkr', 'class']
chess = shuffle(chess, random_state = 0)
chess.head(10)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

d_wkf = pd.get_dummies(chess['wkf'], prefix='wkf')
d_wkr = pd.get_dummies(chess['wkr'], prefix='wkr')
d_wrf = pd.get_dummies(chess['wrf'], prefix='wrf', drop_first=True)
d_wrr = pd.get_dummies(chess['wrr'], prefix='wrr', drop_first=True)
d_bkf = pd.get_dummies(chess['bkf'], prefix='bkf', drop_first=True)
d_bkr = pd.get_dummies(chess['bkr'], prefix='bkr', drop_first=True)
chess_new = pd.concat([d_wkf, d_wkr, d_wrf, d_wrr, d_bkf, d_bkr, chess['class']], axis=1)
X = chess_new.iloc[:, :-1]
y = chess_new['class']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.head(10)


from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

n_max = 12
Err_Train = np.zeros(n_max)
Err_Test = np.zeros(n_max)
indices = 2**np.array(range(0,n_max))

# implement Boosting with Decision Tree
for i in range(12):
    dtc = DecisionTreeClassifier(criterion = 'gini')
    clf = BaggingClassifier(base_estimator = dtc,n_estimators =indices[i])
    clf.fit(X_train, y_train)
    
    y_pred_tst = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    error_tst = 1 - accuracy_score(y_pred_tst,y_test )
    error_train = 1 - accuracy_score(y_pred_train,y_train )
    
    Err_Train[i] = error_train
    Err_Test[i] = error_tst
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()



# Implement AdaBoost with different max-depth of decision trees
from sklearn.ensemble import AdaBoostClassifier
n_max = 12

Depths = [10, 20, 50, 100]


for depth in Depths:
    Err_Train = np.zeros(n_max)
    Err_Test = np.zeros(n_max)
    
    for index in range(n_max):
        dtc = DecisionTreeClassifier(criterion = 'gini', max_depth=depth)
        clf = AdaBoostClassifier(dtc, n_estimators=indices[index])
        clf.fit(X_train, y_train)
        Err_Train[index] = 1 - clf.score(X_train, y_train)
        Err_Test[index] = 1 - clf.score(X_test, y_test)
    print("Detph: " + str(depth))
    plt.figure()
    plt.semilogx(indices,Err_Train, label = "training")
    plt.semilogx(indices,Err_Test, label = "testing")
    plt.show()
    


# Implement random forest from DCT in bagging into adboost
from sklearn.ensemble import AdaBoostClassifier
n_max = 6
Err_Train = np.zeros(n_max)
Err_Test = np.zeros(n_max)
indices = 2**np.array(range(0,n_max))

# when depth = 20, it achieves highest testing accuracy
for i in range(n_max):
    dtc = DecisionTreeClassifier(criterion = 'gini', max_depth= 20)
    bc = BaggingClassifier(base_estimator = dtc ,n_estimators =10)
    clf = AdaBoostClassifier(base_estimator = bc,n_estimators =indices[i] )
    
    clf.fit(X_train, y_train)
    y_pred_tst = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    error_tst = 1 - accuracy_score(y_pred_tst,y_test )
    error_train = 1 - accuracy_score(y_pred_train,y_train )
    Err_Train[i] += error_train
    Err_Test[i] += error_tst

plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()
