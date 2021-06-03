import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.impute import SimpleImputer
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train=train[['Age',  'Flight Distance', 'Inflight wifi service', 'Ease of Online booking', 'Food and drink',
         'Seat comfort', 'Leg room service', 'Cleanliness','Departure Delay in Minutes', 'Arrival Delay in Minutes']]
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
imputer.fit(x_train)
x_train = imputer.transform(x_train)
y_train=train['satisfaction']

x_test=test[['Age',  'Flight Distance', 'Inflight wifi service', 'Ease of Online booking', 'Food and drink',
         'Seat comfort', 'Leg room service', 'Cleanliness','Departure Delay in Minutes', 'Arrival Delay in Minutes']]
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')
imputer.fit(x_test)
x_test = imputer.transform(x_test)
y_test = test['satisfaction']

MODEL1 = GaussianNB()
MODEL2 = MultinomialNB()
MODEL3 = BernoulliNB()
MODEL1 =MODEL1.fit(x_train,y_train)
MODEL2 =MODEL2.fit(x_train,y_train)
MODEL3 =MODEL3.fit(x_train,y_train)
oup1 = MODEL1.predict(x_test)
oup2 = MODEL2.predict(x_test)
oup3 = MODEL3.predict(x_test)
acc1 = accuracy_score(oup1,y_test)
acc2 = accuracy_score(oup2,y_test)
acc3 = accuracy_score(oup3,y_test)
print("accuracy predicted by Gaussian : ",acc1*100)
print("accuracy predicted by Multinomial : ",acc2*100)
print("accuracy predicted by Bernoullie : ",acc3*100)