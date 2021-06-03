from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=2)
ML = SVC()
ML = ML.fit(x_train, y_train)
out = ML.predict(x_test)
score = accuracy_score(out, y_test)
print("accuracy = ", score*100)