
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
iris = load_iris()
MODEL1 = GaussianNB()
MODEL2 = MultinomialNB()
MODEL3 = BernoulliNB()
x = iris.data
y = iris.target
MODEL1 = MODEL1.fit(x, y)
MODEL2 = MODEL2.fit(x, y)
MODEL3 = MODEL3.fit(x, y)
out1 = MODEL1.predict([[3.5,3.5,3.5,3.5]])
out2 = MODEL2.predict([[3.5,3.5,3.5,3.5]])
out3 = MODEL3.predict([[3.5,3.5,3.5,3.5]])
print("Gaussian output = ",out1)
print("Multinomial output = ",out2)
print("Bernoulli output = ",out3)