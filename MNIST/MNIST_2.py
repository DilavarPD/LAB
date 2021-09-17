
from tensorflow import keras
import matplotlib.pyplot as plt

MNISTDB = keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = MNISTDB.load_data()
print("Size of Xtrain:",xtrain.shape)
print("Size of Ytrain:",ytrain.shape)
print("Size of Xtest:",xtest.shape)
print("Size of Ytest:",ytest.shape)
plt.imshow(xtrain[11039],cmap='binary')


xtrain = xtrain.reshape((60000,28,28,1))
xtest = xtest.reshape((10000,28,28,1))
print("Size of Xtrain:",xtrain.shape)
print("Size of Xtest:",xtest.shape)


convnet=keras.models.Sequential()
convnet.add(keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=xtrain.shape[1:]))
convnet.add(keras.layers.Conv2D(64,(3,3),activation="relu"))
convnet.add(keras.layers.BatchNormalization())
convnet.add(keras.layers.MaxPool2D(2,2))
convnet.add(keras.layers.Dropout(0.25))
convnet.add(keras.layers.Flatten())
convnet.add(keras.layers.Dense(128,activation="relu"))
convnet.add(keras.layers.Dropout(0.25))
convnet.add(keras.layers.Dense(10,activation="softmax"))
convnet.summary()


convnet.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics="accuracy")
es=keras.callbacks.EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
cp=keras.callbacks.ModelCheckpoint("/content/SaranyaConvNet.h5",monitor='val_loss')
convnet.fit(xtrain,ytrain,epochs=3,batch_size=15,callbacks=[es,cp])
testloss,testaccuracy=convnet.evaluate(xtest,ytest)
print("Test Loss =",testloss)
print("Test Accuracy =",testaccuracy)
plt.show()