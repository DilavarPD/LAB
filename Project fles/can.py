maxElement = np.amax(preds)
preds = np.argmax(preds, axis=1)
if maxElement >= 1:
    if preds == 0:
        preds = "ID1"
    elif preds == 1:
        preds = "ID2"

else:
    preds = "false"