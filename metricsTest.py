import pandas as pd
import json
import csv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from sklearn import metrics
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import keras as k
import matplotlib.pyplot as plt

image_size = 224
model = load_model('food11-model-NODILATED-softmax.h5') #food11-
trainFile = 'food-11/train.json'# 'newDATAbiggerDB-20/train.json' #
valFile = 'food-11/val.json'#'newDATAbiggerDB-20/val.json' #
testFile = 'food-11/eval.json'# 'newDATAbiggerDB-20/test.json'#
dataset = 'ingredients20.csv'
categories = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
batch_size = 64

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = categories #classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def openJson(file):
    with open(file) as File:
        dict = json.load(File)
    return dict

def getIngredientsHeader():
    mylist = []
    with open(dataset, mode='r') as infile:
        reader = csv.reader(infile)
        ingrsSet = {rows[0] for rows in reader}

    for i in ingrsSet: # create the list with all the ingredients from the .csv file and sort it. Also create a same-size binary list
        mylist.append(i)
    mylist.sort()
    return mylist

def test_generator():
    with open(testFile) as testfile:
        dict_test = json.load(testfile)

    test = pd.DataFrame.from_dict(dict_test, orient='index')
    test.reset_index(level=0, inplace=True)
    test.columns = ['Id', 'Ingredients', 'Binary']
    nb_samples = len(test)
    while True:
        for start in range(0, nb_samples, batch_size):
            test_image =[]
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end): #newDATAbiggerDB-20/test
                img = image.load_img('food-11/evaluation/' + test['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = img / 255
                test_image.append(img)

                # y_batch.append(test['Binary'][i])

            yield (np.array(test_image))#, np.array(y_batch))

def train_generator():
    with open(trainFile) as trainfile:
        dict_test = json.load(trainfile)

    test = pd.DataFrame.from_dict(dict_test, orient='index')
    test.reset_index(level=0, inplace=True)
    test.columns = ['Id', 'Ingredients', 'Binary']
    nb_samples = len(test)
    while True:
        for start in range(0, nb_samples, batch_size):
            test_image =[]
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end): #newDATAbiggerDB-20/test
                img = image.load_img('food-11/training/' + test['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = img / 255
                test_image.append(img)

                # y_batch.append(test['Binary'][i])

            yield (np.array(test_image))#, np.array(y_batch))

def val_generator():
    with open(valFile) as valfile:
        dict_test = json.load(valfile)

    test = pd.DataFrame.from_dict(dict_test, orient='index')
    test.reset_index(level=0, inplace=True)
    test.columns = ['Id', 'Ingredients', 'Binary']
    nb_samples = len(test)
    while True:
        for start in range(0, nb_samples, batch_size):
            test_image =[]
            y_batch = []
            end = min(start + batch_size, nb_samples)
            for i in range(start, end):
                img = image.load_img('food-11/validation/' + test['Id'][i], target_size=(image_size, image_size, 3))
                img = image.img_to_array(img)
                img = img / 255
                test_image.append(img)

                # y_batch.append(test['Binary'][i])

            yield (np.array(test_image))#, np.array(y_batch))

def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        # print('set_true: {0}'.format(set_true))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            # print (y_pred[i])
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n)-1-j])
            # print ('top_n:::',top_n)
            # print('set_finalpred: {0}'.format(set_finalpred))
            inter = set_true.intersection(set_finalpred)
            tmp_a = len(set_true.intersection(set_finalpred))/\
                    float( len(set_true.union(set_finalpred)) )
        # print('tmp_a: {0}'.format(tmp_a))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def precision(y_true, y_pred):
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n) - 1 - j])
                tmp_a = len(set_true.intersection(set_finalpred)) / \
                        float(len(set_finalpred))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def recall(y_true, y_pred):
    acc_list = []
    set_finalpred = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            top_n = np.argsort(y_pred[i])
            for j in range(len(set_true)):
                set_finalpred.append(top_n[len(top_n) - 1 - j])
                tmp_a = len(set_true.intersection(set_finalpred)) / \
                        float(len(set_true))
        del set_finalpred[:]
        acc_list.append(tmp_a)
    return np.mean(acc_list)

noIngr = []
nb_test_samples = len(openJson(testFile))
test_true = openJson(testFile)
values = (test_true.values())
# print ('Values:',values)
keys = test_true.keys()
y_true = np.array([item[1] for item in values])
# print (y_true)
predictions = []
truth = []
expected = []
np.set_printoptions(precision=2)

def averageIngredientsNumber():
    val_true = openJson(valFile)
    val_values = (val_true.values())

    train_true = openJson(trainFile)
    train_values = (train_true.values())

    y_valkeys = np.array([item[0] for item in val_values])
    y_trainkeys = np.array([item[0] for item in train_values])
    for i in y_trainkeys:
        noIngr.append(len(i))
    for j in y_valkeys:
        noIngr.append(len(j))
    averageIngrNo = int(np.mean(noIngr)+1) # I add one because the average may be very small and the prediction
    # print (averageIngrNo)
    return averageIngrNo

def showTagging(values, keys, pred, path, folder, image_path):
    labels = getIngredientsHeader()#categories #
    y_truelabels = np.array([item[0] for item in values])
    y_true = np.array([item[1] for item in values])

    for i in range(len(pred)):
        top_n = np.argsort(pred[i])
        set_true = set(np.where(y_true[i])[0])
        for j in range(len(set_true)):
            predictions.append(top_n[len(top_n)-1-j])
        truth.append(y_truelabels[i])
        # truth = [t.replace('u', '') for t in truth]
        # print ('predictions:',predictions)
        for p in predictions:
            expected.append(labels[p])
        # print ('expected:',expected)

        test = os.listdir(folder)  # newDATAbiggerDB-20/test
        for jpg in test:
            if (jpg == keys[i]):
                img = Image.open(image_path + jpg) #newDATAbiggerDB-20/test/
                # img = img.resize((224, 224), Image.ANTIALIAS)
                fontsize = 15  # starting font size
                # portion of image width you want text width to be
                img_fraction = 0.50
                font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)
                while font.getsize('true labels:' + str(truth) + '\npredictions:' + str(expected))[0] < img_fraction * img.size[0]:
                    # iterate until the text size is just larger than the criteria
                    fontsize += 1
                    font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)

                # optionally de-increment to be sure it is less than criteria
                fontsize -= 1
                font = ImageFont.truetype("alef/Alef-Bold.ttf", fontsize)
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), 'true labels:' + str(truth) + '\npredictions:' + str(expected), font=font, fill="blue")
                img.save(path+jpg)

        del predictions[:]
        del truth[:]
        del expected[:]

with tf.device('/gpu:0'):
    # averageIngrNo = 1#averageIngredientsNumber()
    pred = model.predict_generator(test_generator(), nb_test_samples // batch_size + 1)
    # print (type(pred))
    y_pred = []
    true = []
    for i in range(len(pred)):
        predmax = np.argmax(pred[i])
        truemax = np.argmax(y_true[i])
        y_pred.append(predmax)
        true.append(truemax)

    print ('Metrics: confusion_matrix:\n {0}'.format(metrics.confusion_matrix(true,y_pred)))
    print ('Metrics: accuracy_score: {0}'.format(metrics.accuracy_score(true, y_pred)))
    # print ('Metrics: CategoricalAccuracy: {0}'.format(k.metrics.top_k_categorical_accuracy(true, y_pred,1)))
    print ('Metrics: cohen_kappa_score: {0}'.format(metrics.cohen_kappa_score(true,y_pred)))
    # print (pred)

    # print('Subset accuracy: {0}'.format(metrics.accuracy_score(y_true, pred, normalize=True, sample_weight=None)))
    print('\nHamming score: {0}'.format(hamming_score(y_true, pred))) #label-based accuracy

    log = metrics.log_loss(y_true, pred)
    print ('Metrics: LogLoss: {0}'.format(log))

    r = recall(y_true, pred)
    p = precision(y_true, pred)
    print ('Metrics: recall: {0}'.format(r))
    print ('Metrics: precision: {0}'.format(p))
    f1 = 2*(p*r)/(p+r)
    print ('Metrics: f1score: {0}'.format(f1))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(true, y_pred, classes=categories,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(true, y_pred, classes=categories, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    #path:tagging-food11_evaluation/
    #folder: food-11/evaluation
    #img_path:food-11/evaluation/
    # showTagging(values, keys, pred, 'tagging/', """newDATAbiggerDB-20/test""", 'newDATAbiggerDB-20/test/')

    # # for training images
    # nb_train_samples = len(openJson(trainFile))
    # train_true = openJson(trainFile)
    # trainvalues = (train_true.values())
    # trainkeys = train_true.keys()
    #
    # trainpred = model.predict_generator(train_generator(), nb_train_samples // batch_size + 1)
    # showTagging(trainvalues, trainkeys, trainpred,'tagging/tagging-food11_training/', """food-11/training""", 'food-11/training/')
    #
    # # for validation images
    # nb_val_samples = len(openJson(valFile))
    # val_true = openJson(valFile)
    # valvalues = (val_true.values())
    # valkeys = val_true.keys()
    #
    # valpred = model.predict_generator(val_generator(), nb_val_samples // batch_size + 1)
    # showTagging(valvalues, valkeys, valpred, 'tagging/tagging-food11_validation/', """food-11/validation""", 'food-11/validation/')