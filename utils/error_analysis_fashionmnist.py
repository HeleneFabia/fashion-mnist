import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot


def fn_confusion_matrix(y, preds):
    """
    Show confusion matrix of predicted vs. true labels.
    """

    y_arr = y.numpy()
    preds = preds.to('cpu')
    preds_arr = np.array(preds)
    preds_arr_idx = np.argmax(preds_arr, axis=1)
    
    cm = confusion_matrix(y_arr, preds_arr_idx)

    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    df_cm = pd.DataFrame(cm, 
                     index=[i for i in text_labels], 
                     columns = [i for i in text_labels])
    
    return sn.heatmap(df_cm, annot=True, cmap='Reds', fmt='g')


def get_top_X_wrong_predictions(X, preds, idx_false_preds, test_ds):
    """ 
    Show top X wrong prediction according to the model's confidence in the wrong prediction.
    """
    
    assert (X%5 ==0), 'X must be a multiple of 5'
    
    false_preds = preds[idx_false_preds].detach().cpu().numpy()
    
    predictions_false = []
    probabilities_false = []
    true_labels_false = []

    for i in range(false_preds.shape[0]):
        pred = false_preds[i]
        pred_soft = softmax(pred)
        pred_label = np.argmax(pred_soft)
        predictions_false.append(pred_label)
        pred_prob = pred_soft[pred_label]
        probabilities_false.append(pred_prob)
        idx_pred = idx_false_preds[i]
        true_label = test_ds.labels[idx_pred].item()
        true_labels_false.append(true_label)
        
    probabilities_false_sorted = sorted(probabilities_false, reverse=True)
    predictions_false_sorted = [pred for _,pred in sorted(zip(probabilities_false, predictions_false), reverse=True)]
    true_labels_false_sorted = [label for _,label in sorted(zip(probabilities_false, true_labels_false), reverse=True)]
    idx_false_preds_sorted = [idx for _,idx in sorted(zip(probabilities_false, idx_false_preds), reverse=True)]
    
    idx_top_X_false = idx_false_preds_sorted[:X]
    
    show_false_predictions(test_ds.images[idx_top_X_false].reshape(X,28,28), int(X/5), 5, true=get_label(true_labels_false_sorted[:X]), 
                           pred=get_label(predictions_false_sorted[:X]), probabilities=probabilities_false_sorted[:X])
    

def show_false_predictions(images, num_rows, num_cols, true=None, pred=None, probabilities=None, scale=1.5):
    """
    Show image alongside the predicted and true label.
    """
    
    figsize = (num_cols * 2, num_rows * 1.5)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    figure.tight_layout()
    for i, (ax, images) in enumerate(zip(axes, images.cpu())):
        ax.imshow(np.array(images), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if true and pred:
            ax.set_title(f'Label: {true[i]}\nPred: {pred[i]} ({probabilities[i]:.2f})')
    plt.tight_layout()
    return axes


def get_label(label):
    """
    To get a label as a string when entering a numeric label.
    """
    
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in label]


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()