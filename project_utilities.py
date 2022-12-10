# Imports
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# These are custom image plotting utility functions I wanted to move out of the notebook for the sake of simplicity
def plotImages(a, b):
    plt.figure(figsize=(15, 15))
    for i, k in enumerate(range(1,9)):
        if i < 4:
            plt.subplot(2,4,k)
            plt.title('Not A Ship')
            plt.imshow(a[i])
        else:
            plt.subplot(2,4,k)
            plt.title('Full Ship')
            plt.imshow(b[i])
            
    plt.subplots_adjust(bottom=0.2, top=0.5, hspace=0.5)
    
def plotHistogram(ship, not_ship):
    # plot the ship image first
    plt.figure(figsize = (10,7))
    plt.subplot(2,2,1)
    plt.imshow(ship)
    plt.title('Ship')
    # plot the histogram in the index to the right
    histo = plt.subplot(2,2,2)
    histo.set_ylabel('Count', fontweight = "bold")
    histo.set_xlabel('Pixel Intensities', fontweight = "bold")
    n_bins = 30
    # get the red
    plt.hist(ship[:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
    # get the green
    plt.hist(ship[:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
    # get the blue
    plt.hist(ship[:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
    plt.show()
    # plot the non ship image next
    plt.figure(figsize = (10,7))
    plt.subplot(2,2,3)
    plt.imshow(not_ship)
    plt.title('Not Ship')
    # plot the histogram in the index to the right
    histo = plt.subplot(2,2,4)
    histo.set_ylabel('Count', fontweight = "bold")
    histo.set_xlabel('Pixel Intensity', fontweight = "bold")
    n_bins = 30
    # get the red
    plt.hist(not_ship[:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
    # get the green
    plt.hist(not_ship[:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
    # get the blue
    plt.hist(not_ship[:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
    plt.show()

def plotSingleHistogram(image):
    # first plot the image
    plt.figure(figsize = (10,7))
    plt.subplot(2,2,1)
    plt.imshow(image)
    # then plot the pixel intensity histogram
    plt.subplot(2,2,2)
    n_bins = 30
    plt.hist(image[:,:,0].flatten(), bins = n_bins, lw = 0, color = 'r', alpha = 0.5);
    plt.hist(image[:,:,1].flatten(), bins = n_bins, lw = 0, color = 'g', alpha = 0.5);
    plt.hist(image[:,:,2].flatten(), bins = n_bins, lw = 0, color = 'b', alpha = 0.5);
    plt.ylabel('Count', fontweight = "bold")
    plt.xlabel('Pixel Intensity', fontweight = "bold")
    plt.figure(figsize = (10,7))
    plt.show()

def plotRGBChannels(image):
    my_list = [(0, 'R channel'), (1, 'G channel'), (2, 'B channel')]

    plt.figure(figsize = (15,15))

    for i, k in my_list:
        plt.subplot(1,3,i+1)
        plt.title(k)
        plt.ylabel('Height {}'.format(image.shape[0]))
        plt.xlabel('Width {}'.format(image.shape[1]))
        # this goes through each of the color channels
        plt.imshow(image[ : , : , i])

def plot_confusion(model, X_val, y_val):
    pred = model.predict(X_val)
    predict_class = np.argmax(pred, axis=1)
    expected_class = np.argmax(y_val, axis=1)
    confusion_mtx = confusion_matrix(expected_class, predict_class) 
    # Plot the confusion matrix
    f,ax = plt.subplots(figsize=(7, 7))
    sb.heatmap(confusion_mtx, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
        
# The functions below came from the lab util file
def plot_history(h, plot_type='loss'):
    history_df = pd.DataFrame(h.history)
    history_df = history_df.reset_index()
    
    loss_df_long = history_df.melt(id_vars='index', value_vars=['loss', 'val_loss'])
    loss_df_long.columns = ['Epoch', 'Loss Type', 'Loss']
    
    acc_df_long = history_df.melt(id_vars='index', value_vars=['accuracy', 'val_accuracy'])
    acc_df_long.columns = ['Epoch', 'Accuracy Type', 'Accuracy']
    
    if plot_type == 'loss':
        sb.lineplot(data=loss_df_long, x='Epoch', y='Loss', hue='Loss Type')
        plt.legend(title='Loss Type', labels=['Training Loss', 'Validation Loss'])
        
    elif plot_type == 'accuracy':
        sb.lineplot(data=acc_df_long, x='Epoch', y='Accuracy', hue='Accuracy Type')
        plt.legend(title='Accuracy Type', labels=['Training Accuracy', 'Validation Accuracy'])
        
    else:
        plt.subplots(2, 1, figsize=(15, 7))
        plt.subplot(1, 2, 1)
        sb.lineplot(data=loss_df_long, x='Epoch', y='Loss', hue='Loss Type')
        plt.legend(title='Loss Type', labels=['Training Loss', 'Validation Loss'])
        plt.subplot(1, 2, 2)
        sb.lineplot(data=acc_df_long, x='Epoch', y='Accuracy', hue='Accuracy Type')
        plt.legend(title='Accuracy Type', labels=['Training Accuracy', 'Validation Accuracy'])
        
def compute_accuracy(model, X_val, y_val):
    pred = model.predict(X_val)
    predict_class = np.argmax(pred, axis=1)
    expected_class = np.argmax(y_val, axis=1)
    return accuracy_score(expected_class, predict_class)

def compute_accuracy_binary_class(model, X_val, y_val):
    pred = model.predict(X_val)
    return accuracy_score(y_val, pred)

def plot_confusion_binary_class(model, X_val, y_val):
    pred = model.predict(X_val)
    confusion_mtx = confusion_matrix(y_val, pred) 
    # Plot the confusion matrix
    f,ax = plt.subplots(figsize=(7, 7))
    sb.heatmap(confusion_mtx, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
def plot_grid_search (grid_search, output_score='mean_test_score', output_label='Accuracy', x=None, hue=None, log_scale=False):
    params = grid_search.cv_results_['params']
    scores = grid_search.cv_results_[output_score]
    rows = [dict(param_dict, score=score) for param_dict, score in zip(params, scores)]
    cols = list(params[0].keys())
    cols.append(output_label)

    results_df = pd.DataFrame(rows)
    results_df.columns = cols
    
    if x is None:
        x = cols[0]
    if hue is None and len(cols) > 2:
        hue = cols[1]
        
    if hue is None:
        splot = sb.lineplot(data=results_df, x=x, y=output_label, marker='o')
    else: 
        splot = sb.lineplot(data=results_df, x=x, y=output_label, hue=hue, marker='o', palette='bright')
    
    if log_scale:
        splot.set(xscale='log')

    return results_df