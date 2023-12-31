import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl  # Add import for mpl
import seaborn as sns

import numpy as np
from sklearn.metrics import confusion_matrix

from Code.utilities import predict

############################################
### Plot letters
############################################
def plot_some_imgs(X_test, y_test):

    for i, image in enumerate(X_test[0:5]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')

        if len(image.shape) == 2:
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            n = int((image.shape[0])**(1/2))
            plt.imshow(image.reshape(n,n), cmap=plt.cm.gray_r, interpolation='nearest')
        
        plt.title(f"Label: {np.argmax(y_test[i])}")
    plt.show()

def plot_faulty_predictions(X_test, y_test, model, beta):

    # Example true labels and predicted labels
    nums_pred = np.argmax(predict(model, beta, X_test), axis=1)
    nums_gt   = np.argmax(y_test, axis=1)

    # Get the faulty predictions
    indeces = nums_gt != nums_pred

    imgs = X_test[indeces]
    labels_pred = nums_pred[indeces]
    labels_gt   = nums_gt[indeces]

    # Plot the 5 first
    for i, image in enumerate(imgs[0:5]):

        plt.subplot(1, 5, i+1)
        plt.axis('off')

        if len(image.shape) == 2:
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        else:
            n = int((image.shape[0])**(1/2))
            plt.imshow(image.reshape(n,n), cmap=plt.cm.gray_r, interpolation='nearest')
            
        plt.title(f"Label: {labels_gt[i]} \nPredicted: {labels_pred[i]}", size=12)

    plt.show()


############################################
### Confusion matrix
############################################
def plot_confusion_matrix(X_test, y_test, model, beta, filename=None, convert_to_percent=False, title="Confusion matrix"):

    # Initialise the figure
    plt.figure(figsize=(15,13))

    # Example true labels and predicted labels
    nums_pred = np.argmax(predict(model, beta, X_test), axis=1)
    nums_gt   = np.argmax(y_test, axis=1)

    # Create confusion matrix. If desired, convert to percentage
    conf_matrix = confusion_matrix(nums_gt, nums_pred)

    # Plot with or without converting to percentages
    if convert_to_percent: 
        conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        sns.heatmap(conf_matrix, annot=True, fmt=".1%", cmap="Blues", square=True,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))
    else:
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", square=True,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))

    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


############################################
### 
############################################



############################################
### Parallell Coordinates
############################################
def plot_grid_search_result(data):
    """
        Creates plot much like the ones provided by "weights and biases". 


        Input: a dictionary like 

            data = {
                "Epochs": [10, 20, 30, 40, 59],
                "Batch size": [8, 16, 32, 64, 54],
                "Learning rate": [0.01, 0.01, 0.02, 0.03, 0.07],
                "Regularisation": [0.1, 0.01, 0.02, 0.03, 0.07],
                "Accuracy": [1, 0.9, 0.8, 0.7, 0.4],
            }

            Must at least contain the Accurazy row...

        Code inspired by work in the following thread:
        https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib

        """


    # Use mpl.cm for colormaps
    cmap = mpl.cm.get_cmap('viridis')
    acc_min, acc_max = min(data["Accuracy"]), max(data["Accuracy"])
    colors = [cmap((acc - acc_min) / (acc_max - acc_min)) for acc in data["Accuracy"]]

    ynames = data.keys()
    yvals = np.array([data[key] for key in ynames]).T
    ys = yvals

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05
    ymaxs += dys * 0.05

    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


    ### MAKE THE PLOT
    fig, host = plt.subplots(figsize=(10, 4))

    # Make axes
    axes = [host] + [host.twinx() for _ in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
            if i == len(axes) - 1:  # Exclude ticks and labels for the rightmost axis
                ax.set_yticks([])  # Remove ticks
                ax.set_yticklabels([])  # Remove tick labels
            
            if list(ynames)[i] == "Regularisation":
                ax.set_yscale("log")  # Remove tick labels

    # Customise axis
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title('Result of grid search', fontsize=18, pad=12)

    # Add the spline curves
    for j in range(ys.shape[0]):

        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                        np.repeat(zs[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1.5, alpha=0.7, edgecolor=colors[j])
        host.add_patch(patch)

    cax = fig.add_axes([0.95, 0.06, 0.02, 0.7])
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=acc_min, vmax=acc_max), cmap=cmap), cax=cax)
    
    plt.tight_layout()
    plt.show()


############################################
### Plot a training run
############################################
def plot_test_results(test_loss_list, train_loss_list, ylabel="INSERT"):

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))  # 1 row, 2 columns

    ax.plot(test_loss_list, label="test")
    ax.plot(train_loss_list, label="train")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title("End of epoch error")
    ax.legend()

    plt.tight_layout()
    plt.show()
