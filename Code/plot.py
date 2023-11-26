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
        n = int((image.shape[0])**(1/2))
        plt.imshow(image.reshape(n,n), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"Label: {np.argmax(y_test[i])}")
    plt.show()

def plot_faulty_predictions(X_test, y_test, model, beta):

    # Example true labels and predicted labels
    nums_pred = predict(model, beta, X_test)
    nums_gt   = np.array([np.argmax(y_sample) for y_sample in y_test])

    # Format the faulty predictions back
    indeces = nums_gt != nums_pred
    if X_test.shape[1] not in [8, 28]:
        n = int(np.sqrt(X_test.shape[1]))
        imgs = [img.reshape(n, n) for img in X_test[indeces]]
    else:
        imgs = X_test[indeces]
    labels_pred = nums_pred[indeces]
    labels_gt   = nums_gt[indeces]

    # Plot the 5 first
    for i, img in enumerate(imgs[0:5]):

        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"Label: {labels_gt[i]} \nPredicted: {labels_pred[i]}", size=12)

    plt.show()


############################################
### Confusion matrix
############################################
def plot_confusion_matrix(X_test, y_test, model, beta, filename=None, convert_to_percent=False, title="Confusion matrix"):

    # Initialise the figure
    plt.figure(figsize=(10, 8))

    # Example true labels and predicted labels
    nums_pred = predict(model, beta, X_test)
    nums_gt   = np.array([np.argmax(y_sample) for y_sample in y_test])

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

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
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
        patch = patches.PathPatch(path, facecolor='none', lw=1.5, alpha=1, edgecolor=colors[j])
        host.add_patch(patch)

    cax = fig.add_axes([0.95, 0.06, 0.02, 0.7])
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=acc_min, vmax=acc_max), cmap=cmap), cax=cax)
    
    plt.tight_layout()
    plt.show()


############################################
### Plot a training run
############################################
def plot_test_results(test_loss_list, train_loss_list, num_batches, ylabel="INSERT"):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))  # 1 row, 2 columns

    # Subplot 1
    axs[0].plot(test_loss_list, label="test")
    axs[0].plot(train_loss_list, label="train")
    axs[0].set_xlabel("Training step")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title("Over all sub-epochs")
    axs[0].legend()

    # Subplot 2
    axs[1].plot(test_loss_list[::num_batches], label="test")
    axs[1].plot(train_loss_list[::num_batches], label="train")
    axs[1].set_xlabel("Training step")
    axs[1].set_title("End of epoch error")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
