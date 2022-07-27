import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix1(cm, classes, epoch,normalize=False, cmap=plt.cm.Blues):
    title = 'Confusion matrix for gsop50 model test'+str(epoch)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig('/mnt/gsop/matrix.tif', transparent=True, dpi=800)

    plt.show()




if __name__ == "__main__":
    #label = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    label = ["Orah", "Shiranui", "Murcott", "Jinyou", "Haruka", "Caracara", "Tarocco", "Jincheng"]
    # label = ["C1", "C2", "C3", "C4", "C5"]
    cm50=np.array([[287,1,1,0,3,2,4,2],
                   [3,338,2,1,1,0,2,1],
                   [1,5,145,1,0,0,2,0],
                   [0,0,0,155,0,0,0,0],
                   [0,0,0,0,216,0,0,0],
                   [1,1,2,0,0,224,3,1],
                   [4,5,2,0,0,10,228,0],
                   [0,2,0,0,0,0,0,238]])
    cm101=np.array([[293,1,3,0,0,1,2,0],
                    [3,335,1,1,1,2,5,0],
                    [1,2,148,0,0,1,2,0],
                    [0,0,0,154,1,0,0,0],
                    [1,1,0,0,214,0,0,0],
                    [2,1,1,0,0,225,3,0],
                    [4,4,3,0,0,8,230,0],
                    [2,1,1,0,1,0,0,235]])
    cm_TL50=np.array([[298,2,0,0,0,0,0,0],
                      [2,343,0,1,1,0,1,0],
                      [0,0,153,0,0,1,0,0],
                      [0,0,0,155,0,0,0,0],
                      [0,0,0,0,216,0,0,0],
                      [1,0,1,0,0,229,1,0],
                      [0,2,1,0,0,1,244,1],
                      [0,0,1,0,0,0,0,239]])
    cm_TL101=np.array([[299,1,0,0,0,0,0,0],
                       [2,343,0,1,0,0,2,0],
                       [0,1,153,0,0,0,0,0],
                       [0,0,0,155,0,0,0,0],
                       [0,0,0,0,216,0,0,0],
                       [0,0,0,0,0,231,0,1],
                       [1,0,0,0,0,0,248,0],
                       [2,0,0,0,0,0,1,237]])
    cmtx=np.array([[153,1,1,1,2,1,2,11],
          [1,242,4,3,1,1,0,2],
          [0,5,98,3,0,0,1,2],
          [0,6,2,126,3,1,0,1],
          [0,0,0,0,257,0,1,0],
          [0,2,1,1,2,144,11,6],
          [1,6,3,0,0,4,146,6],
          [4,0,1,0,2,0,0,270]])
    cm=np.array([[156,2,3,0,1,0,3,7],
                 [4,231,2,1,8,1,2,5],
                 [0,7,98,3,0,1,0,0],
                 [0,6,3,124,4,2,0,0],
                 [0,0,0,0,256,0,1,1],
                 [0,9,4,0,0,143,10,1],
                 [0,5,6,0,0,5,144,6],
                 [7,0,1,0,2,2,0,265]])
    plot_confusion_matrix1(cm_TL101, label,1)

