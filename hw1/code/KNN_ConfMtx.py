import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

class KNN_ConfMtx:  
    
    def get_confusion_matrix(self, label_test, prediction_array):
        """
        This method evaluates the confusion matrix of the model's predictions on a new test dataset.
        
        :param label_test: true test labels
        :param prediction_array: predicted labels
        :return confusion_mat: confusion matrix in np.array
        """
        num_classes = len(self.class_list)
        confusion_mat = np.zeros((num_classes, num_classes), dtype = np.int32)
        class_to_index = {c:i for i, c in enumerate(self.class_list)}

        for each_true, each_pred in zip(label_test, prediction_array):
            each_true_index = class_to_index[each_true]
            each_pred_index = class_to_index[each_pred]
            confusion_mat[each_true_index, each_pred_index] += 1
            
        return confusion_mat
        
    def visualize_confusion_matrix(self, confusion_mat):
        """
        This method visualizes the confusion matrix.
        
        NOTE: DO NOT EDIT
        
        :param confusion_mat: confusion matrix in np.array
        :return fig_confusion: display figure
        :return ax_confusion: display axis
        """
        
        num_classes = len(self.class_list)
        num_samples = np.max(np.sum(confusion_mat, axis = 1))
        
        fig_confusion, ax_confusion = plt.subplots()

        cmap = mpl.cm.GnBu
        norm = mpl.colors.Normalize(vmin=0, vmax=num_samples)

        ax_confusion.imshow(confusion_mat, 
                            cmap = cmap, 
                            vmin = 0, 
                            vmax = num_samples)
        
        ax_confusion.set_xticks(range(num_classes))
        ax_confusion.set_xticklabels(self.class_list)
        ax_confusion.set_yticks(range(num_classes))
        ax_confusion.set_yticklabels(self.class_list)
        
        ax_confusion.set_ylabel("True Class")
        ax_confusion.set_xlabel("Predicted Class")
        ax_confusion.set_title("Confusion Matrix")

        ax_confusion.set_xticks(np.arange(-.5, num_classes, 1), minor=True)
        ax_confusion.set_yticks(np.arange(-.5, num_classes, 1), minor=True)
        ax_confusion.grid(which="minor", color="k", linestyle="-", linewidth=1)

        for each_true_index, each_pred_index \
            in itertools.product(range(num_classes), range(num_classes)):
            ax_confusion.text(each_pred_index, 
                              each_true_index, 
                              confusion_mat[each_true_index, each_pred_index], 
                              ha = "center", 
                              va = "center")

        fig_confusion.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            orientation="vertical", 
            label="Counts", 
            shrink = 0.83)

        fig_confusion.set_size_inches([8, 8])
        
        return fig_confusion, ax_confusion
    