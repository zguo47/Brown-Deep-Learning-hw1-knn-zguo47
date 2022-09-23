import numpy as np
from KNN_ConfMtx import KNN_ConfMtx


class KNN_Model(KNN_ConfMtx):

    """
    imported from KNN_ConfMtx:
        get_confusion_matrix
        visualize_confusion_matrix
    """

    def __init__(self, class_list, k_neighbors):
        """
        Initialize your KNN Model here.

        NOTE: DO NOT EDIT

        :param class_list:
            This is the list of class labels the model will learn how to classify.
            For example, they can be the list of digits in the MNIST dataset
                class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            As another example, the class labels can be the picture objects in the CIFAR-10 dataset
                class_list = ["airplane", "automobile", "bird", "cat", "deer",
                              "dog", "frog", "horse", "ship", "truck"]
        :param k_neighbors:
            This is the "K" in KNN, the number of neighbors of which true label will be surveyed
                by the model to make a prediction.
            For example, if k_neighbors == 9, and and the true labels
                of these nine neighbors are [2, 2, 2, 2, 3, 2, 5, 2, 2],
                the label 2 wins the majority vote.
                In this example, the model's prediction will be 2.
            As another example, the neighbors are ["cat", "cat", "cat",
                                                    "deer", "dog", "cat",
                                                    "cat", "dog", "cat"],
                the label "cat" wins the majority vote.
                In this example, the model's prediction will be "cat".
        :return: This method does not return anything.
        """
        self.class_list = class_list
        self.k_neighbors = k_neighbors

    def __call__(self, new_image):
        """
        This method makes calling the model class's instance
        the same thing as calling the predict method.
        When you build TensorFlow subclass models in HW 2,
        you will do pretty much the same thing

        NOTE: DO NOT EDIT
        """
        return self.predict(new_image)

    def fit(self, image_train, label_train):
        """
        This method 'trains' your model.

        NOTE: DO NOT EDIT

        :param image_train: the images in the train set.
                            in the Numpy array format of shape (n_images, image_size)
                            for example, if you have 1000 images of 28*28 in your train set,
                            the shape of the Numpy array should be (1000, 784)
        :param label_train: the true labels in the train set.
        :return: This method does not return anything.
        """
        self.image_train = np.array(image_train).copy()
        self.label_train = np.array(label_train).copy()

    def get_neighbor_counts(self, new_image, return_indices=False):
        """
        This method compares the new image to all images in self.image_train,
        and identifies the self.k_neighbors nearest neighbors
        and returns the number of occurrences of each class label in self.class_list
        among the neighbors

        :param new_image:
            The new image that the model will predict the class of.
            This new image must be a single image, flattened to be a one-dimensional Numpy array.
            For example, if the original pixel size of the image is 28*28,
                then then image must be flattened to be a np.array of size 784
                (new_image.shape == (784, ))
            As another example, if the original pixel size of the image is 32*32*3,
                then then image must be flattened to be a np.array of size 3072
                (new_image.shape == (3072, ))
            Also, new_image is NOT a batch of multiple images.
                For example, if you have five images on which you want to make predictions,
                call this method FIVE SEPARATE TIMES,
                and DO NOT plug in a 2D array of the size (5, 784) to this method.
            You will eventually batch your input in the later homework assignments with TensorFlow,
                but not this time.
        :return class_counts:
            Python list containing the number of occurrences of the classes among the neighbors.
            For example, if the neighbors are ["cat", "cat", "cat",
                                               "deer", "dog", "cat",
                                               "cat", "dog", "cat"],
            Then, class_counts should be [0, 0, 0, 6, 1, 2, 0, 0, 0, 0],
                because that's the order in self.class_list, which is defined as
                ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
                in this example.
        """

        # TODO #1: create a Numpy array called "distances".
        # This is a one-dimensional array of the squared Euclidean distances
        #     between the new image and all images in self.image_train
        # For example, if this model was trained with 1,000 images,
        #     there are 1,000 distances:
        #         distance[0] = squared distance between the new image and the first training image
        #         distance[1] = squared distance between the new image and the second training image
        #         distance[2] = squared distance between the new image and the third training image
        #         ...
        #         distance[999] = squared distance between the new image and the 1000th training image
        #     Thus, the length of this variable should also be 1000. (distances.shape == 1000)
        # Hint: If you make clever use of numpy functions and the Numpy broadcasting rules,
        #     you can construct this variable without using any for-loops.
        distances = None

        # TODO #2: create a Numpy array called "nearest_indices".
        # This is a one-dimensional array of the indices
        #     of the nearest neighboring images in self.image
        # For example, if the nearest images are
        #     [self.image[0], self.image[20], self.image[35],
        #      self.image[159], self.image[489], self.image[527],
        #      self.image[744], self.image[895], self.image[941]],
        #     their indices are [0, 20, 35, 159, 489, 527, 744, 895, 941],
        # then nearest_indices should be
        #     np.array([0, 20, 35, 159, 489, 527, 744, 895, 941]) in this example.
        # Hint: If you make a clever use of np.argpartition,
        #     you can construct this variable
        #     without using any for-loops or any if-else statements
        # Hint2: If np.argpartition is too confusing, you can use a different numpy function instead
        #        The difference is that np.argpartition takes O(n) amount of time,
        #        while the alternative takes O(n*log(n)) amount of time.
        nearest_indices = None
        nearest_label = None
        class_counts = None

        return (class_counts, nearest_indices) if return_indices else class_counts

    def predict(self, new_image):
        """
        This method takes a majority vote on the neighbors to make the final prediction.
        For example, if class_counts is [0, 0, 7, 1, 0, 1, 0, 0, 0, 0],
            and self.class_list is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            then the the model's prediction will be 2.
        As another example, if class_counts is [0, 0, 0, 6, 1, 2, 0, 0, 0, 0],
            and self.class_list is ["airplane", "automobile", "bird", "cat", "deer",
                                    "dog", "frog", "horse", "ship", "truck"],
            then the the model's prediction will be "cat".

        :param new_image: the new image that the model will classify
                          into one of the classes in its self.class_list
        :return prediction_label:
        """
        # TODO: return the most frequent label among the neighbors
        # Hint: use numpy and remember the variable self.class_list
        class_counts_array = None
        prediction_label = None

        return prediction_label

    def get_prediction_array(self, image_test):
        """
        This method returns multiple predictions in the form of a Numpy array.

        :param image_test: test images
        :param label_test: test labels
        :return prediction_array: the Numpy array of predictions.
        """
        prediction_list = []
        for i, each_image in enumerate(image_test):
            # TODO #1: Populate the prediction_list array.

            # TODO #2: Print something when the index i is a multiple of some number,
            #      so that you can check the progress. For the MNIST dataset you probably don't need this progress check,
            #      but for CIFAR, you will probably find it useful. Progress checks are also a useful pattern commonly
            #      used when training machine learning models.
            continue

        return np.array(prediction_list)
