import gzip
import pickle

import numpy as np


def get_data_MNIST(subset, data_path="../data"):
    """
    Takes in a subset of data ("train" or "test"), unzips the inputs and labels files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy array of labels).

    Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data.

    If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the MNIST data
    :return:
        inputs (NumPy array of float32)
        labels (NumPy array of uint8)
    """
    ## http://yann.lecun.com/exdb/mnist/
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    ## TODO: read the image file and normalize, flatten, and type-convert image
    image = None

    ## TODO: read the label file
    label = None

    return image, label


def get_data_CIFAR(subset, data_path="../data"):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ...,
    as well as test_batch, so you'll need to combine all train batches
    into one batch. Each of these files is a Python "pickled"
    object produced with cPickle. The code below will open up each
    "pickled" object (i.e. each file) and return a dictionary.

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the CIFAR data
    :return:
        inputs (NumPy array of uint8),
        labels (NumPy array of string),
        label_names (NumPy array of strings)
    """

    ## https://www.cs.toronto.edu/~kriz/cifar.html
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    data_files = {
        "train": [f"data_batch_{i+1}" for i in range(5)],
        "test": ["test_batch"],
    }[subset]
    data_meta = f"{data_path}/cifar/batches.meta"
    data_files = [f"{data_path}/cifar/{file}" for file in data_files]

    # TODO #1:
    #   Pull in all of the data into cifar_dict.
    #   Check the cifar website above for the API to unpickle the files.
    #   Then, you can access the components i.e. 'data' via cifar_dict[b"data"].
    #   If data_files contains multple entries, make sure to unpickle all of them
    #   and concatenate the results together into a single training set.

    cifar_dict = {  ## HINT: Might help to start out with this
        b"data": [],
        b"labels": [],
    }
    cifar_meta = None

    image = cifar_dict[b"data"]
    label = cifar_dict[b"labels"]
    label_names = cifar_meta[b"label_names"]

    # TODO #2:
    #   Currently, the variable "label" is a list of integers between 0 and 9,
    #     with 0 meaning "airplane", 1 meaning "automobile" and so on.
    #   You should change the label with more descriptive names, given in the
    #   Numpy array variable "label_names" (remember that label_names contains
    #   binary strings and not UTF-8 strings right now)
    #   This variable "label" should be a Numpy array, not a Python list.

    label = None

    # TODO #3:
    #   You should reshape the input image np.array to (num, width, height, channels).
    #   Currently, it is a 2D array in the shape of (images, flattened pixels)
    #   You should reshape it into (num, 3, 32, 32), because the pickled images are in
    #     three channels(RGB), and 32 pixels by 32 pixels.
    #   However, we want the order of the axis to be in (num, width, height, channels),
    #     with the RGB channel in the last dimension.
    #   We want the final shape to be (num, 32, 32, 3)

    image = None

    # DO NOT normalize the images by dividing them with 255.0.
    # With the MNIST digits, we did normalize the images, but not with CIFAR,
    # because we will be using the pre-trained ResNet50 model, which requires
    # the pixel values to be unsigned integer values between 0 and 255.

    return image, label, label_names


def shuffle_data(image_full, label_full, seed):
    """
    Shuffles the full dataset with the given random seed.

    NOTE: DO NOT EDIT

    It's important that you don't edit this function,
    so that the autograder won't be confused.

    :param: the dataset before shuffling
    :return: the dataset after shuffling
    """
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full


def get_specific_class(image_full, label_full, specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a specific digit.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image_full: the image array returned by the get_data function
    :param label_full: the label array returned by the get_data function
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """
    # TODO:
    #   Return the first "num" number of images and labels from the full dataset
    #   For example, if your dataset looks something like this:
    #     example = ["a1", "b1", "a2", "a3", "c1", b2", "c2", "a4", "c3", ...]
    #     and if you want only the first 3 samples of the class "a"
    #     you will return ["a1", "a2", "a3"]
    # Hint: Numpy mask operations and index slicing will be useful.

    image = None
    label = None

    return image, label


def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a list of specific digits.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image: the image array returned by the get_data function
    :param label: the label array returned by the get_data function
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """
    # TODO:
    #   Repeat collecting the first "num" number of images and labels
    #     from the full dataset for each class in class_list, and concatenate them all
    #     for example, if your dataset looks something like this:
    #       example = ["a1", "b1", "a2", "a3", "c1", b2", "c2", "a4", "c3", ...]
    #     and if you want only the first 3 samples from each class
    #     you will return ["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3"]
    # Hint 1: Use the get_specific_class function as a helper function.
    # Hint 2: You may follow the suggested steps below, but you don't have to.

    image_list = []
    label_list = []

    # TODO: use the get_specific_class function for each class in class_list

    # TODO: concatenate the image and label arrays for all classes and
    # make sure the return statement follows the specifications of the docstring
    image = None
    label = None

    return image, label
