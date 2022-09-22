import tensorflow as tf
import numpy as np
import math


def augment_data(dataset,
                 dataset_labels,
                 use_rotate90=True,
                 use_rotate180=True,
                 use_rotate270=True,
                 use_flip_lr=True,
                 use_flip_ud=True
                 ):
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):

        # original image:
        augmented_image.append(dataset[num])
        augmented_image_labels.append(dataset_labels[num])

        if use_rotate90:
            augmented_image.append(np.rot90(dataset[num], k=1, axes=(0, 1)))
            augmented_image_labels.append(np.rot90(dataset_labels[num], k=1, axes=(0, 1)))

        if use_rotate180:
            augmented_image.append(np.rot90(dataset[num], k=2, axes=(0, 1)))
            augmented_image_labels.append(np.rot90(dataset_labels[num], k=2, axes=(0, 1)))

        if use_rotate270:
            augmented_image.append(np.rot90(dataset[num], k=3, axes=(0, 1)))
            augmented_image_labels.append(np.rot90(dataset_labels[num], k=3, axes=(0, 1)))

        if use_flip_lr:
            augmented_image.append(np.fliplr(dataset[num]))
            augmented_image_labels.append(np.fliplr(dataset_labels[num]))

        if use_flip_ud:
            augmented_image.append(np.flipud(dataset[num]))
            augmented_image_labels.append(np.flipud(dataset_labels[num]))

    return np.array(augmented_image), np.array(augmented_image_labels)


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, one_hot=True, augment=False, label_smoothing=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augment
        self.smoothing = label_smoothing
        self.onehot = one_hot


    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_arr = np.array([np.load(file_name).astype('float32') for file_name in batch_x])
        batch_y_arr = np.array([np.load(file_name).astype('float32') for file_name in batch_y])

        batch_y_arr[tuple([batch_y_arr < 0])] = 0

        if self.augment:
            batch_x_arr, batch_y_arr = augment_data(batch_x_arr, batch_y_arr)

        if self.onehot:
            batch_y_arr = tf.keras.utils.to_categorical(batch_y_arr, num_classes=2).astype(int)

            if self.smoothing:
                batch_y_arr = smooth_labels(batch_y_arr.astype(float))

        # return batch_x_arr, np.expand_dims(batch_y_arr, axis=3)
        return batch_x_arr, batch_y_arr


class TestDataLoader:


    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set


    def load_data(self):
        batch_x_arr = np.array([np.load(file_name).astype('float32') for file_name in self.x])
        batch_y_arr = np.array([np.load(file_name).astype('float32') for file_name in self.y])

        batch_y_arr[tuple([batch_y_arr < 0])] = 0
        # batch_y_arr_encoded = tf.keras.utils.to_categorical(batch_y_arr, num_classes=2).astype(int)

        return batch_x_arr, batch_y_arr


def filter_all_zero_samples(images_list, labels_list):

    filtered_img_list = []
    filtered_lab_list = []

    for count, item in enumerate(zip(images_list, labels_list)):

        img, lab = item

        lab_data = np.load(lab)

        if np.all((lab_data == 0)):
            pass
        else:
            filtered_img_list.append(img)
            filtered_lab_list.append(lab)

    return filtered_img_list, filtered_lab_list


def filter_all_one_samples(images_list, labels_list):

    filtered_img_list = []
    filtered_lab_list = []

    for count, item in enumerate(zip(images_list, labels_list)):

        img, lab = item

        lab_data = np.load(lab)

        if np.all((lab_data == 1)):
            pass
        else:
            filtered_img_list.append(img)
            filtered_lab_list.append(lab)

    return filtered_img_list, filtered_lab_list
