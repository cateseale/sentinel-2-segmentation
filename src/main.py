import os
import mlflow
import tensorflow as tf
from glob import glob
from loss_functions import sorensen_dice_coef_loss, sobel_loss
from data_loading import DataLoader, TestDataLoader, filter_all_zero_samples, filter_all_one_samples
from models import unet, lobnet, unet_with_dropout, hyperopt_model, Deeplabv3
from callbacks import MlflowCallback
from testing import calculate_test_metrics, plot_roc_curve
from natsort import natsorted
from mlflow import log_metric, log_param, log_artifact
from mlflow.keras import log_model
import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K


if __name__ == "__main__":

    K.clear_session()

    mlflow.set_experiment('ExpName')

    data_dir = '/home/ubuntu/data'

    training_image_dir = os.path.join(data_dir, 'train','images')
    training_label_dir = os.path.join(data_dir, 'train', 'labels')

    val_image_dir = os.path.join(data_dir, 'hold_out_test','images')
    val_label_dir = os.path.join(data_dir, 'hold_out_test', 'labels')

    test_image_dir = os.path.join(data_dir, 'oos_test','images')
    test_label_dir = os.path.join(data_dir, 'oos_test', 'labels')

    image_paths_list = natsorted(glob(os.path.join(training_image_dir, "*.npy")))
    label_paths_list = natsorted(glob(os.path.join(training_label_dir, "*.npy")))

    val_image_paths_list = natsorted(glob(os.path.join(val_image_dir, "*.npy")))
    val_label_paths_list = natsorted(glob(os.path.join(val_label_dir, "*.npy")))

    test_image_paths_list = natsorted(glob(os.path.join(test_image_dir, "*.npy")))
    test_label_paths_list = natsorted(glob(os.path.join(test_label_dir, "*.npy")))

    # To filter out any training images that contain no water or land, run these lines
    image_paths_list, label_paths_list = filter_all_zero_samples(image_paths_list, label_paths_list)
    image_paths_list, label_paths_list = filter_all_one_samples(image_paths_list, label_paths_list)


    BATCH_SIZE = 8
    WORKERS = 8

    enq_train = tf.keras.utils.OrderedEnqueuer(DataLoader(image_paths_list, label_paths_list, batch_size=BATCH_SIZE,
                                                          one_hot=True, augment=True, label_smoothing=True), shuffle=True)
    enq_train.start(workers=WORKERS)
    gen_train = enq_train.get()

    enq_val = tf.keras.utils.OrderedEnqueuer(DataLoader(val_image_paths_list, val_label_paths_list, batch_size=BATCH_SIZE))
    enq_val.start(workers=WORKERS)
    gen_val = enq_val.get()

    # Choose the model architecture
    model = unet(activation='relu', init='he_uniform')
    # model = hyperopt_model()
    # model= Deeplabv3(weights=None, input_tensor=None, input_shape=(256, 256, 12), classes=2, backbone='xception', OS=16,
    #           alpha=1., activation='softmax')

    # Set the loss function
    loss = tf.keras.losses.CategoricalCrossentropy()
    # loss = sorensen_dice_coef_loss
    # loss = sobel_loss

    # Set the optimiser
    learning_rate = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-07
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    # optimizer = tf.keras.optimizers.RMSprop()


    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    epochs = 50
    steps = len(image_paths_list) // BATCH_SIZE
    val_steps = len(val_image_paths_list) // BATCH_SIZE
    mlflow.end_run()
    with mlflow.start_run(run_name='ExpName') as run:

        run_id = run.info.run_id


        # Callbacks
        patience = 10
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, verbose=1,
                                                      restore_best_weights=True)
        checkpoint = ModelCheckpoint(run_id + '_weights.best.hdf5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')


        log_param('Batch size', BATCH_SIZE)
        log_param('Workers', WORKERS)
        log_param('Loss function', loss)
        log_param('Optimizer', optimizer)
        log_param('Patience', patience)
        log_param('Learning rate', learning_rate)
        log_param('Epochs', epochs)
        log_param('Steps', steps)
        log_param('Validation steps', val_steps)


        # Fit model
        history = model.fit(gen_train,
                            validation_data=gen_val,
                            epochs=epochs,
                            steps_per_epoch=steps,
                            validation_steps=val_steps,
                            callbacks=[early_stop, checkpoint, MlflowCallback()])

        log_model(model, 'model')

        # Reset metrics before saving so that loaded model has same state, since metric states are not preserved by
        # Model.save_weights
        model.reset_metrics()
        model.load_weights(run_id + '_weights.best.hdf5')
        model.save(run_id + '.h5')

        with open(run_id + '_history.h5', 'wb') as out_history:
            pickle.dump(history.history, out_history)

        log_artifact(run_id + '_history.h5')

        # Calculate test metrics on out-of-sample images and log in MLFlow
        dl = TestDataLoader(test_image_paths_list, test_label_paths_list)

        test_images, test_labels = dl.load_data()

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions, axis=3)

        np.save(run_id + '_predictions.npy', predictions)
        log_artifact(run_id + '_predictions.npy')

        test_acc, test_precision, test_recall, test_f1_score, test_matt_corr_coef, test_jaccard, test_kappa = \
            calculate_test_metrics(test_labels, predictions)
        plot_roc_curve(test_labels, predictions, run_id)

        log_metric('Test accuracy', test_acc)
        log_metric('Test precision', test_precision)
        log_metric('Test recall', test_recall)
        log_metric('Test F1 score', test_f1_score)
        log_metric('Test matthews correlation coef', test_matt_corr_coef)
        log_metric('Test jaccard score', test_jaccard)
        log_metric('Test kappa score', test_kappa)

        log_artifact(run_id + '_roc.png')

        mlflow.end_run()

    print('Pipeline complete.')
