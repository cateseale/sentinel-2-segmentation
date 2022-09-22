import mlflow
import tensorflow as tf


class MlflowCallback(tf.keras.callbacks.Callback):

    # This function will be called after each epoch.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        # Log the metrics from Keras to MLflow
        mlflow.log_metric('Train loss', logs['loss'], step = epoch)
        mlflow.log_metric('Train accuracy', logs['accuracy'], step = epoch)
        mlflow.log_metric('Validation loss', logs['val_loss'], step = epoch)
        mlflow.log_metric('Validation accuracy', logs['val_accuracy'], step = epoch)


        # This function will be called after training completes.
    def on_train_end(self, logs=None):
        mlflow.log_param('num_layers', len(self.model.layers))
        mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)

