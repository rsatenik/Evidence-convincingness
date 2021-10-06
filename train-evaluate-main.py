from datetime import datetime
from tensorflow import keras
import tensorflow as tf

from preprocessing import import_data
from model import my_models
max_length = 128  # Maximum length of input sentence
batch_size = 32
epochs = 2


if __name__ == "__main__":
    train_data, test_data, valid_data = import_data()
    print("Create our model...")
    model, bert_model = my_models(max_length)
    # Define the Keras TensorBoard callback
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
    )

    # fine-tuning
    # Unfreeze the bert_model.
    bert_model.trainable = True
    # Recompile the model to make the change effective
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        callbacks=[tensorboard_callback]
    )

    model.evaluate(test_data, verbose=1)  # Evaluate the model on test data
    model.save_weights('./checkpoints/my_checkpoint')  # save the weights so that they can be directly used in test.py
