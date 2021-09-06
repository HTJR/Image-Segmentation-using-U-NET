import tensorflow as tf
from Data import load_data, tf_dataset
from model import build_unet


if __name__ == "__main__":
    
    path = "oxford-iiit-pet/"
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    shape = (256, 256, 3)
    num_classes = 3
    lr = 1e-4
    batch_size = 64
    epochs = 50

    """ Model """
    model = build_unet(shape, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr))

    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size
    model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_data=valid_dataset,
        validation_steps=valid_steps,
        epochs=epochs,
        verbose=1,
    )
    model.save('saved_model/my_model')
