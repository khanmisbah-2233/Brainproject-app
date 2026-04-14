import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from src.config import MODEL_PATH, TRAINING_LOG_PATH, LEARNING_RATE, VERBOSE


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def weighted_cce(y_true, y_pred):
    class_weights = tf.constant([0.2, 1.2, 1.2, 1.4], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    return tf.reduce_mean(cce * weights)


def combined_loss(y_true, y_pred):
    return weighted_cce(y_true, y_pred) + dice_loss(y_true, y_pred)


def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(TRAINING_LOG_PATH)
    ]


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=[dice_coef, iou_metric, "accuracy"]
    )
    return model


def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs):
    model = compile_model(model)

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=get_callbacks(),
        verbose=VERBOSE
    )

    return model, history