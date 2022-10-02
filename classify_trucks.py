import tensorflow as tf


def conv_net_truck_classifier() -> tf.keras:
    """Convolutional Neural Net
    input: 224x224x3 (RGB)
    we just set up a CNN architecture but we don't train it"""

    model = tf.keras.models.Sequential([
        # first layer needs input_shape attribute; the other layers can just assume it is the same as the previous layer's output
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                               activation="relu", kernel_initializer="he_normal", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=20, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(units=1, activation="sigmoid", kernel_initializer="glorot_normal")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"])

    print(type(model))
    return model


print(conv_net_truck_classifier())

