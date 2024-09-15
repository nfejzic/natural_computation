import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import keras.api as tf_keras
from keras.api.datasets import mnist
from keras.api.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.api.models import Sequential

import argparse


def get_activation_fn(name: str):
    match name:
        case "relu":
            return tf.nn.relu, "relu"
        case "silu":
            return tf.nn.silu, "silu"
        case "swish":
            return tf.nn.silu, "silu"
        case "sigmoid":
            return tf.nn.sigmoid, "sigmoid"
        case "tanh":
            return tf.nn.tanh, "tanh"
        case _:
            raise ValueError(f"Unknown activation function: {name}")


######################### MODEL ##################################
def create_model(activation="relu"):
    nn_act, act = get_activation_fn(activation)

    model = Sequential()
    model.add(tf_keras.Input(shape=(28, 28, 1)))
    # Convo Layers damits a CNN is...
    _ = (
        model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation=act,
            )
        ),
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convert to a 1-dimensional array as layers need such input
    # Flatten automatically converts N-dimensional arrays to 1-dimensional array
    model.add(Flatten())

    # Dense -> All neurons connected to each other.
    # We can initialize the layer directly with kernel_initializer, instead of taking a function
    model.add(
        Dense(
            32,
            activation=tf.nn.silu,
            kernel_initializer="glorot_uniform",
        )
    )

    model.add(Dense(32, activation=nn_act))
    model.add(Dense(32, activation=nn_act))
    model.add(Dense(32, activation=nn_act))

    # Softmax -> Sum of outputs equals to 1. We have 10 digits in dataset, so 10 outputs
    model.add(Dense(10, activation="softmax"))

    return model


######################### Initialization functions ##################################


def initialize_weights_xavier(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf_keras.initializers.GlorotUniform()
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


def initialize_weights_he(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf_keras.initializers.HeNormal()
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


def initialize_weights_random_normal(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf_keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


def initialize_weights_alternating(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf_keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
            new_weights = initializer(layer.get_weights()[0].shape)
            bias_initializer = tf_keras.initializers.RandomUniform(
                minval=-0.01, maxval=0.01
            )
            new_biases = bias_initializer(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


######################### Analysis ##################################


def evaluate_initialization(model, images, labels, initialization_fn):
    initialization_fn(model)
    # We predict on the first image to setup the model.
    _ = model.predict(images[:1])
    outputs = model.predict(images)

    predicted_labels = np.argmax(outputs, axis=1)
    correct_predictions = np.sum(predicted_labels == labels)
    total_predictions = labels.shape[0]
    accuracy = correct_predictions / total_predictions * 100
    print(f"\nAccuracy without training: {accuracy:.2f}%")

    class_probs = outputs[np.arange(len(labels)), labels]
    non_class_probs = outputs[np.arange(len(labels)), :].copy()
    non_class_probs[np.arange(len(labels)), labels] = np.nan

    # Here we add (in rows) and then divide by 9 - because 1 of 10 is a "nan".
    # The resulting i-th Row becomes an element in the i-th Column of the `non_class_probs`,
    # so that the `non_class_probs` same structure as `class_probs` has.
    non_class_probs = np.nanmean(non_class_probs, axis=1)

    class_mean = np.mean(class_probs)
    non_class_mean = np.mean(non_class_probs)
    mean_difference = abs(class_mean - non_class_mean)
    print(f"Mean difference between class and non-class values: {mean_difference:.8f}")

    return accuracy, mean_difference, outputs


def evaluate_with_activation(images, labels, runs=5, activation="relu"):
    xavier_acc, xavier_diff, xavier_outputs = ([], [], [])
    he_acc, he_diff, he_outputs = ([], [], [])
    rand_norm_acc, rand_norm_diff, rand_norm_outputs = ([], [], [])
    pos_neg_acc, pos_neg_diff, pos_neg_outputs = ([], [], [])

    for _ in range(runs):
        print("Testing Xavier Initialization")
        model = create_model(activation)

        # Uncomment `model.summary()` line to see the created model.
        # model.summary()

        accuracy_xavier, mean_diff_xavier, outputs_xavier = evaluate_initialization(
            model, images, labels, initialize_weights_xavier
        )

        xavier_acc.append(accuracy_xavier)
        xavier_diff.append(mean_diff_xavier)
        xavier_outputs.append(outputs_xavier)

        print(
            "###########################################################################\n"
        )

        print("\nTesting He Initialization")
        model = create_model(activation)
        # model.summary()
        accuracy_he, mean_diff_he, outputs_he = evaluate_initialization(
            model, images, labels, initialize_weights_he
        )

        he_acc.append(accuracy_he)
        he_diff.append(mean_diff_he)
        he_outputs.append(outputs_he)

        print(
            "###########################################################################\n"
        )

        print("\nTesting Random Normal Initialization")
        model = create_model(activation)
        # model.summary()
        accuracy_random_normal, mean_diff_random_normal, outputs_random_normal = (
            evaluate_initialization(
                model, images, labels, initialize_weights_random_normal
            )
        )

        rand_norm_acc.append(accuracy_random_normal)
        rand_norm_diff.append(mean_diff_random_normal)
        rand_norm_outputs.append(outputs_random_normal)

        print(
            "###########################################################################\n"
        )

        print("Testing Alternating Positive/Negative Initialization")
        model = create_model(activation)
        # model.summary()
        accuracy_alternating, mean_diff_alternating, outputs_alternating = (
            evaluate_initialization(
                model, images, labels, initialize_weights_alternating
            )
        )

        pos_neg_acc.append(accuracy_alternating)
        pos_neg_diff.append(mean_diff_alternating)
        pos_neg_outputs.append(outputs_alternating)

        print(
            "###########################################################################\n"
        )

    output = ""

    output += format(f"\nWith '{activation}' activation function:\n")

    # XAVIER
    output += format("Xavier:\n")
    output += format(
        f"\tAccurracy: std = {np.std(xavier_acc):.8f}; mean = {np.mean(xavier_acc):.8f}; median = {np.median(xavier_acc):.8f}\n"
    )
    output += format(
        f"\tMean Difference: std = {np.std(xavier_diff):.8f}; mean = {np.mean(xavier_diff):.8f}; median = {np.median(xavier_diff):.8f}\n"
    )
    output += format(
        f"\tOutputs: std = {np.std(xavier_outputs):.8f}; variance = {np.var(xavier_outputs)}; mean = {np.mean(xavier_outputs):.8f}; median = {np.median(xavier_outputs):.8f}\n"
    )

    # He
    output += format("\n\nHe:\n")
    output += format(
        f"\tAccurracy: std = {np.std(he_acc):.8f}; mean = {np.mean(he_acc):.8f}; median = {np.median(he_acc):.8f}\n"
    )
    output += format(
        f"\tMean Difference: std = {np.std(he_diff):.8f}; mean = {np.mean(he_diff):.8f}; median = {np.median(he_diff):.8f}\n"
    )
    output += format(
        f"\tOutputs: std = {np.std(he_outputs):.8f}; variance = {np.var(he_outputs)}; mean = {np.mean(he_outputs):.8f}; median = {np.median(he_outputs):.8f}\n"
    )

    # Random Normal
    output += format("\n\nRandom Normal:\n")
    output += format(
        f"\tAccurracy: std = {np.std(rand_norm_acc):.8f}; mean = {np.mean(rand_norm_acc):.8f}; median = {np.median(rand_norm_acc):.8f}\n"
    )
    output += format(
        f"\tMean Difference: std = {np.std(rand_norm_diff):.8f}; mean = {np.mean(rand_norm_diff):.8f}; median = {np.median(rand_norm_diff):.8f}\n"
    )
    output += format(
        f"\tOutputs: std = {np.std(rand_norm_outputs):.8f}; variance = {np.var(rand_norm_outputs)}; mean = {np.mean(rand_norm_outputs):.8f}; median = {np.median(rand_norm_outputs):.8f}\n"
    )

    # Alternating
    output += format("\n\nAlternating positive negative:\n")
    output += format(
        f"\tAccurracy: std = {np.std(pos_neg_acc):.8f}; mean = {np.mean(pos_neg_acc):.8f}; median = {np.median(pos_neg_acc):.8f}\n"
    )
    output += format(
        f"\tMean Difference: std = {np.std(pos_neg_diff):.8f}; mean = {np.mean(pos_neg_diff):.8f}; median = {np.median(pos_neg_diff):.8f}\n"
    )
    output += format(
        f"\tOutputs: std = {np.std(pos_neg_outputs):.8f}; variance = {np.var(pos_neg_outputs)}; mean = {np.mean(pos_neg_outputs):.8f}; median = {np.median(pos_neg_outputs):.8f}\n"
    )

    return output


def run_tests(runs: int, show_image: bool = False):
    # Dataset - links training set; underscore discards the test set because we do not use it.
    (images, labels), (_, _) = mnist.load_data()

    # Normalisation
    images = images.astype("float32") / 255.0
    images = np.expand_dims(images, axis=-1)

    print(f"Shape of images: {images.shape}")
    print(f"Shape of labels: {labels.shape}")

    if show_image:
        image = images[50000]
        label = labels[50000]
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.show()

    activations = ["swish", "relu", "sigmoid", "tanh"]

    # Outputs of evaluations, see how `evaluate_with_activation` produces output
    outputs = []

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for activation in activations:
            pool.apply_async(
                evaluate_with_activation,
                args=(images, labels, runs, activation),
                callback=lambda x: outputs.append(x),
            )
        pool.close()
        pool.join()

    for output in outputs:
        print(output)


if __name__ == "__main__":
    # have an argument for number of runs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs to test the models."
    )
    parser.add_argument(
        "--show-img",
        action="store_true",
        help="Show a plot of example image.",
    )

    args = parser.parse_args()

    run_tests(args.runs, args.show_img)
