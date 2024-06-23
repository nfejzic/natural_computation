import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

# Datensatz - links trainingset - rechts das testset mit underscore, da es nicht gebraucht wird und nicht
# verwendet wird im folgenden...
(images, labels), (_, _) = mnist.load_data()

# Normalisierung
images = images.astype("float32") / 255.0
images = np.expand_dims(images, axis=-1)

print(f"Shape of images: {images.shape}")
print(f"Shape of labels: {labels.shape}")

# Das hier auskommentieren, um Beispiele der Bilder zu sehen...bedenke labels und images sollten
# den gleichen index haben, damit sie zusammenpassen...
# image = images[50000]
# label = labels[50000]
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(f"Label: {label}")
# plt.show()


######################### MODEL ##################################
def create_model():
    model = Sequential(
        [
            # Convo Layers damits a CNN is...
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Erzeugung 1-Dim Array - Schichten erfordert 1-Dim Eingabe
            Flatten(input_shape=(28, 28)),
            # Dense -> Jedes Neuron mit jedem verbunden
            # mit kernel_initializer kann Schicht auch direkt initalisiert werden, statt die funktionen zu nehmen
            Dense(
                32,
                activation=tf.nn.silu,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
            ),
            Dense(32, activation=tf.nn.relu),
            Dense(32, activation=tf.nn.silu),
            Dense(32, activation=tf.nn.relu),
            # Softmax -> Summe der Outputs erzeugt 1. Es sind 10 Outputs, da 10 Ziffern im Dataset...
            Dense(10, activation="softmax"),
        ]
    )
    return model


######################### Initialisierungsfunktionen ##################################


def initialize_weights_xavier(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf.keras.initializers.GlorotUniform()
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


def initialize_weights_he(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf.keras.initializers.HeNormal()
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


def initialize_weights_random_normal(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
            new_weights = initializer(layer.get_weights()[0].shape)
            new_biases = np.zeros(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])


# Hab ich bissi herumgespielt seit den tests..müsste vlt nochmal kontrolliert und angepasst werden..
def initialize_weights_alternating(model):
    positive_initializer = tf.keras.initializers.RandomUniform(
        minval=0.0, maxval=1.0, seed=None
    )
    negative_initializer = tf.keras.initializers.RandomUniform(
        minval=-1.0, maxval=0.0, seed=None
    )

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
            layer, tf.keras.layers.Dense
        ):
            if i % 2 == 0:
                new_weights = positive_initializer(layer.get_weights()[0].shape)
            else:
                new_weights = negative_initializer(layer.get_weights()[0].shape)
            bias_initializer = tf.keras.initializers.RandomUniform(
                minval=-0.01, maxval=0.01
            )
            new_biases = bias_initializer(layer.get_weights()[1].shape)
            layer.set_weights([new_weights, new_biases])

            # Debug prints...kann ma ignorieren
            # print(f"Layer {i} weights initialized with {'positive' if i % 2 == 0 else 'negative'} values:")
            # print("new biases:")
            # print(new_biases)
            # print("new weights:")
            # print(new_weights)


######################### Auswertung ##################################


def evaluate_initialization(model, images, labels, initialization_fn):
    initialization_fn(model)
    # am ersten bild wird predicted damit das Model eingerichtet ist für den durchlauf
    _ = model.predict(images[:1])
    outputs = model.predict(images)
    predicted_labels = np.argmax(outputs, axis=1)
    correct_predictions = np.sum(predicted_labels == labels)
    total_predictions = labels.shape[0]
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy without training: {accuracy:.2f}%")

    class_probs = outputs[np.arange(len(labels)), labels]
    non_class_probs = outputs[np.arange(len(labels)), :].copy()
    non_class_probs[np.arange(len(labels)), labels] = np.nan

    # Hier wird REIHENweise addiert und dann durch 9 dividiert - da 1 von 10 eine "nan" ist.
    # Anschließend wird diese i-te Reihe/Zeile zum element der i-ten Spalte in non_class_probs.
    # Damit non_class_probs gleiche struktur hat wie class_probs
    non_class_probs = np.nanmean(non_class_probs, axis=1)

    class_mean = np.mean(class_probs)
    non_class_mean = np.mean(non_class_probs)
    mean_difference = abs(class_mean - non_class_mean)
    print(f"Mean difference between class and non-class values: {mean_difference:.8f}")

    return accuracy, mean_difference


xavier_acc, xavier_diff = ([], [])
he_acc, he_diff = ([], [])
rand_norm_acc, rand_norm_diff = ([], [])
pos_neg_acc, pos_neg_diff = ([], [])

for i in range(5):
    # Um das erzeugte Model zu sehen (also ein print davon), die model.summary() zeile auskommentieren...
    print("Testing Xavier Initialization")
    model = create_model()
    # model.summary()
    accuracy_xavier, mean_diff_xavier = evaluate_initialization(
        model, images, labels, initialize_weights_xavier
    )
    xavier_acc.append(accuracy_xavier)
    xavier_diff.append(mean_diff_xavier)
    print(
        "###########################################################################\n"
    )

    print("\nTesting He Initialization")
    model = create_model()
    # model.summary()
    accuracy_he, mean_diff_he = evaluate_initialization(
        model, images, labels, initialize_weights_he
    )
    he_acc.append(accuracy_he)
    he_diff.append(mean_diff_he)
    print(
        "###########################################################################\n"
    )

    print("\nTesting Random Normal Initialization")
    model = create_model()
    # model.summary()
    accuracy_random_normal, mean_diff_random_normal = evaluate_initialization(
        model, images, labels, initialize_weights_random_normal
    )
    rand_norm_acc.append(accuracy_random_normal)
    rand_norm_diff.append(mean_diff_random_normal)
    print(
        "###########################################################################\n"
    )

    print("Testing Alternating Positive/Negative Initialization")
    model = create_model()
    # model.summary()
    accuracy_alternating, mean_diff_alternating = evaluate_initialization(
        model, images, labels, initialize_weights_alternating
    )
    pos_neg_acc.append(accuracy_alternating)
    pos_neg_diff.append(mean_diff_alternating)
    print(
        "###########################################################################\n"
    )


print("Xavier:")
print(
    f"\tAccurracy: std = {np.std(xavier_acc):.8f}; mean = {np.mean(xavier_acc):.8f}; median = {np.median(xavier_acc):.8f}"
)
print(
    f"\tMean Difference: std = {np.std(xavier_diff):.8f}; mean = {np.mean(xavier_diff):.8f}; median = {np.median(xavier_diff):.8f}"
)

print("\nHe:")
print(
    f"\tAccurracy: std = {np.std(he_acc):.8f}; mean = {np.mean(he_acc):.8f}; median = {np.median(he_acc):.8f}"
)
print(
    f"\tMean Difference: std = {np.std(he_diff):.8f}; mean = {np.mean(he_diff):.8f}; median = {np.median(he_diff):.8f}"
)

print("\nRandom Normal:")
print(
    f"\tAccurracy: std = {np.std(rand_norm_acc):.8f}; mean = {np.mean(rand_norm_acc):.8f}; median = {np.median(rand_norm_acc):.8f}"
)
print(
    f"\tMean Difference: std = {np.std(rand_norm_diff):.8f}; mean = {np.mean(rand_norm_diff):.8f}; median = {np.median(rand_norm_diff):.8f}"
)

print("\nAlternating positive negative:")
print(
    f"\tAccurracy: std = {np.std(pos_neg_acc):.8f}; mean = {np.mean(pos_neg_acc):.8f}; median = {np.median(pos_neg_acc):.8f}"
)
print(
    f"\tMean Difference: std = {np.std(pos_neg_diff):.8f}; mean = {np.mean(pos_neg_diff):.8f}; median = {np.median(pos_neg_diff):.8f}"
)
