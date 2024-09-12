import os
import random
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from scipy.stats import wasserstein_distance

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)


# Normalization values
mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]

# Function to normalize the images
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 and normalize to [0, 1]
    image = (image - mean) / std  # Normalize with mean and std
    label = tf.one_hot(label, 10)
    label = tf.squeeze(label)  # Remove the extra dimension
    return image, label

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Obtain validation set (1/5 of train data to be equal to size of test data)
rng = np.random.default_rng(seed=42)
val_inds = rng.choice(np.arange(len(train_images)), size=len(train_images)//5, replace=False)
train_inds = np.delete(np.arange(len(train_images)), val_inds)

train_images, val_images = train_images[train_inds], train_images[val_inds]
train_labels, val_labels = train_labels[train_inds], train_labels[val_inds]

# Convert to tf.data.Dataset and apply preprocessing
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

train_dataset = train_dataset.shuffle(buffer_size=1024, seed=42).map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

batch_size = 128
train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

class CNN(models.Model):

    def __init__(self, out_features, activation_type='selu', selu_dtype=tf.float32):
        super(CNN, self).__init__()
        self.out_features = out_features
        self.selu_dtype = selu_dtype
        self.activation_type = activation_type
        activation = layers.Activation(activation_type, dtype=selu_dtype)

        self.net = models.Sequential([
            layers.Conv2D(128, kernel_size=5, padding='same', input_shape=(32, 32, 3), kernel_initializer='lecun_normal',bias_initializer='zeros'),
            layers.MaxPooling2D(pool_size=2),
            activation,
            layers.Conv2D(128, kernel_size=3, kernel_initializer='lecun_normal',bias_initializer='zeros'),
            layers.MaxPooling2D(pool_size=2),
            activation,
            layers.Conv2D(128, kernel_size=3, kernel_initializer='lecun_normal',bias_initializer='zeros'),
            activation,
            layers.Flatten(),
            layers.Dense(2048, kernel_initializer='lecun_normal',bias_initializer='zeros'),
            activation,
            layers.Dense(1024, kernel_initializer='lecun_normal',bias_initializer='zeros'),
            activation,
            layers.Dense(out_features),
            # layers.Activation('softmax')

        ])
    def call(self, x):
        # tensorflow dtype is different from Pytorch, just call is OK 
        return self.net(x)
    def get_config(self):
        # Return the configuration of the model
        config = super(CNN, self).get_config()
        config.update({
            "out_features": self.out_features,
            "activation_type": self.activation_type,
            'selu_dtype': self.selu_dtype
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from its configuration
        return cls(**config)

def accuracy(y_true, y_pred):
    y_pred = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    y_true = tf.argmax(y_true, axis=1)
    return accuracy_score(y_true.numpy(), y_pred.numpy())
def calculate_weight_max_difference(model1: models.Model, model2: models.Model) -> float:
    max_distance = 0.0
    for weight1, weight2 in zip(model1.trainable_variables, model2.trainable_variables):
        # Convert tensors to numpy arrays
        weight1_np = weight1.numpy().flatten()
        weight2_np = weight2.numpy().flatten()
        distance = np.max(np.abs(weight1_np-weight2_np))
        if distance>max_distance:
            max_distance = distance
    return max_distance
def calculate_weight_difference(model1: models.Model, model2: models.Model) -> float:
    distance = 0.0
    for weight1, weight2 in zip(model1.trainable_variables, model2.trainable_variables):
        # Convert tensors to numpy arrays
        weight1_np = weight1.numpy().flatten()
        weight2_np = weight2.numpy().flatten()
        
        # Calculate Wasserstein distance
        # distance += wasserstein_distance(weight1_np, weight2_np)
        distance += mean_squared_error(weight1_np, weight2_np)
    return distance

def _forward(network, dataset, metric):

    for x, y in dataset:
        y_hat = network(x, training=False)
        loss = metric(y, y_hat)
        yield loss

def update_with_weight_diff(network1: models.Model, network2: models.Model, dataset, 
                            loss_fn, opt1, opt2) -> list:

    errs1 = []
    errs2 = []
    weight_diffs = []
    update_time = 0

    for x, y in dataset:
        # Forward and update pass for the first network (float32)
        with tf.GradientTape() as tape1:
            y_hat1 = network1(x, training=True)
            err1 = loss_fn(y, y_hat1)
        grads1 = tape1.gradient(err1, network1.trainable_variables)
        opt1.apply_gradients(zip(grads1, network1.trainable_variables))
        errs1.append(err1.numpy())  # Append error to errs list

        # Forward pass for the second network (bfloat16)
        with tf.GradientTape() as tape2:
            y_hat2 = network2(x, training=True)
            err2 = loss_fn(y, y_hat2)
        grads2 = tape2.gradient(err2, network2.trainable_variables)
        opt2.apply_gradients(zip(grads2, network2.trainable_variables))
        errs2.append(err2.numpy())
        
        update_time += 1

        # Calculate weight differences every 20 updates
        # if (update_time % 20) == 0:
            
        #     # weight_diff = calculate_weight_max_difference(network1, network2)
        #     weight_diff = calculate_weight_difference(network1, network2)
        #     weight_diff_max = calculate_weight_max_difference(network1,network2)
        #     weight_diffs.append(weight_diff)
        #     print(f'weight mse diff: {weight_diff}')
        #     print(f'weight max diff: {weight_diff_max}')

    return errs1, errs2, weight_diffs

def update(network, dataset, loss_fn, optimizer):

    errs = []
    for x, y in dataset:
        with tf.GradientTape() as tape:
            y_hat = network(x, training=True)
            loss = loss_fn(y, y_hat)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
        errs.append(loss)
    return errs

def evaluate(network, dataset, metric):
    performance = []
    for loss in _forward(network, dataset, metric):
        performance.append(loss)
    return np.mean(performance)
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import copy

def fit_with_weight_diff(network1: models.Model, network2: models.Model, trainloader, 
                         valloader, testloader, epochs: int, lr: float):
    # Initialize optimizers
    optimizer1 = tf.keras.optimizers.SGD(learning_rate=lr)
    optimizer2 = tf.keras.optimizers.SGD(learning_rate=lr)
    
    # Loss function
    ce = losses.CategoricalCrossentropy(from_logits=True)
    

    train_losses, val_losses, accuracies, weight_diffs = [], [], [], []

    # Evaluate performance before training
    val_losses.append(evaluate(network=network1, dataset=valloader, metric=ce))

    pbar = tqdm(range(epochs))
    best_model = None

    for ep in pbar:
        # Update networks
        tl1, tl2, wd = update_with_weight_diff(network1=network1, network2=network2, dataset=trainloader, 
                                         loss_fn=ce, opt1=optimizer1, opt2=optimizer2)
        train_losses.extend(tl1)
        weight_diffs.extend(wd)

        # Validation loss
        vl = evaluate(network=network1, dataset=valloader, metric=ce)
        val_losses.append(vl)
        
        # Validation accuracy
        ac = evaluate(network=network1, dataset=valloader, metric=accuracy)

        accuracies.append(ac)

        print(f"train loss: {round(np.mean(tl1), 4):.4f}, "
              f"train loss: {round(np.mean(tl2), 4):.4f}, "
              f"val loss: {round(vl, 4):.4f}, "
              f"accuracy: {round(ac * 100, 2):.2f}%, ")

        pbar.set_description_str(desc=f"Epoch {ep+1}")

    # Final test accuracy
    acc = evaluate(network=best_model, dataset=testloader, metric=accuracy)

    return train_losses, val_losses, accuracies, acc, weight_diffs

def fit(network, train_dataset, val_dataset, test_dataset, epochs, lr):

    optimizer = optimizers.SGD(learning_rate=lr)
    ce = losses.CategoricalCrossentropy(from_logits=True)
    
    train_losses, val_losses, accuracies = [], [], []

    val_losses.append(evaluate(network=network, dataset=val_dataset, metric=ce))

    pbar = tqdm(range(epochs))
    for ep in pbar:
        tl = update(network=network, dataset=train_dataset, loss_fn=ce, optimizer=optimizer)
        train_losses.extend(tl)
        vl = evaluate(network=network, dataset=val_dataset, metric=ce)
        val_losses.append(vl)
        ac = evaluate(network=network, dataset=val_dataset, metric=accuracy)

        if len(accuracies) == 0 or ac > max(accuracies):
            best_model = models.clone_model(network)
            best_model.set_weights(network.get_weights())

        accuracies.append(ac)

        print(f"train loss: {round(np.mean(tl), 4):.4f}, "
              f"val loss: {round(vl, 4):.4f}, "
              f"accuracy: {round(ac * 100, 2):.2f}%")

        pbar.set_description_str(desc=f"Epoch {ep+1}")

    acc = evaluate(network=best_model, dataset=test_dataset, metric=accuracy)

    print(f"Final accuracy on test set: {round(acc*100, 2):.2f}%")

    return train_losses, val_losses, accuracies, acc

import sys
epochs = 20
lr = sys.argv[1]

# SELU training
network1 = CNN(out_features=10, activation_type = 'selu')
network2 = CNN(out_features=10, activation_type ='selu', selu_dtype=tf.bfloat16)
network2.set_weights(network1.get_weights())

stl, svl, saccs, sacc, weight_diffs= fit_with_weight_diff(network1, network2, train_dataset, val_dataset, test_dataset, epochs, lr)
