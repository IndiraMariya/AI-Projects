import tensorflow as tf

# Load the MNIST dataset
# (x_train, y_train) = tf.keras.datasets.mnist.load_data(path='dataset/mnist_train.csv')
# (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='dataset/mnist_test.csv')

mnist_dataset = tf.keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = mnist_dataset

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the 2D images into 1D vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert the labels to categorical format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
