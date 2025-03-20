import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Constants
latent_dim = 100      # Dimensionality of the generator input
num_examples_to_generate = 16  # Number of images to generate
generator = None
discriminator = None
gan = None

def build_generator():
    model = Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))  # Assuming input images are 28x28 pixels
    model.add(layers.Reshape((28, 28, 1)))
    return model

def build_discriminator():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def compile_models(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer='adam')

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.normal(0, 1, size=(n_samples, latent_dim))
    return x_input

def generate_fake_samples(generator, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def generate_real_samples(n_samples):
    # For example: Using MNIST dataset
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)  # Expand to (samples, 28, 28, 1)
    indices = np.random.randint(0, X_train.shape[0], n_samples)
    X = X_train[indices]
    y = np.ones((n_samples, 1))
    return X, y

def train_gan(epochs, batch_size):
    for _ in range(epochs):
        # Generate real and fake samples
        X_real, y_real = generate_real_samples(batch_size)
        X_fake, y_fake = generate_fake_samples(generator, batch_size)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(X_real, y_real)
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)

        # Create points in latent space
        x_gan = generate_latent_points(latent_dim, batch_size)
        y_gan = np.ones((batch_size, 1))

        # Train the generator
        g_loss = gan.train_on_batch(x_gan, y_gan)

        if _ % 10 == 0:
            print(f"Epoch: {_}, Discriminator Loss: {d_loss_real[0]}, Generator Loss: {g_loss}")

def plot_generated_images(generator, n_images=16):
    noise = generate_latent_points(latent_dim, n_images)
    generated_images = generator.predict(noise)
    plt.figure(figsize=(4, 4))
    for i in range(n_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    compile_models(generator, discriminator)

    # Start training GAN
    train_gan(epochs=100, batch_size=128)

    # Plot generated images after training
    plot_generated_images(generator)