import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def create_mnist_datasets():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    num_train = 60000
    num_val_test = 10000
    nsteps = 5
    x_dim = 784  # 28*28 flattened image
    a_dim = 1  # Dummy action dimension
    r_dim = 1  # Dummy reward dimension
    mask_dim = 1
    u_dim = 2  # Updated to 2 to match the model's expectation

    def flatten_image(image):
        return image.flatten()

    def collect_data(images, num_samples):
        states = []
        actions = []
        rewards = []
        masks = []
        rich = []

        for i in range(num_samples):
            state = flatten_image(images[i % len(images)])
            ep_states = []
            ep_actions = []
            ep_rewards = []
            ep_masks = []
            ep_rich = []

            for _ in range(nsteps):
                action = np.random.randint(0, 2)  # Random dummy action (0 or 1)
                reward = np.random.random()  # Random dummy reward

                ep_states.append(state)
                ep_actions.append([action])
                ep_rewards.append([reward])
                ep_masks.append([1.0])
                ep_rich.append([np.random.randint(0, 2), np.random.randint(0, 2)])  # Generate two random Bernoulli values for rich

            states.append(ep_states)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            masks.append(ep_masks)
            rich.append(ep_rich)

        return np.array(states), np.array(actions), np.array(rewards), np.array(masks), np.array(rich)

    # Collect training, validation, and testing data
    x_train, a_train, r_train, mask_train, rich_train = collect_data(x_train, num_train)
    x_validation, a_validation, r_validation, mask_validation, rich_validation = collect_data(x_test, num_val_test)
    x_test, a_test, r_test, mask_test, rich_test = collect_data(x_test, num_val_test)

    # Save datasets as npz files
    np.savez('mnist_training_data.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train, rich_train=rich_train)
    np.savez('mnist_validation_data.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation, rich_validation=rich_validation)
    np.savez('mnist_testing_data.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test, rich_test=rich_test)

    print("MNIST datasets created and saved.")

create_mnist_datasets()
