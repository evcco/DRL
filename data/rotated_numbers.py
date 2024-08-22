import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def create_mnist_datasets():
    (x_train, _), (x_test, _) = mnist.load_data()
    
    num_train = 60000
    num_val_test = 10000
    nsteps = 5  # Number of steps (actions) per episode
    x_dim = 784  # 28x28 flattened image
    a_dim = 1  # Action dimension (rotate clockwise or counterclockwise)
    r_dim = 1  # Reward dimension
    mask_dim = 1  # Mask dimension

    def flatten_image(image):
        return image.flatten()

    def rotate_image(image, angle):
        """Rotate image by angle (0, 90, 180, 270)."""
        if angle == 0:
            return image
        elif angle == 90:
            return np.rot90(image, 1)
        elif angle == 180:
            return np.rot90(image, 2)
        elif angle == 270:
            return np.rot90(image, 3)
        else:
            raise ValueError("Invalid angle for rotation.")

    def collect_data(images, num_samples):
        states = []
        actions = []
        rewards = []
        masks = []

        for i in range(num_samples):
            initial_angle = np.random.choice([0, 90, 180, 270])
            image = rotate_image(images[i % len(images)], initial_angle)
            state = initial_angle  # Initial state is the angle

            ep_states = []
            ep_actions = []
            ep_rewards = []
            ep_masks = []

            for _ in range(nsteps):
                action = np.random.choice([-90, 90])  # Rotate by -90 (counterclockwise) or 90 (clockwise)
                next_state = (state + action) % 360  # Update state (angle)

                ep_states.append([state])
                ep_actions.append([action])

                # Check if the new state is the correct orientation (0°)
                if next_state == 0:
                    reward = 1.0
                else:
                    reward = 0.0
                
                ep_rewards.append([reward])
                ep_masks.append([1.0])

                state = next_state  # Move to the next state

            states.append(ep_states)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            masks.append(ep_masks)

        return np.array(states), np.array(actions), np.array(rewards), np.array(masks)

    # Collect training, validation, and testing data
    x_train, a_train, r_train, mask_train = collect_data(x_train, num_train)
    x_validation, a_validation, r_validation, mask_validation = collect_data(x_test, num_val_test)
    x_test, a_test, r_test, mask_test = collect_data(x_test, num_val_test)

    # Save datasets as npz files
    np.savez('mnist_training_data_rl.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train)
    np.savez('mnist_validation_data_rl.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation)
    np.savez('mnist_testing_data_rl.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test)

    print("MNIST RL datasets created and saved.")

create_mnist_datasets()
