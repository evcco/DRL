import numpy as np

def create_dummy_datasets():
    num_train = 140000
    num_val_test = 28000
    nsteps = 5
    x_dim = 784
    a_dim = 1
    r_dim = 1
    mask_dim = 1
    u_dim = 1  # Assuming u_dim for rich_train

    # Create training data
    x_train = np.random.rand(num_train, nsteps, x_dim).astype(np.float32)
    a_train = np.random.rand(num_train, nsteps, a_dim).astype(np.float32)
    r_train = np.random.rand(num_train, nsteps, r_dim).astype(np.float32)
    mask_train = np.ones((num_train, nsteps, mask_dim), dtype=np.float32)
    rich_train = np.random.rand(num_train, nsteps, u_dim).astype(np.float32)  # Dummy rich_train

    # Create validation data
    x_validation = np.random.rand(num_val_test, nsteps, x_dim).astype(np.float32)
    a_validation = np.random.rand(num_val_test, nsteps, a_dim).astype(np.float32)
    r_validation = np.random.rand(num_val_test, nsteps, r_dim).astype(np.float32)
    mask_validation = np.ones((num_val_test, nsteps, mask_dim), dtype=np.float32)
    rich_validation = np.random.rand(num_val_test, nsteps, u_dim).astype(np.float32)  # Dummy rich_validation

    # Create testing data
    x_test = np.random.rand(num_val_test, nsteps, x_dim).astype(np.float32)
    a_test = np.random.rand(num_val_test, nsteps, a_dim).astype(np.float32)
    r_test = np.random.rand(num_val_test, nsteps, r_dim).astype(np.float32)
    mask_test = np.ones((num_val_test, nsteps, mask_dim), dtype=np.float32)
    rich_test = np.random.rand(num_val_test, nsteps, u_dim).astype(np.float32)  # Dummy rich_test

    # Save datasets as npz files
    np.savez('dummy_training_data.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train, rich_train=rich_train)
    np.savez('dummy_validation_data.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation, rich_validation=rich_validation)
    np.savez('dummy_testing_data.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test, rich_test=rich_test)

    print("Dummy datasets created and saved.")

create_dummy_datasets()
