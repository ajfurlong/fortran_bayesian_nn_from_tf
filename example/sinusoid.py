import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import h5py
import time

def generate_data(num_samples=5000):
    """Generate dummy data for a transformed sinusoid."""
    tf.keras.utils.set_random_seed(1234)

    # Generate x1 and x2 values (arbitrary transformation)
    x1 = np.linspace(0, 6 * np.pi, num_samples)
    x2 = np.linspace(3, 6, num_samples)
    
    # Define a transformed sinusoid function with two inputs
    y = 100 * (np.sin(x1) + x2) + 0.1 * np.random.randn(num_samples)
    
    # Combine x1 and x2 into a single input array
    x = np.column_stack((x1, x2))
    y = y.reshape(-1, 1)
    
    # Shuffle the data
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    return x, y

tfd = tfp.distributions
tfb = tfp.bijectors

def negloglik(y, distr): 
  return -distr.log_prob(y) 

def normal_sp(params): 
  return tfd.Normal(loc=params[:,0:1], scale=1e-6 + tf.math.softplus(params[:,1:2]))

def train_model(x_train, y_train, epochs=500, batch_size=32):
    """Create and fit the two-layer DNN model."""

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x_train.shape[0] * 1.0)
    bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (x_train.shape[0] * 1.0)

    # Define the model with two input features
    inputs = layers.Input(shape=(x_train.shape[1],))
    hidden = tfp.layers.DenseFlipout(16,
                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,
                                     activation='relu')(inputs)
    hidden = tfp.layers.DenseFlipout(16,
                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,
                                     activation='relu')(hidden)
    hidden = tfp.layers.DenseFlipout(16,
                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn,
                                     activation='relu')(hidden)
    params = tfp.layers.DenseFlipout(2,
                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params)

    model = tf.keras.Model(inputs=inputs, outputs=dist)
    
    # Compile the model
    model.compile(optimizer='adam', loss=negloglik)
    
    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def compute_metrics(y_true, y_pred, y_pred_sigma, inference_time):
    epsilon = 1e-6
    relative_error = 100 * np.abs((y_true.flatten() - y_pred.flatten()) / (y_true.flatten() + epsilon))
    absolute_error = np.abs(y_pred.flatten() - y_true.flatten())
    ferr_above_10 = np.sum(relative_error > 10.0)
    rrmse = np.sqrt(np.mean(((y_pred.flatten() - y_true.flatten()) / (y_true.flatten() + epsilon)) ** 2))
    rStd = (y_pred_sigma.flatten() / np.abs(y_pred.flatten() + epsilon)) * 100

    metrics = {
        "MAE": np.mean(absolute_error),
        "Max AE": np.max(absolute_error),
        "Min AE": np.min(absolute_error),
        "Std AE": np.std(absolute_error),
        "Mean Sample Vector Std": np.mean(y_pred_sigma),
        "Max Sample Vector Std": np.max(y_pred_sigma),
        "MAPE": np.mean(relative_error),
        "Max APE": np.max(relative_error),
        "Min APE": np.min(relative_error),
        "Std APE": np.std(relative_error),
        "Mean Sample Vector rStd (%)": np.mean(rStd),
        "Max Sample Vector rStd (%)": np.max(rStd),
        "rRMSE (%)": rrmse * 100,
        "Ferr > 10% (%)": 100 * ferr_above_10 / len(relative_error),
        "R^2": r2_score(y_true.flatten(), y_pred.flatten()),
        "Inference Time (s)": inference_time
    }
    return metrics

# Save data and model
def save_data_and_model(x_test_rescaled, y_test_rescaled, y_pred_rescaled, y_pred_sigma_rescaled, model,
                        x_scaler, y_scaler, time_elapsed,
                        data_file='../data/sinusoid_data_test.h5',
                        model_file='../models/sinusoid_model_tf.h5',
                        metadata_file='../models/sinusoid_metadata_tf.h5'):
    """Save the test data and model."""
    model.save(model_file)
    print(f"Model has been saved to {model_file}")

    with h5py.File(metadata_file, 'w') as f:
        grp1 = f.create_group("scaler")
        grp1.create_dataset("x_mean", data=x_scaler.mean_)
        grp1.create_dataset("y_mean", data=y_scaler.mean_)
        grp1.create_dataset("x_std", data=x_scaler.scale_)
        grp1.create_dataset("y_std", data=y_scaler.scale_)
    print(f"Model metadata has been saved to {metadata_file}")

    test_data = np.column_stack((x_test_rescaled, 
                                y_test_rescaled.flatten(), 
                                y_pred_rescaled.flatten(), 
                                y_pred_sigma_rescaled.flatten()))
    headers = ["input1", "input2", "output_true", "output_pred", "output_pred_sigma"]

    with h5py.File(data_file, 'w') as f:
        for i, header in enumerate(headers):
            f.create_dataset(header, data=test_data[:, i])
        f.create_dataset("inference_time", data=time_elapsed)
    print(f"Data has been saved to {data_file}")

def main():
    # Generate data
    x, y = generate_data()
    
    # Split the data into training and test sets
    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Standardize the input data
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    # Standardize the output data
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Train the model
    model = train_model(x_train, y_train)

    # For timing statistics
    start_time = time.time()

    # Set the number of prediction samples to generate using .predict()
    num_samples = 20000

    # Create and fill performance arrays with prediction values
    y_pred_samples = np.zeros((y_test.shape[0], num_samples))
    y_pred_mean_rescaled = np.zeros(y_test.shape)
    y_pred_sigma_rescaled = np.zeros(y_test.shape)

    for i in range(num_samples):
        temp = model.predict(x_test, verbose=0)
        temp_rescaled = y_scaler.inverse_transform(temp.reshape(-1, 1)).flatten()
        y_pred_samples[:, i] = temp_rescaled

    # Compute the mean and standard deviation of the adjusted predictions
    y_pred_mean_rescaled = np.mean(y_pred_samples, axis=1)
    y_pred_mean_rescaled = y_pred_mean_rescaled.reshape(-1, 1)
    y_pred_sigma_rescaled = np.std(y_pred_samples, axis=1)
    y_true_rescaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    end_time = time.time()

    x_test_rescaled = x_scaler.inverse_transform(x_test)
    y_test_rescaled = y_scaler.inverse_transform(y_test)

    # Compute metrics
    metrics = compute_metrics(y_test_rescaled, y_pred_mean_rescaled, y_pred_sigma_rescaled, end_time - start_time)

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(metrics, index=['Values']).T

    # Print metrics table
    print(metrics_df)

    # Save results table
    with open('results_sinusoid_tf_bnn.txt', 'w') as f:
        f.write(metrics_df.to_string(float_format='%.6f'))

    # Print the means and standard deviations used for standardization
    print("Input means:", x_scaler.mean_)
    print("Output mean:", y_scaler.mean_)
    print("Input standard deviations:", x_scaler.scale_)
    print("Output standard deviation:", y_scaler.scale_)
    
    # Save the test data and model
    save_data_and_model(x_test_rescaled, y_test_rescaled, y_pred_mean_rescaled, y_pred_sigma_rescaled,
                        model, x_scaler, y_scaler, end_time - start_time)

if __name__ == "__main__":
    main()