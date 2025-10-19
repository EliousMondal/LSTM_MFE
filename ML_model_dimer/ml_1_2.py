import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
    TimeDistributed, Attention, Concatenate, RepeatVector, LayerNormalization,
    MultiHeadAttention, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import glob
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import tempfile
import pickle
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Set a consistent style for plots
plt.style.use('default')


class WavefunctionLSTMPredictor:
 
     
    def __init__(self, lstm_units: int = 128, dropout_rate: float = 0.2, attention_heads: int = 3):
        
        # sequence_length is no longer a parameter.
        # It will be determined dynamically from the loaded data.
        self.sequence_length = None
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.attention_heads = attention_heads
        self.model: Optional[Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
 
        self.energy_scaler = RobustScaler()
        self.psi_real_scaler = RobustScaler()

    def load_wavefunction_data(self, filepath: str) -> Optional[np.ndarray]:
        
        #Loads wavefunction data from R1t_i.txt files, using only the first column.
        
        try:
             
            data = np.loadtxt(filepath)

            # Extract the first column
            if data.ndim == 1:
                psi_real = data  # The file is just a single column
            elif data.ndim >= 2:
                psi_real = data[:, 0]  # Take the first column
            else:
                print(f"Warning: Unexpected data dimension {data.ndim} in {filepath}")
                return None

            # Filter out NaN and infinite values
            psi_real = psi_real[np.isfinite(psi_real)]

            if len(psi_real) == 0:
                print(f"Warning: No valid real numbers found in the first column of {filepath}")
                return None

            # Remove extreme outliers using IQR method as explained in our papaer
            def remove_outliers(data, factor=3.0):
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                return np.clip(data, lower_bound, upper_bound)

            psi_real = remove_outliers(psi_real)

            return psi_real

        except Exception as e:
            print(f"Error loading wavefunction data from {filepath}: {e}")
            return None

    def load_energy_data(self, filepath: str) -> Optional[np.ndarray]:
        
        #MODIFIED: Loads energy data using the first two columns as separate features. 
        try:
            raw_data = np.loadtxt(filepath)

            #  Use the first two columns directly as inputs 
            data = None

            # Ensure the data is 2D and has at least two columns
            if raw_data.ndim == 2 and raw_data.shape[1] >= 2:
                # Use the first two columns as the two input features
                data = raw_data[:, :2]
            else:
                # If data doesn't meet the criteria above criteria then dont use
                print(f"Warning: Energy data in {filepath} does not have at least two columns. Shape is {raw_data.shape}. Skipping file.")
                return None
             

            # Filter out NaN and infinite values from the processed data
            mask = np.isfinite(data).all(axis=1)
            data = data[mask]

            if len(data) == 0:
                print(f"Warning: No valid energy data after filtering in {filepath}")
                return None

            return data

        except Exception as e:
            print(f"Error loading energy data from {filepath}: {e}")
            return None

    def get_available_trajectories_from_local(self, base_path: str) -> List[int]:
        #Scans a local directory for available trajectory data folders.
        trajectories = []
        if not os.path.isdir(base_path):
            print(f"Error: Directory not found at {base_path}")
            return []

        for folder_name in os.listdir(base_path):
            # Check if the folder name is a number (trajectory index)
            if folder_name.isdigit():
                traj_idx = int(folder_name)
                # Verify that the necessary files exist inside the folder
                energy_file = os.path.join(base_path, folder_name, f"energy_CPA_t_{traj_idx}_01.txt")
                psi_file = os.path.join(base_path, folder_name, f"R1t_{traj_idx}.txt")
                if os.path.exists(energy_file) and os.path.exists(psi_file):
                    trajectories.append(traj_idx)

        trajectories.sort()  # Ensure consistent order
        print(f"Found {len(trajectories)} trajectories in {base_path}")
        return trajectories

    def load_data_from_local(self, base_path: str, trajectory_indices: List[int]) -> Tuple[List, List]:
        
        # 
       # The full sequences are loaded 
         
        all_energy_data, all_psi_real_data = [], []

        for traj_idx in trajectory_indices:
            print(f"Loading trajectory {traj_idx} from local path...")
            energy_local = os.path.join(base_path, str(traj_idx), f"energy_CPA_t_{traj_idx}_01.txt")
            psi_local = os.path.join(base_path, str(traj_idx), f"R1t_{traj_idx}.txt")

            if os.path.exists(energy_local) and os.path.exists(psi_local):
                energy_data = self.load_energy_data(energy_local)
                psi_real = self.load_wavefunction_data(psi_local)

                if energy_data is not None and psi_real is not None:
                    all_energy_data.append(energy_data)
                    all_psi_real_data.append(psi_real)
                    print(f"Successfully loaded trajectory {traj_idx} with lengths: energy={len(energy_data)}, psi={len(psi_real)}")
                else:
                    print(f"Skipping trajectory {traj_idx}: failed to load data")
            else:
                print(f"Skipping trajectory {traj_idx}: data files not found.")
        
        print(f"Loaded {len(all_energy_data)} raw trajectories")
        return all_energy_data, all_psi_real_data

    def plot_loaded_data(self, energy_data: List[np.ndarray], psi_real_data: List[np.ndarray], save_path: str = 'loaded_data_plot.png'):
       #Plot one loaded data sample to verify it looks correct

        if not energy_data or not psi_real_data:
            print("No data to plot!")
            return
        
        # Use the first trajectory
        energy_sample = energy_data[0]
        psi_sample = psi_real_data[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Energy data (both columns if 2D)
        if energy_sample.ndim == 2:
            axes[0, 0].plot(energy_sample[:, 0], label='Energy Feature 1', color='blue')
            axes[0, 0].set_title('Energy Data - Column 1')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(energy_sample[:, 1], label='Energy Feature 2', color='red')
            axes[0, 1].set_title('Energy Data - Column 2')
            axes[0, 1].set_ylabel('Energy Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 0].plot(energy_sample, label='Energy', color='blue')
            axes[0, 0].set_title('Energy Data (1D)')
            axes[0, 0].set_ylabel('Energy Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].axis('off')
        
        # Psi real part
        axes[1, 0].plot(psi_sample, label='Real Part of Psi', color='green')
        axes[1, 0].set_title('Wavefunction Real Part')
        axes[1, 0].set_ylabel('Real(Psi)')
        axes[1, 0].set_xlabel('Position/Time Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        energy_stats = f"Energy Stats:\nMean: {np.mean(energy_sample):.4f}\nStd: {np.std(energy_sample):.4f}\nMin: {np.min(energy_sample):.4f}\nMax: {np.max(energy_sample):.4f}"
        psi_stats = f"Psi Real Stats:\nMean: {np.mean(psi_sample):.4f}\nStd: {np.std(psi_sample):.4f}\nMin: {np.min(psi_sample):.4f}\nMax: {np.max(psi_sample):.4f}"
        
        axes[1, 1].text(0.1, 0.7, energy_stats, transform=axes[1, 1].transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].text(0.1, 0.3, psi_stats, transform=axes[1, 1].transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_title('Data Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Data shape information:")
        print(f"Energy data shape: {energy_sample.shape}")
        print(f"Psi real data shape: {psi_sample.shape}")
        print(f"Sequence length: {len(psi_sample)}")

    def create_sequences(self, energy_data: List[np.ndarray], psi_real_data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        #Enhanced sequence creation with data augmentation. 
        X, y_real = [], []

        for i in range(len(energy_data)):
            # Basic sequence
            X.append(energy_data[i])
            y_real.append(psi_real_data[i])

            #  add noise to increase dataset size
            if len(energy_data) < 20:  # if we have limited data
                for noise_level in [0.01, 0.02]:
                    # Add Gaussian noise to energy data
                    energy_noisy = energy_data[i] + np.random.normal(0, noise_level * np.std(energy_data[i]), energy_data[i].shape)
                    psi_real_noisy = psi_real_data[i] + np.random.normal(0, noise_level * np.std(psi_real_data[i]), psi_real_data[i].shape)

                    X.append(energy_noisy)
                    y_real.append(psi_real_noisy)

        # Stack the sequences - y is now 1D  
        return np.array(X), np.array(y_real)

    def build_model(self, input_features: int = 2, output_features: int = 1) -> Model:
      
       #Enhanced one-to-one mapping model for real part prediction only.
        
       # Input layer
       inputs = Input(shape=(self.sequence_length, input_features), name='energy_inputs')

       # Multi-layer bidirectional LSTM stack
       x = inputs

       # First LSTM layer
       x = Bidirectional(
           LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate/2, kernel_regularizer=l2(1e-4)),
           name='lstm_layer_1'
       )(x)
       x = LayerNormalization()(x)

       # Second LSTM layer with residual connection
       lstm_2 = Bidirectional(
           LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate/2, kernel_regularizer=l2(1e-4)),
           name='lstm_layer_2'
       )(x)
       lstm_2 = LayerNormalization()(lstm_2)

       # Residual connection  
       if x.shape[-1] == lstm_2.shape[-1]:
           x = Add()([x, lstm_2])
       else:
           x = lstm_2

       # Multi-head self-attention for capturing long-range dependencies, we tried 1, 2, 3,5 heads
       attention = MultiHeadAttention(
           num_heads=self.attention_heads,
           key_dim=self.lstm_units//2,
           dropout=self.dropout_rate,
           name='self_attention'
       )(x, x)

       # Residual connection with attention
       x = Add()([x, attention])
       x = LayerNormalization()(x)

       # Feed-forward network for each time step
       ff = TimeDistributed(
           Dense(2 * self.lstm_units, activation='gelu', kernel_regularizer=l2(1e-4)),
           name='feedforward_1'
       )(x)
       ff = Dropout(self.dropout_rate)(ff)

       # Another residual connection
       x = Add()([x, ff])
       x = LayerNormalization()(x)

       # Second feed-forward layer  
       ff2 = TimeDistributed(
           Dense(self.lstm_units, activation='gelu', kernel_regularizer=l2(1e-4)),
           name='feedforward_2'
       )(x)
       ff2 = Dropout(self.dropout_rate/2)(ff2)

       # Output layer  only real part 
       outputs = TimeDistributed(
           Dense(output_features, activation='linear', kernel_regularizer=l2(1e-4)),
           name='real_wavefunction'
       )(ff2)

       model = Model(inputs=inputs, outputs=outputs)

       # Simplified loss function for real part prediction
       def real_wavefunction_loss(y_true, y_pred):
           # Basic MSE loss
           mse = tf.reduce_mean(tf.square(y_true - y_pred))

           # First derivative loss because it is smoothw ave
           grad_true = y_true[:, 1:] - y_true[:, :-1]
           grad_pred = y_pred[:, 1:] - y_pred[:, :-1]
           gradient_loss = tf.reduce_mean(tf.square(grad_true - grad_pred))

           # Second derivative loss again becuase of wave like behavior
           grad2_true = grad_true[:, 1:] - grad_true[:, :-1]
           grad2_pred = grad_pred[:, 1:] - grad_pred[:, :-1]
           curvature_loss = tf.reduce_mean(tf.square(grad2_true - grad2_pred))

           # Combine losses with appropriate weights, can play with it but this is good too
           total_loss = mse + 0.1 * gradient_loss + 0.05 * curvature_loss

           return total_loss

       # Compile model
       optimizer = Adam(
           learning_rate=1e-3,
           clipnorm=1.0,
           beta_1=0.9,
           beta_2=0.999,
           epsilon=1e-8
       )

       model.compile(
           optimizer=optimizer,
           loss=real_wavefunction_loss,
           metrics=['mae', 'mse']
       )

       self.model = model
       return model

    def fit_and_transform_data(self, energy_data, psi_real_data):
         
        # Flatten all data for fitting scalers
        all_energy = np.vstack(energy_data)
        all_psi_real = np.concatenate(psi_real_data).reshape(-1, 1)

        # Fit scalers
        self.energy_scaler.fit(all_energy)
        self.psi_real_scaler.fit(all_psi_real)

        return self.transform_data(energy_data, psi_real_data)

    def transform_data(self, energy_data, psi_real_data):
        #Transform data using fitted scalers. 
        scaled_energy, scaled_psi_real = [], []

        for i in range(len(energy_data)):
            # Transform energy data
            scaled_energy.append(self.energy_scaler.transform(energy_data[i]))

            # Transform wavefunction data
            psi_real_2d = psi_real_data[i].reshape(-1, 1)
            scaled_real = self.psi_real_scaler.transform(psi_real_2d).flatten()
            scaled_psi_real.append(scaled_real)

        return scaled_energy, scaled_psi_real

    def train(self, data_base_path: str, trajectory_indices: List[int], validation_split: float = 0.2,
              epochs: int = 200, batch_size: int = 16):
      
        print("Loading raw data from local path...")
        raw_energy_data, raw_psi_real_data = self.load_data_from_local(
            data_base_path, trajectory_indices
        )

        if not raw_energy_data:
            raise ValueError("No valid trajectory data was loaded. Aborting training.")

        # Determine a consistent sequence length by finding the minimum length across all files
        min_lengths = [min(len(e), len(p)) for e, p in zip(raw_energy_data, raw_psi_real_data)]
        if not min_lengths:
            raise ValueError("Could not determine sequence lengths from the loaded data.")

        self.sequence_length = min(min_lengths)
        print(f"Using consistent sequence length of {self.sequence_length} (minimum found across all trajectories).")

        # Trim all sequences to the determined uniform length
        energy_data = [e[:self.sequence_length] for e in raw_energy_data]
        psi_real_data = [p[:self.sequence_length] for p in raw_psi_real_data]

        print(f"Successfully processed {len(energy_data)} trajectories to uniform length.")
        
        # Plot one loaded data sample
        print("Plotting a sample of the processed data...")
        self.plot_loaded_data(energy_data, psi_real_data)
        
        print("Preprocessing and scaling data...")
        scaled_energy, scaled_psi_real = self.fit_and_transform_data(
            energy_data, psi_real_data
        )

        print("Creating training sequences...")
        X, y = self.create_sequences(scaled_energy, scaled_psi_real)

        if X.shape[0] == 0:
            raise ValueError("Sequence creation resulted in zero samples. Check data integrity.")

        print(f"Dataset shape: X={X.shape}, y={y.shape}")

        print("Building real part wavefunction model...")
        self.build_model(input_features=X.shape[2], output_features=1)
        print(f"Model has {self.model.count_params():,} parameters")

        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=30, restore_best_weights=True,
                verbose=1, min_delta=1e-6
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.7, patience=15, min_lr=1e-7,
                verbose=1, cooldown=5
            ),
            ModelCheckpoint(
                'best_real_wavefunction_model_401_1_2_2_48hour_3heads.h5', monitor='val_loss',
                save_best_only=True, verbose=1
            )
        ]

        # Determine batch size based on data size
        effective_batch_size = min(batch_size, max(1, X.shape[0] // 4))
        print(f"Using batch size: {effective_batch_size}")

        print("Starting real part wavefunction model training...")
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=effective_batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        return self.history

    def predict(self, energy_sequence: np.ndarray) -> np.ndarray:
        
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")

        # Handle input shape
        if len(energy_sequence.shape) == 2:
            energy_sequence = np.expand_dims(energy_sequence, axis=0)

        # Ensure correct sequence length
        if energy_sequence.shape[1] != self.sequence_length:
            if energy_sequence.shape[1] < self.sequence_length:
                # Pad the sequence
                padding = np.tile(energy_sequence[0, -1:, :],
                                  (1, self.sequence_length - energy_sequence.shape[1], 1))
                energy_sequence = np.concatenate([energy_sequence, padding], axis=1)
            else:
                # Truncate the sequence
                energy_sequence = energy_sequence[:, :self.sequence_length, :]

        # Scale input
        scaled_input = self.energy_scaler.transform(energy_sequence[0])
        scaled_input = np.expand_dims(scaled_input, axis=0)

        # Make prediction
        pred_scaled = self.model.predict(scaled_input, verbose=0)

        # Inverse transform
        pred_real = self.psi_real_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        return pred_real

    def evaluate_model(self, data_base_path: str, test_trajectory_indices: List[int]):
         
        print("\n Evaluating Real Part Wavefunction Model on Test Data ")
        test_energy, test_psi_real = self.load_data_from_local(
            data_base_path, test_trajectory_indices
        )
        
        if not test_energy:
            print("No test data found. Skipping evaluation.")
            return None, None
            
        # Trim test data to the same sequence length used for training
        test_energy = [e[:self.sequence_length] for e in test_energy]
        test_psi_real = [p[:self.sequence_length] for p in test_psi_real]


        scaled_energy, scaled_psi_real = self.transform_data(
            test_energy, test_psi_real
        )
        X_test, y_test_scaled = self.create_sequences(scaled_energy, scaled_psi_real)

        if X_test.shape[0] == 0:
            print("No valid test samples after processing. Skipping evaluation.")
            return None, None

        # Evaluate model
        results = self.model.evaluate(X_test, y_test_scaled, verbose=0)
        loss, mae, mse = results
        print(f"Test Loss: {loss:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test MSE: {mse:.6f}")

        # Generate predictions
        pred_scaled = self.model.predict(X_test, verbose=0)

        # Inverse transform predictions and true values
        predictions = self.psi_real_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
        y_test_original = self.psi_real_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test_scaled.shape)

        # Calculate additional metrics
        real_mse = mean_squared_error(y_test_original.flatten(), predictions.flatten())
        real_mae = mean_absolute_error(y_test_original.flatten(), predictions.flatten())

        print(f"Real part - MSE: {real_mse:.6f}, MAE: {real_mae:.6f}")

        return predictions, y_test_original

    def plot_predictions(self, predictions: np.ndarray, y_test: np.ndarray, save_path: str = 'real_predictions_plot.png'):
         
        n_samples = min(3, predictions.shape[0])
        fig, axes = plt.subplots(n_samples, 2, figsize=(16, 6 * n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Real Part Prediction vs Actual
            axes[i, 0].plot(y_test[i], label='Actual', color='blue', alpha=0.8, linewidth=2)
            axes[i, 0].plot(predictions[i], label='Predicted', color='red', linestyle='--', alpha=0.8, linewidth=2)
            axes[i, 0].set_title(f'Sample {i+1}: Real Part of Wavefunction', fontsize=14)
            axes[i, 0].set_ylabel('Real(Psi)', fontsize=12)
            axes[i, 0].set_xlabel('Position/Time Index')
            axes[i, 0].legend(fontsize=11)
            axes[i, 0].grid(True, alpha=0.3)

            # Residuals (Error)
            residuals = y_test[i] - predictions[i]
            axes[i, 1].plot(residuals, label='Residuals', color='green', alpha=0.8, linewidth=2)
            axes[i, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1}: Prediction Residuals', fontsize=14)
            axes[i, 1].set_ylabel('Actual - Predicted', fontsize=12)
            axes[i, 1].set_xlabel('Position/Time Index')
            axes[i, 1].legend(fontsize=11)
            axes[i, 1].grid(True, alpha=0.3)

            # Add statistics as text
            mse_sample = np.mean(residuals**2)
            mae_sample = np.mean(np.abs(residuals))
            axes[i, 1].text(0.05, 0.95, f'MSE: {mse_sample:.6f}\nMAE: {mae_sample:.6f}',
                          transform=axes[i, 1].transAxes, fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics
        for i in range(n_samples):
            residuals = y_test[i] - predictions[i]
            print(f"Sample {i+1} prediction stats:")
            print(f"  MSE: {np.mean(residuals**2):.6f}")
            print(f"  MAE: {np.mean(np.abs(residuals)):.6f}")
            print(f"  Max error: {np.max(np.abs(residuals)):.6f}")

    def plot_training_history(self, save_path: str = 'real_training_history.png'):
         
        if self.history is None:
            print("No training history available.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE', color='blue')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE', color='red')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Training MSE', color='blue')
        axes[1, 0].plot(self.history.history['val_mse'], label='Validation MSE', color='red')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate', color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model_and_scalers(self, model_path: str, scalers_path: str):
        #save it
        if self.model:
            self.model.save(model_path)
            print(f"Real part wavefunction model saved to {model_path}")

        scalers = {
            'energy': self.energy_scaler,
            'psi_real': self.psi_real_scaler,
            'sequence_length': self.sequence_length
        }
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {scalers_path}")

    def load_model_and_scalers(self, model_path: str, scalers_path: str):
        #load
        try:
            self.model = tf.keras.models.load_model(model_path)

            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)

            self.energy_scaler = scalers['energy']
            self.psi_real_scaler = scalers['psi_real']

            if 'sequence_length' in scalers:
                self.sequence_length = scalers['sequence_length']

            print(f"Real part wavefunction model and scalers loaded successfully from {model_path} and {scalers_path}")

        except Exception as e:
            print(f"Error loading model and scalers: {e}")


# Main execution block for real part prediction only
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

     
    # It will be determined automatically from the data.
    predictor = WavefunctionLSTMPredictor(
        lstm_units=128,
        dropout_rate=0.2,
        attention_heads=3
    )

    data_base_path = "/scratch/amiakhel/python_script/attention/datagen/LSTM_MFE/LSTM_MFE/PLDM_LS_CPA/model2_dimer/data800/Data"

    print("Detecting available trajectories from local directory...")
    try:
        all_trajectories = predictor.get_available_trajectories_from_local(data_base_path)
    except Exception as e:
        print(f"An error occurred while scanning local directories: {e}")
        all_trajectories = []

    if not all_trajectories:
        print("Could not auto-detect trajectories from local path. Using fallback list.")
        # Fallback if the directory scan fails
        all_trajectories = list(range(1, 11))

    print(f"Using {len(all_trajectories)} trajectories: {all_trajectories}")

    # Improved train-test split
    n_trajectories = len(all_trajectories)
    if n_trajectories >= 6:
        train_size = max(4, int(0.75 * n_trajectories))
    elif n_trajectories >= 3:
        train_size = n_trajectories - 1
    else:
        train_size = n_trajectories

    train_trajectories = all_trajectories[:train_size]
    test_trajectories = all_trajectories[train_size:] if train_size < n_trajectories else []

    print(f"Training trajectories ({len(train_trajectories)}): {train_trajectories}")
    print(f"Testing trajectories ({len(test_trajectories)}): {test_trajectories}")

    try:
        # Train the model for real part prediction
        print("\n Starting Real Part Wavefunction Training ")
        print("Key Feature: PREDICTING ONLY REAL PART OF WAVEFUNCTION")
        print("Using architecture: Energy[N] -> Real_Wavefunction[N] where N is determined from data.")
        
        predictor.train(
            data_base_path=data_base_path,
            trajectory_indices=train_trajectories,
            epochs=300,
            batch_size=16,
            validation_split=0.15
        )

        # Plot training history
        predictor.plot_training_history()

        # Save the model and scalers
        predictor.save_model_and_scalers(
            model_path="real_wavefunction_predictor_401_1_2_2_48hour_3heads.h5",
            scalers_path="real_wavefunction_scalers_401_1_2_2_48hour_3heads.pkl"
        )

        # Evaluate on test set if available
        if test_trajectories:
            print("\n Evaluating Real Part Model on Test Set ")
            predictions, y_test = predictor.evaluate_model(data_base_path, test_trajectories)
            if predictions is not None and y_test is not None:
                predictor.plot_predictions(predictions, y_test, 'real_test_predictions.png')
        else:
            print("\n No separate test set available, showing training predictions ")
            # Make predictions on a subset of training data for visualization
            energy_data, psi_real_data = predictor.load_data_from_local(
                data_base_path, train_trajectories[:2]
            )
            if energy_data:
                # Trim data to the learned sequence length
                energy_data = [e[:predictor.sequence_length] for e in energy_data]
                psi_real_data = [p[:predictor.sequence_length] for p in psi_real_data]

                scaled_energy, scaled_psi_real = predictor.transform_data(
                    energy_data, psi_real_data
                )
                X_sample, y_sample = predictor.create_sequences(scaled_energy, scaled_psi_real)

                if X_sample.shape[0] > 0:
                    pred_sample = predictor.model.predict(X_sample[:2])
                    pred_real = predictor.psi_real_scaler.inverse_transform(pred_sample.reshape(-1, 1)).reshape(pred_sample.shape)
                    y_real = predictor.psi_real_scaler.inverse_transform(y_sample[:2].reshape(-1, 1)).reshape(y_sample[:2].shape)

                    predictor.plot_predictions(pred_real, y_real, 'real_training_sample_predictions.png')

        print("\n Real Part Wavefunction Training Complete!!!!!!")
     
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()

        print("\nDebugging information:")
        print(f"Data Base Path: {data_base_path}")
        print(f"Available trajectories: {all_trajectories}")
        print(f"Training trajectories: {train_trajectories}")
        print("Check if the local data path is correct and the data format matches expectations.")


#  functions for real part analysis
def plot_wavefunction_real_properties(wavefunction_real: np.ndarray, title: str = "Real Wavefunction Properties"):
 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Real part
    axes[0, 0].plot(wavefunction_real, 'b-', linewidth=2)
    axes[0, 0].set_title('Real Part of Wavefunction')
    axes[0, 0].set_ylabel('Re[ψ]')
    axes[0, 0].grid(True, alpha=0.3)

    # Derivative (shows rate of change)
    derivative = np.gradient(wavefunction_real)
    axes[0, 1].plot(derivative, 'r-', linewidth=2)
    axes[0, 1].set_title('First Derivative')
    axes[0, 1].set_ylabel("d(Re[ψ])/dx")
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of values
    axes[1, 0].hist(wavefunction_real, bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Distribution of Values')
    axes[1, 0].set_xlabel('Re[ψ]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics
    stats_text = f"""Statistics:
Mean: {np.mean(wavefunction_real):.4f}
Std: {np.std(wavefunction_real):.4f}
Min: {np.min(wavefunction_real):.4f}
Max: {np.max(wavefunction_real):.4f}
Range: {np.ptp(wavefunction_real):.4f}"""
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=12,
                  verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Statistics')
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"\nReal Wavefunction Statistics for {title}:")
    print(f"  Mean value: {np.mean(wavefunction_real):.6f}")
    print(f"  Standard deviation: {np.std(wavefunction_real):.6f}")
    print(f"  Value range: [{np.min(wavefunction_real):.6f}, {np.max(wavefunction_real):.6f}]")

