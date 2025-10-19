import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import pickle
from tqdm import tqdm  # For progress bars

 #check if the file already exist
try:
    from ml_1_2 import WavefunctionLSTMPredictor
except ImportError:
    print("Error: Could not import 'WavefunctionLSTMPredictor' from 'ml_1_2.py'.")
    print("Please make sure 'ml_1_2.py' is in the same folder as this script.")
    sys.exit(1)

 
# we define our custome loss function
def real_wavefunction_loss(y_true, y_pred):
  
 
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # First derivative loss same as trainging
    grad_true = y_true[:, 1:] - y_true[:, :-1]
    grad_pred = y_pred[:, 1:] - y_pred[:, :-1]
    gradient_loss = tf.reduce_mean(tf.square(grad_true - grad_pred))

    # Second derivative loss same as tringing
    grad2_true = grad_true[:, 1:] - grad_true[:, :-1]
    grad2_pred = grad_pred[:, 1:] - grad_pred[:, :-1]
    curvature_loss = tf.reduce_mean(tf.square(grad2_true - grad2_pred))

    # Combine losses with appropriate weightssame as traingng
    total_loss = mse + 0.1 * gradient_loss + 0.05 * curvature_loss

    return total_loss


def run_predictions():
 
 
    #
 
    #
    # Path to the saved Keras model file.
    MODEL_PATH = "best_real_wavefunction_model_401_1_2_2_48hour_3heads.h5"
    # Path to the saved scalers file.
    SCALERS_PATH = "real_wavefunction_scalers_401_1_2_2.pkl"
    # Base directory where the trajectory data folders (e.g., '1', '2', '3') are located.
    DATA_BASE_PATH = "/scratch/amiakhel/python_script/attention/datagen/LSTM_MFE/LSTM_MFE/PLDM_LS_CPA/model2_dimer/data800/Data"
    # Range of trajectories to process
    TRAJECTORY_RANGE = list(range(1, 14500))  # 1 to whatever
    # Directory to save individual predictions
    PREDICTIONS_DIR = "predictions_saved"
    # Directory to save final results
    OUTPUT_DIR = "final_results"
    # Batch size for processing  
    BATCH_SIZE = 1000
	# so we know what file is running
   
    print("Large-Scale Wavefunction Prediction Script")
    print(f"Processing {len(TRAJECTORY_RANGE)} trajectories in batches of {BATCH_SIZE}")

    # Create output directories
    for directory in [PREDICTIONS_DIR, OUTPUT_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    # Check if the model and scaler files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALERS_PATH):
        print(f"\nError: Model ('{MODEL_PATH}') or scalers ('{SCALERS_PATH}') not found.")
        print("Please ensure the trained model and scaler files are in the same directory as this script.")
        return

 
    try:
        predictor = WavefunctionLSTMPredictor()

        # Load the scalers first from the pickle file
        with open(SCALERS_PATH, 'rb') as f:
            scalers = pickle.load(f)
        predictor.energy_scaler = scalers['energy']
        predictor.psi_real_scaler = scalers['psi_real']
        predictor.sequence_length = scalers.get('sequence_length')

        # Define the custom objects Keras needs to know about
        custom_objects = {'real_wavefunction_loss': real_wavefunction_loss}

        # Load the model within the custom object scope
        with tf.keras.utils.custom_object_scope(custom_objects):
            predictor.model = tf.keras.models.load_model(MODEL_PATH)

        print("Model and scalers loaded successfully.")
        print(f"Model expects sequences of length: {predictor.sequence_length}")

    except Exception as e:
        print(f"An error occurred while loading the model or scalers: {e}")
        import traceback
        traceback.print_exc()
        return

    
    
    all_predictions = []
    all_actual_values = []
    successful_trajectories = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(TRAJECTORY_RANGE), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(TRAJECTORY_RANGE))
        current_batch = TRAJECTORY_RANGE[batch_start:batch_end]
        
        print(f"\n  Processing batch {batch_start//BATCH_SIZE + 1}: trajectories {current_batch[0]} to {current_batch[-1]}")
        
        # Load data for current batch
        try:
            energy_data_raw, psi_real_data_raw = predictor.load_data_from_local(
                DATA_BASE_PATH, current_batch
            )
        except Exception as e:
            print(f"Warning: Error loading batch data: {e}")
            continue
        
        if not energy_data_raw:
            print(f"Warning: No data loaded for batch starting at trajectory {current_batch[0]}")
            continue
        
        # Process each trajectory in the batch
        batch_predictions = []
        batch_actual = []
        
        for i, traj_idx in enumerate(current_batch):
            if i >= len(energy_data_raw):  # check so it matches 
                break
                
            try:
                energy_sequence = energy_data_raw[i]
                actual_psi_sequence = psi_real_data_raw[i]
                
                # Make prediction
                predicted_psi = predictor.predict(energy_sequence)
                
                # Trim actual data to match prediction length
                comparison_length = predictor.sequence_length
                actual_psi_trimmed = actual_psi_sequence[:comparison_length]
                
                # Save individual prediction
                pred_filename = os.path.join(PREDICTIONS_DIR, f"prediction_{traj_idx}.npy")
                np.save(pred_filename, predicted_psi)
                
                # Save individual actual values
                actual_filename = os.path.join(PREDICTIONS_DIR, f"actual_{traj_idx}.npy")
                np.save(actual_filename, actual_psi_trimmed)
                
                # Store for averaging
                batch_predictions.append(predicted_psi)
                batch_actual.append(actual_psi_trimmed)
                successful_trajectories.append(traj_idx)
                
            except Exception as e:
                print(f"Warning: Error processing trajectory {traj_idx}: {e}")
                continue
        
        # Add batch results to overall collection
        all_predictions.extend(batch_predictions)
        all_actual_values.extend(batch_actual)
        
        print(f"Completed batch: {len(batch_predictions)} successful predictions")
        
        # Clean up memory
        del energy_data_raw, psi_real_data_raw
        del batch_predictions, batch_actual

     
    if len(all_predictions) == 0:
        print("Error: No successful predictions were made. Please check your data paths and format.")
        return

    
    all_predictions = np.array(all_predictions)
    all_actual_values = np.array(all_actual_values)
    
    # Calculate elementwise averages
    average_prediction = np.mean(all_predictions, axis=0)
    average_actual = np.mean(all_actual_values, axis=0)
    
    print(f"Average prediction shape: {average_prediction.shape}")
    print(f"Average actual shape: {average_actual.shape}")

 
    
    # Save average prediction
    avg_pred_path = os.path.join(OUTPUT_DIR, "average_prediction.npy")
    np.save(avg_pred_path, average_prediction)
    
    # Also save as text file for future mmaybe
    avg_pred_txt_path = os.path.join(OUTPUT_DIR, "average_prediction.txt")
    np.savetxt(avg_pred_txt_path, average_prediction, fmt='%.8f')
    
    # Save average actual values
    avg_actual_path = os.path.join(OUTPUT_DIR, "average_actual.npy")
    np.save(avg_actual_path, average_actual)
    
    # Also save as text file for later
    avg_actual_txt_path = os.path.join(OUTPUT_DIR, "average_actual.txt")
    np.savetxt(avg_actual_txt_path, average_actual, fmt='%.8f')
    
    print(f"Saved average prediction to: {avg_pred_path}")
    print(f"Saved average actual values to: {avg_actual_path}")
v
    
    plt.style.use('default')
    plt.figure(figsize=(16, 8))
    
    # Plot both averages
    time_indices = np.arange(len(average_actual))
    plt.plot(time_indices, average_actual, label='Average Actual Wavefunction', 
             color='blue', linewidth=2.5, alpha=0.8)
    plt.plot(time_indices, average_prediction, label='Average Predicted Wavefunction', 
             color='red', linestyle='--', linewidth=2.5, alpha=0.9)
    
    plt.title(f'Average Wavefunction Comparison\n(Based on {len(successful_trajectories)} trajectories)', 
              fontsize=18, fontweight='bold')
    plt.xlabel('Position/Time Index', fontsize=14)
    plt.ylabel('Real(Psi)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_plot_path = os.path.join(OUTPUT_DIR, "average_comparison_plot_50k.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(comparison_plot_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    
    print(f"Saved comparison plot to: {comparison_plot_path}")
    
    # Show the plot
    plt.show()

 
    
    # Calculate Mean Squared Error
    mse = np.mean((average_prediction - average_actual) ** 2)
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(average_prediction - average_actual))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(average_prediction, average_actual)[0, 1]
    
    print(f"\nComparison Statistics:")
    print(f"  Mean Squared Error (MSE): {mse:.8f}")
    print(f"  Mean Absolute Error (MAE): {mae:.8f}")
    print(f"  Correlation Coefficient: {correlation:.6f}")
    
    # Save statistics to file
    stats_path = os.path.join(OUTPUT_DIR, "comparison_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("Wavefunction Prediction Comparison Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total trajectories processed: {len(successful_trajectories)}\n")
        f.write(f"Successful predictions: {len(all_predictions)}\n")
        f.write(f"Sequence length: {len(average_prediction)}\n\n")
        f.write("Comparison Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.8f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.8f}\n")
        f.write(f"  Correlation Coefficient: {correlation:.6f}\n\n")
        f.write("Successful trajectory IDs:\n")
        f.write(f"{successful_trajectories}\n")
    
    print(f"Saved statistics to: {stats_path}")
    print("\n Large-scale prediction process complete!")
    print(f"\nSummary:")
    print(f"  - Processed {len(successful_trajectories)} out of {len(TRAJECTORY_RANGE)} trajectories")
    print(f"  - Individual predictions saved to: {PREDICTIONS_DIR}/")
    print(f"  - Average results saved to: {OUTPUT_DIR}/")
    print(f"  - Comparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    run_predictions()

