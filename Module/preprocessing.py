import torch
import numpy as np
from scipy.signal import butter, filtfilt, stft

def processing(data, fs=100.0):
    """
    Preprocessing function for time-series data of robot force and torque sensors.
    
    This function performs the following steps:
    1. Low-pass filtering on Fx, Fy, Fz, Tx, Ty, Tz.
    2. Computation of force vector rate of change (dF/dt).
    3. Computation of force vector angle change (angle_F) between consecutive timesteps.
    4. Computation of torque vector rate of change (dT/dt).
    5. Computation of torque vector angle change (angle_T) between consecutive timesteps.
    6. Computation of angle between force vector and torque vector (angle_FT).
    7. STFT-based frequency domain feature extraction for **all channels** (Fx, Fy, Fz, Tx, Ty, Tz), 
       where each channel’s mean spectral power is computed and added as a feature.
    8. Min-Max normalization of all features.

    **Original Input Features (after low-pass filtering)**:
    - Fx, Fy, Fz: Force components along x, y, z axes.
    - Tx, Ty, Tz: Torque components along x, y, z axes.
    (Total 6 features)

    **Newly Added Features**:
    - dF/dt (3 features): Rate of change of force vector (Fx, Fy, Fz).
    - angle_F (1 feature): Angle change between force vectors at consecutive timesteps.
    - angle_FT (1 feature): Angle between force and torque vector at each timestep.
    - dT/dt (3 features): Rate of change of torque vector (Tx, Ty, Tz).
    - angle_T (1 feature): Angle change between torque vectors at consecutive timesteps.
    - STFT features (6 features): For each channel (Fx, Fy, Fz, Tx, Ty, Tz), 
      mean spectral energy computed from STFT is added as a constant feature across time.
    
    **Final Feature Count**:
    - Original: 6 (Fx, Fy, Fz, Tx, Ty, Tz)
    - dF/dt: +3
    - angle_F: +1
    - angle_FT: +1
    - dT/dt: +3
    - angle_T: +1
    - STFT features (all 6 channels): +6
    --------------------------------------
    Total = 6 + 3 + 1 + 1 + 3 + 1 + 6 = 21 features per timestep.

    Args:
        data (torch.Tensor): Input data of shape (sequence_length, input_dim).
                             Input order assumed: [Fx, Fy, Fz, Tx, Ty, Tz]
        fs (float, optional): Sampling frequency in Hz. Default is 100.0.

    Returns:
        torch.Tensor: Processed data of shape (sequence_length, 21), where all features
                      are min-max normalized to [0,1].
    """
    data_np = data.detach().cpu().numpy()  # (seq_length, input_dim)
    
    # Set index for each channel
    Fx_idx, Fy_idx, Fz_idx = 0, 1, 2
    Tx_idx, Ty_idx, Tz_idx = 3, 4, 5

    # Low-pass filter parameters
    cutoff = 10.0  # Hz
    order = 4
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply low-pass filter to each channel
    for ch_idx in [Fx_idx, Fy_idx, Fz_idx, Tx_idx, Ty_idx, Tz_idx]:
        data_np[:, ch_idx] = filtfilt(b, a, data_np[:, ch_idx])

    # Force and Torque vectors
    F = data_np[:, [Fx_idx, Fy_idx, Fz_idx]]  # (seq_length, 3)
    T = data_np[:, [Tx_idx, Ty_idx, Tz_idx]]  # (seq_length, 3)

    # dF/dt (Force vector rate of change)
    dF = np.diff(F, axis=0)
    dF = np.vstack([dF, np.zeros((1, 3))])  # zero padding at last timestep

    # Force vector angle change (angle_F)
    F_norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    F_next = np.roll(F, -1, axis=0)
    F_next_norm = np.linalg.norm(F_next, axis=1, keepdims=True) + 1e-8
    dot_F = np.sum(F * F_next, axis=1, keepdims=True)
    cos_theta_F = dot_F / (F_norm * F_next_norm)
    angle_F = np.arccos(np.clip(cos_theta_F, -1.0, 1.0))
    angle_F[-1, :] = 0.0

    # dT/dt (Torque vector rate of change)
    dT = np.diff(T, axis=0)
    dT = np.vstack([dT, np.zeros((1, 3))])  # zero padding at last timestep

    # Torque vector angle change (angle_T)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    T_next = np.roll(T, -1, axis=0)
    T_next_norm = np.linalg.norm(T_next, axis=1, keepdims=True) + 1e-8
    dot_T = np.sum(T * T_next, axis=1, keepdims=True)
    cos_theta_T = dot_T / (T_norm * T_next_norm)
    angle_T = np.arccos(np.clip(cos_theta_T, -1.0, 1.0))
    angle_T[-1, :] = 0.0

    # Angle between force vector and torque vector (angle_FT)
    dot_FT = np.sum(F * T, axis=1, keepdims=True)
    cos_FT = dot_FT / (F_norm * T_norm)
    angle_FT = np.arccos(np.clip(cos_FT, -1.0, 1.0))

    # STFT for all channels
    # Compute STFT-based feature (mean power) for each channel and stack them
    stft_features_list = []
    for ch_idx in [Fx_idx, Fy_idx, Fz_idx, Tx_idx, Ty_idx, Tz_idx]:
        f, t_stft, Zxx = stft(data_np[:, ch_idx], fs=fs, nperseg=32, noverlap=16)
        stft_power = np.mean(np.abs(Zxx)**2)
        # Create a column of stft_power repeated for all timesteps
        stft_feature_ch = np.ones((data_np.shape[0], 1)) * stft_power
        stft_features_list.append(stft_feature_ch)
    # Concatenate all channel STFT features (seq_length, 6)
    stft_features = np.hstack(stft_features_list)

    # Stack all extra features
    # Already existing extra features: dF(3), angle_F(1), angle_FT(1)
    # New extra features: dT(3), angle_T(1), and stft_features(6 channels)
    # Total new extra_features dimension: dF(3) + angle_F(1) + angle_FT(1) + dT(3) + angle_T(1) + stft_features(6) = 3+1+1+3+1+6 = 15 features
    extra_features = np.hstack([dF, angle_F, angle_FT, dT, angle_T, stft_features])  # (seq_length, 15)

    # Combine original data and extra features
    processed_data = np.hstack([data_np, extra_features])  # original 6 + 15 = 21 features total

    # Min-Max Normalization
    min_vals = processed_data.min(axis=0, keepdims=True)
    max_vals = processed_data.max(axis=0, keepdims=True)
    processed_data = (processed_data - min_vals) / (max_vals - min_vals + 1e-8)

    return torch.tensor(processed_data, dtype=torch.float32) # Convert to tensor
