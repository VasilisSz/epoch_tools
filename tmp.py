import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pickle


def load_dlc(path):
    df = pd.read_csv(path, header=[1,2])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

# --------------------------------
# Feauture extraction functions
# --------------------------------
def distance_moved(df, bodypoint):
    """
    Calculate the distance moved by a given bodypoint based on consecutive frames.
    
    Parameters:
      df : pd.DataFrame
          DataFrame containing columns: f"{bodypoint}_x", f"{bodypoint}_y", f"{bodypoint}_likelihood"
      bodypoint : str
          Name of the bodypoint.
          
    Returns:
      distance : pd.Series
          Euclidean distance moved between consecutive frames.
      distance_likelihood : pd.Series
          For each frame, the minimum likelihood of the two frames used to compute the difference.
    """
    x = df[f"{bodypoint}_x"]
    y = df[f"{bodypoint}_y"]
    likelihood = df[f"{bodypoint}_likelihood"]
    
    # Compute differences between consecutive frames
    dx = x.diff()
    dy = y.diff()
    
    # Euclidean distance moved between frames
    distance = np.sqrt(dx**2 + dy**2)
    
    # For likelihood, take the minimum likelihood between the current and previous frame.
    distance_likelihood = pd.concat([likelihood, likelihood.shift(1)], axis=1).min(axis=1)
    
    return distance, distance_likelihood

def calculate_speed(df, bodypoint, time_interval=1.0):
    """
    Calculate the speed of a given bodypoint based on consecutive frames.
    
    Parameters:
      df : pd.DataFrame
          DataFrame containing columns: f"{bodypoint}_x", f"{bodypoint}_y", f"{bodypoint}_likelihood"
      bodypoint : str
          Name of the bodypoint.
      time_interval : float, optional
          Time difference between frames (default is 1.0).
          
    Returns:
      speed : pd.Series
          Euclidean speed (distance per time_interval).
      speed_likelihood : pd.Series
          For each frame, the minimum likelihood of the two frames used to compute the difference.
    """
    x = df[f"{bodypoint}_x"]
    y = df[f"{bodypoint}_y"]
    likelihood = df[f"{bodypoint}_likelihood"]
    
    # Compute differences between consecutive frames
    dx = x.diff()
    dy = y.diff()
    
    # Euclidean distance divided by the time interval gives speed.
    speed = np.sqrt(dx**2 + dy**2) / time_interval
    
    # For likelihood, take the minimum between the current and previous frame’s likelihood.
    speed_likelihood = pd.concat([likelihood, likelihood.shift(1)], axis=1).min(axis=1)
    
    return speed, speed_likelihood

def calculate_area(df, bodypoints):
    """
    Calculate the area of the polygon defined by the given list of bodypoints using the shoelace formula.
    
    Parameters:
      df : pd.DataFrame
          DataFrame containing columns for each bodypoint:
          f"{bodypoint}_x", f"{bodypoint}_y", f"{bodypoint}_likelihood"
      bodypoints : list of str
          List of bodypoint names. The order in the list defines the order of the polygon vertices.
          
    Returns:
      area : np.array
          The polygon area computed for each frame.
      area_likelihood : np.array
          The minimum likelihood among the provided bodypoints for each frame.
    """
    # Stack the x and y coordinates for all bodypoints; shape is (n_frames, n_points)
    xs = np.column_stack([df[f"{bp}_x"].values for bp in bodypoints])
    ys = np.column_stack([df[f"{bp}_y"].values for bp in bodypoints])
    
    # Compute “shifted” coordinates (cyclically) to apply the shoelace formula.
    xs_shift = np.roll(xs, shift=-1, axis=1)
    ys_shift = np.roll(ys, shift=-1, axis=1)
    
    # Shoelace formula for polygon area
    area = 0.5 * np.abs(np.sum(xs * ys_shift - xs_shift * ys, axis=1))
    
    # For likelihood, take the minimum likelihood over all the selected bodypoints.
    likelihoods = np.column_stack([df[f"{bp}_likelihood"].values for bp in bodypoints])
    area_likelihood = np.min(likelihoods, axis=1)
    
    return area, area_likelihood

def calculate_length(df, point_pair):
    """
    Calculate the Euclidean distance between two bodypoints.
    
    Parameters:
      df : pd.DataFrame
          DataFrame containing columns: f"{bodypoint}_x", f"{bodypoint}_y", f"{bodypoint}_likelihood"
      point_pair : tuple of str
          Tuple containing two bodypoint names, e.g. ("center", "nose").
          
    Returns:
      length : pd.Series
          Euclidean distance between the two points.
      length_likelihood : pd.Series
          The minimum likelihood between the two bodypoints for each frame.
    """
    bp1, bp2 = point_pair
    x1 = df[f"{bp1}_x"]
    y1 = df[f"{bp1}_y"]
    x2 = df[f"{bp2}_x"]
    y2 = df[f"{bp2}_y"]
    
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    length_likelihood = pd.concat([df[f"{bp1}_likelihood"], df[f"{bp2}_likelihood"]], axis=1).min(axis=1)
    
    return length, length_likelihood

def calculate_angular_change(df, center_bodypoint, target_bodypoint):
    """
    Calculate the angular change (delta angle) of the vector from a center bodypoint to a target bodypoint over time.
    
    Parameters:
      df : pd.DataFrame
          DataFrame containing columns: f"{bodypoint}_x", f"{bodypoint}_y", f"{bodypoint}_likelihood"
      center_bodypoint : str
          Name of the center (reference) bodypoint.
      target_bodypoint : str
          Name of the target bodypoint.
          
    Returns:
      angular_change : np.array
          The change in angle (in radians) between consecutive frames.
      angular_likelihood : np.array
          For each frame, the minimum likelihood computed from the center and target bodypoints (considering consecutive frames).
    """
    # Compute the angle (in radians) of the vector from the center to the target.
    dx = df[f"{target_bodypoint}_x"] - df[f"{center_bodypoint}_x"]
    dy = df[f"{target_bodypoint}_y"] - df[f"{center_bodypoint}_y"]
    angle = np.arctan2(dy, dx)
    
    # Compute the change in angle between consecutive frames.
    d_angle = np.diff(angle)
    # Adjust the differences to be within [-pi, pi] (account for wrap-around)
    d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi
    # Insert a NaN at the beginning so that the result has the same length as the input.
    angular_change = np.insert(d_angle, 0, np.nan)
    
    # For the likelihood, take the minimum of the two likelihoods for the center and target,
    # and then compute a rolling minimum over two consecutive frames.
    pair_lik = pd.concat([df[f"{center_bodypoint}_likelihood"], df[f"{target_bodypoint}_likelihood"]], axis=1).min(axis=1)
    angular_likelihood = pair_lik.rolling(window=2, min_periods=1).min().values
    
    return angular_change, angular_likelihood

def interpolate_low_likelihood(df, p_cutoff, method='linear'):
    """
    For each feature column in the DataFrame that has a corresponding likelihood column 
    (i.e. column name + '_likelihood'), this function sets the feature values to NaN when 
    the likelihood is below a specified threshold (p_cutoff) and then interpolates these values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and their corresponding likelihood columns.
    p_cutoff : float
        The likelihood threshold below which the feature values are considered unreliable.
    method : str, optional
        Interpolation method to use (default is 'linear'). Other methods (like 'spline', etc.) 
        are supported by pandas.
        
    Returns
    -------
    df_interpolated : pd.DataFrame
        A new DataFrame where each feature column (with an associated likelihood column) has 
        been interpolated at points where the likelihood is below p_cutoff.
    """
    df_interpolated = df.copy()
    # Loop over all columns that are considered features (i.e. do not end with '_likelihood')
    for col in df.columns:
        if col.endswith('_likelihood'):
            continue  # Skip likelihood columns themselves
        likelihood_col = col + '_likelihood'
        if likelihood_col in df.columns:
            # Identify rows with low likelihood values
            low_mask = df_interpolated[likelihood_col] < p_cutoff
            # Replace low-likelihood feature values with NaN
            df_interpolated.loc[low_mask, col] = np.nan
            # Interpolate missing values; limit_direction='both' ensures that missing values at 
            # the beginning or end of the series are also filled if possible.
            df_interpolated[col] = df_interpolated[col].interpolate(method=method, limit_direction='both')
    return df_interpolated

def window_sum(x):
    return np.sum(x)

def window_mean(x):
    return np.mean(x)

def window_median(x):
    return np.median(x)

def window_min(x):
    return np.min(x)

def window_max(x):
    return np.max(x)

def window_variance(x):
    return np.var(x)

def window_sd(x):
    return np.std(x)

def window_slope(x):
    """
    Compute the slope of the values in x by fitting a first-degree polynomial (line).
    """
    if len(x) < 2:
        return np.nan
    # Use np.polyfit with x-coordinates as the index positions
    slope, _ = np.polyfit(np.arange(len(x)), x, 1)
    return slope

def window_autocorrelation(x):
    """
    Compute the lag-1 autocorrelation of the series x.
    """
    if len(x) < 2:
        return np.nan
    x = np.array(x)
    x1 = x[:-1]
    x2 = x[1:]
    # If the standard deviation is zero, autocorrelation is undefined (return NaN)
    if np.std(x1) == 0 or np.std(x2) == 0:
        return np.nan
    return np.corrcoef(x1, x2)[0, 1]

def window_entropy(x, bins=10):
    """
    Compute the Shannon entropy of the values in x using histogram binning.
    
    Parameters
    ----------
    x : array-like
        The data over which entropy is computed.
    bins : int, optional
        Number of bins to use for the histogram (default is 10).
        
    Returns
    -------
    entropy : float
        The computed Shannon entropy (in bits).
    """
    x = np.array(x)
    counts, _ = np.histogram(x, bins=bins, density=True)
    # Convert counts to a probability distribution (avoid division by zero)
    if np.sum(counts) == 0:
        return 0.0
    p = counts / np.sum(counts)
    # Remove zero probabilities to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# Map statistic names to functions for convenience.


def sliding_window_features(df, window_size=None, step_size=None, stat_methods='all', features=None, window_pairs=None):
    """
    Applies windowing to each feature in the DataFrame and computes descriptive statistics 
    over each window. The computed statistics for each feature are returned in a new DataFrame 
    where each row corresponds to one window.
    
    There are two modes of operation:
    
    1. Sliding Window Mode:
       If `window_pairs` is None, a sliding window is applied using `window_size` and `step_size`.
       For each window, the start index is recorded as `window_start` and the end index as `window_end`
       (which is `start + window_size - 1`).
       
    2. Custom Windows Mode:
       If `window_pairs` is provided as a 2-dimensional array-like object (each row is [start_frame, end_frame]),
       each pair defines a window. Pairs containing NaN values are skipped. The window is assumed to include
       the frame at `end_frame` (i.e. slicing uses `df.iloc[start:end+1]`).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric feature columns.
    window_size : int, optional
        The number of samples in each sliding window (ignored if window_pairs is provided).
    step_size : int, optional
        The number of samples to move the window at each step (ignored if window_pairs is provided).
    stat_methods : list of str or 'all'
        List of statistical methods to compute. Supported methods include:
        "sum", "mean", "median", "min", "max", "variance", "sd", "slope", "autocorrelation", "entropy".
    features : list of str, optional
        List of column names to process. If None, all numeric columns are used (excluding columns 
        ending in '_likelihood').
    window_pairs : array-like, shape (n_windows, 2), optional
        A 2-dimensional array-like object where each row is a pair [start_frame, end_frame]. 
        If provided, these pairs are used to define the windows. Pairs containing NaN values are skipped.
    
    Returns
    -------
    stats_df : pd.DataFrame
        A DataFrame where each row corresponds to a window. Columns are named as "feature_stat" 
        (e.g. "nose_mean", "center_slope", etc.). Two extra columns, 'window_start' and 'window_end', 
        indicate the indices of the original DataFrame that define the window.
    """

    # If features are not provided, use all numeric columns except those that look like likelihood columns.
    if features is None:
        features = [col for col in df.columns 
                    if np.issubdtype(df[col].dtype, np.number) and not col.endswith('_likelihood')]
    
    # Dictionary mapping statistic names to functions (assumed to be defined elsewhere)
    stat_funcs = {
        'sum': window_sum,
        'mean': window_mean,
        'median': window_median,
        'min': window_min,
        'max': window_max,
        'variance': window_variance,
        'sd': window_sd,
        'slope': window_slope,
        'autocorrelation': window_autocorrelation,
        'entropy': window_entropy,
        # Additional statistics can be added here.
    }
    if stat_methods == 'all':
        stat_methods = list(stat_funcs.keys())

    windows = []
    
    if window_pairs is not None:
        # Use provided window pairs
        window_pairs = np.asarray(window_pairs)
        for pair in window_pairs:
            # Skip pair if any element is NaN
            if np.isnan(np.array(pair)).any():
                continue

            start, end = int(pair[0]), int(pair[1])
            # Assume the window includes the end_frame, so we slice until end+1.
            window_data = df.iloc[start:end+1]
            stats_dict = {'window_start': start, 'window_end': end}
            
            for feature in features:
                # Remove missing values before computing statistics.
                series = window_data[feature].dropna().values
                for stat in stat_methods:
                    if stat in stat_funcs:
                        # For entropy we need to pass the number of bins (defaulting to 10)
                        if stat == 'entropy':
                            value = stat_funcs[stat](series, bins=10)
                        else:
                            value = stat_funcs[stat](series)
                        stats_dict[f"{feature}_{stat}"] = value
                    else:
                        raise ValueError(f"Statistic method '{stat}' is not supported.")
            windows.append(stats_dict)
    else:
        # Use sliding window mode: require both window_size and step_size
        if window_size is None or step_size is None:
            raise ValueError("Either window_pairs must be provided or both window_size and step_size must be provided.")
        
        n_rows = df.shape[0]
        for start in range(0, n_rows - window_size + 1, step_size):
            end = start + window_size
            window_data = df.iloc[start:end]
            stats_dict = {'window_start': start, 'window_end': end - 1}
            
            for feature in features:
                series = window_data[feature].dropna().values
                for stat in stat_methods:
                    if stat in stat_funcs:
                        if stat == 'entropy':
                            value = stat_funcs[stat](series, bins=10)
                        else:
                            value = stat_funcs[stat](series)
                        stats_dict[f"{feature}_{stat}"] = value
                    else:
                        raise ValueError(f"Statistic method '{stat}' is not supported.")
            windows.append(stats_dict)
    
    stats_df = pd.DataFrame(windows)
    return stats_df
