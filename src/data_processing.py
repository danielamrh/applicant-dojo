"""
Core Data Processing Functions for FDSE Challenge

CANDIDATE TASK: Implement the three functions below according to their specifications.

These functions form the core of an industrial data processing pipeline.
You will work with real-world challenges like missing data, connection failures,
and noisy sensor readings.

IMPORTANT NOTES:
- Function signatures (names, parameters, return types) must not be changed
- You may add helper functions in this file or create new modules
- Focus on robustness, error handling, and data quality
- Document your assumptions and trade-offs in NOTES.md
- Aim for production-quality code, not just passing tests
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def ingest_data(
    data_batches: List[pd.DataFrame],
    validate: bool = True,
) -> pd.DataFrame:
    """
    Ingest and consolidate multiple batches of industrial sensor data.
    
    This function must handle real-world data quality issues:
    - Missing or null values
    - Duplicate readings
    - Out-of-order timestamps
    - Data from different sensors with different units
    - Potentially empty batches
    
    Args:
        data_batches: List of DataFrames, each with columns:
            - timestamp (datetime): When the reading was taken
            - sensor (str): Sensor identifier (e.g., "temperature", "pressure")
            - value (float): Sensor reading (may be NaN)
            - unit (str): Unit of measurement
            - quality (str): Data quality flag ("GOOD", "BAD", "UNCERTAIN")
        validate: If True, perform data validation and cleanup
    
    Returns:
        Consolidated DataFrame with cleaned, deduplicated, and sorted data.
        Should maintain all original columns plus any derived quality metrics.
    
    Raises:
        ValueError: If data_batches is empty or contains invalid data structures
    
    Example:
        >>> batches = simulator.get_batch_readings(num_batches=5)
        >>> clean_data = ingest_data(batches, validate=True)
        >>> print(f"Ingested {len(clean_data)} readings from {len(batches)} batches")
    
    CANDIDATE TODO:
    - Implement robust data ingestion
    - Handle edge cases (empty batches, all bad quality, etc.)
    - Remove duplicates intelligently
    - Sort by timestamp
    - Consider filtering by quality flags
    - Document your data cleaning strategy in NOTES.md
    """

    if not data_batches:
        raise ValueError("Input 'data_batches' list cannot be empty.")

    # 1. Concatenate all batches into a single DataFrame
    # Using ignore_index=True ensures the final DataFrame has a continuous index
    # even if individual batch indices overlap.
    try:
        raw_data = pd.concat(data_batches, ignore_index=True)
    except Exception as e:
        # Catch issues during concatenation (e.g., inconsistent columns, bad data structure)
        raise ValueError(f"Failed to concatenate data batches: {e}")

    # Handle the case where concatenation results in an empty DataFrame
    if raw_data.empty:
        # Return an empty DataFrame with the expected columns if no data was read
        # or all batches were empty.
        expected_cols = ["timestamp", "sensor", "value", "unit", "quality"]
        # Include quality_score for consistency with the success path
        expected_cols.append("quality_score")
        return pd.DataFrame(columns=expected_cols)

    if validate:
        # 2. Data Type Conversion and Cleaning
        # Ensure 'timestamp' is in datetime format and coerce errors (handle strings/invalids)
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], errors='coerce')
        # Ensure 'value' is float (important for calculations)
        raw_data['value'] = pd.to_numeric(raw_data['value'], errors='coerce')
        
        # Drop rows where essential columns are NaT/NaN after conversion
        # Essential columns are 'timestamp' and 'sensor'. 'value' NaNs are kept for now.
        # Dropping rows with missing timestamp or sensor ID (using inplace=True)
        raw_data.dropna(subset=['timestamp', 'sensor'], inplace=True)
        
        # Re-check if the DataFrame is empty after dropping essential NaNs
        if raw_data.empty:
            expected_cols = ["timestamp", "sensor", "value", "unit", "quality", "quality_score"]
            return pd.DataFrame(columns=expected_cols)


        # 3. Handle Duplicates
        # Industrial data can have duplicate records (same time, sensor, value).
        # We assume that duplicates with the same (timestamp, sensor, value)
        # and unit are genuine duplicates. We keep the first one found.
        # FIX: Removed the assignment back to raw_data since inplace=True is used.
        raw_data.drop_duplicates(
            subset=['timestamp', 'sensor', 'value', 'unit'], 
            keep='first', 
            inplace=True
        )
        
    # 4. Sort by Timestamp
    # Sorting ensures time-series analysis is performed in the correct order,
    # correcting for simulated out-of-order arrival.
    consolidated_data = raw_data.sort_values(by='timestamp').reset_index(drop=True)
    
    # 5. Add Derived Quality Metrics (optional but good practice)
    # Convert quality flags to a numerical representation for later analysis
    quality_mapping = {"GOOD": 1, "UNCERTAIN": 0.5, "BAD": 0}
    consolidated_data['quality_score'] = consolidated_data['quality'].map(quality_mapping).fillna(0)
    
    return consolidated_data



def detect_anomalies(
    data: pd.DataFrame,
    sensor_name: str,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect anomalies in sensor data using statistical methods.
    
    Industrial sensors can produce anomalous readings due to:
    - Equipment malfunctions
    - Environmental changes
    - Sensor calibration drift
    - Communication errors
    
    Args:
        data: DataFrame from ingest_data() containing sensor readings
        sensor_name: Name of the sensor to analyze (e.g., "temperature")
        method: Detection method - "zscore", "iqr", or "rolling"
            - "zscore": Flag values beyond threshold standard deviations from mean
            - "iqr": Flag values beyond threshold * IQR from quartiles
            - "rolling": Flag based on rolling window statistics
        threshold: Sensitivity parameter (interpretation depends on method)
    
    Returns:
        DataFrame with original data plus new columns:
            - is_anomaly (bool): True if reading is anomalous
            - anomaly_score (float): Numeric score indicating severity
            - detection_method (str): Method used for detection
    
    Raises:
        ValueError: If sensor_name not found or method not supported
        ValueError: If insufficient data for the chosen method
    
    Example:
        >>> anomalies = detect_anomalies(clean_data, "temperature", method="zscore", threshold=3.0)
        >>> num_anomalies = anomalies['is_anomaly'].sum()
        >>> print(f"Found {num_anomalies} anomalies in temperature data")
    
    CANDIDATE TODO:
    - Implement at least the "zscore" method (others are optional but valued)
    - Handle missing values appropriately
    - Consider data quality flags in anomaly detection
    - Return meaningful anomaly scores for ranking/prioritization
    - Think about edge cases: what if all data is anomalous? None is?
    - Document your approach and limitations in NOTES.md
    """
    
    # Check for empty input data
    if data.empty:
        # Return empty data with expected anomaly columns
        data['is_anomaly'] = pd.Series(dtype='bool')
        data['anomaly_score'] = pd.Series(dtype='float64')
        data['detection_method'] = pd.Series(dtype='object')
        return data
    
    # Filter data for the specific sensor
    sensor_data = data[data['sensor'] == sensor_name].copy()
    
    if sensor_data.empty:
        raise ValueError(f"Sensor name '{sensor_name}' not found in the data.")

    # Prepare columns for anomaly results
    sensor_data['is_anomaly'] = False
    sensor_data['anomaly_score'] = 0.0
    sensor_data['detection_method'] = method

    # Drop NaNs from the 'value' column for statistical calculations
    values = sensor_data['value'].dropna()
    
    if values.empty:
        raise ValueError(
            f"Insufficient data for sensor '{sensor_name}'. Value column contains only NaN or has fewer than 2 non-NaN values."
        )
    if len(values) < 2 and method == "zscore":
         raise ValueError(
            f"Insufficient data for sensor '{sensor_name}'. Z-score method requires at least 2 non-NaN values."
        )

    if method == "zscore":
        # Z-Score Method

        # Calculate mean and standard deviation
        mean_val = values.mean()
        std_val = values.std()

        # Handle case where std is zero
        if std_val == 0:
            return sensor_data
        
        # Compute z-scores
        z_scores = (sensor_data['value'] - mean_val) / std_val
        anomaly_score = z_scores.abs()

        ## Flag anomalies: where the absolute Z-score exceeds the threshold
        # We also ensure the value is not NaN before flagging
        is_anomaly = (anomaly_score > threshold) & sensor_data['value'].notna()
        
        # Update the sensor_data DataFrame
        sensor_data['anomaly_score'] = anomaly_score.fillna(0.0) # Set score to 0 for NaNs
        sensor_data['is_anomaly'] = is_anomaly.fillna(False)

    elif method == "iqr":
        # Implementation of IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Compute anomaly scores based on distance from nearest bound
        def compute_iqr_score(x):
            if pd.isna(x):
                return 0.0
            elif x < lower_bound:
                return (lower_bound - x) / IQR
            elif x > upper_bound:
                return (x - upper_bound) / IQR
            else:
                return 0.0
        
        # Apply the scoring function to compute anomaly scores
        anomaly_score = sensor_data['value'].apply(compute_iqr_score)

        # Flag anomalies
        is_anomaly = (anomaly_score > 0) & sensor_data['value'].notna()

        # Update the sensor_data DataFrame
        sensor_data['anomaly_score'] = anomaly_score
        sensor_data['is_anomaly'] = is_anomaly

    elif method == "rolling":
        # Implementation of Rolling Window method
        WINDOW_SIZE = 20 # Example fixed window size
        if len(values) < WINDOW_SIZE:
            raise ValueError(
                f"Insufficient data for sensor '{sensor_name}'. Rolling method requires at least {WINDOW_SIZE} non-NaN values."
            )
        
        # Calculate rolling mean and std
        rolling_mean = sensor_data['value'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        rolling_std = sensor_data['value'].rolling(window=WINDOW_SIZE, min_periods=1).std().replace(0, np.nan)

        # Compute rolling z-scores
        rolling_z_scores = (sensor_data['value'] - rolling_mean) / rolling_std
        anomaly_score = rolling_z_scores.abs()

        # Flag anomalies
        is_anomaly = (anomaly_score > threshold) & sensor_data['value'].notna()
        
        # Update the sensor_data DataFrame
        sensor_data['anomaly_score'] = anomaly_score.fillna(0.0) # Set score to 0 for NaNs
        sensor_data['is_anomaly'] = is_anomaly.fillna(False)

    else:
        raise ValueError(f"Anomaly detection method '{method}' is not supported.")

    # Merge the anomaly results back into the original data
    # Use a left join to preserve all original data
    # This ensures that rows not related to the specific sensor retain their original values
    result_data = pd.merge(
        data, 
        sensor_data[['timestamp', 'sensor', 'is_anomaly', 'anomaly_score', 'detection_method']],
        on=['timestamp', 'sensor'],
        how='left',
        suffixes=('_original', None) # Keep the original column names
    )
    
    # Fill NaNs created by the left merge for the new columns
    # Rows not belonging to the sensor_name will get default values
    result_data['is_anomaly'] = result_data['is_anomaly'].fillna(False)
    result_data['anomaly_score'] = result_data['anomaly_score'].fillna(0.0)
    result_data['detection_method'] = result_data['detection_method'].fillna('')

    # Drop potential duplicate columns that might arise from edge cases
    result_data = result_data.loc[:,~result_data.columns.duplicated()].copy()

    return result_data


def summarize_metrics(
    data: pd.DataFrame,
    group_by: Optional[str] = "sensor",
    time_window: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for industrial sensor data.
    
    Summaries help operators and engineers understand system behavior:
    - Overall sensor performance
    - Data quality metrics
    - Temporal patterns
    - Anomaly rates
    
    Args:
        data: DataFrame from ingest_data() or detect_anomalies()
        group_by: Column to group by (typically "sensor")
        time_window: Optional pandas frequency string for time-based aggregation
            Examples: "1h" (hourly), "15min" (15 minutes), "1d" (daily)
            If None, compute overall statistics without time grouping
    
    Returns:
        Nested dictionary structure:
        {
            "sensor_name": {
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                "count": int,
                "null_count": int,
                "good_quality_pct": float,
                "anomaly_rate": float,  # if anomaly data available
                # ... additional metrics as appropriate
            },
            ...
        }
        
        If time_window is specified, returns time-indexed groups.
    
    Raises:
        ValueError: If group_by column doesn't exist
        ValueError: If data is empty or invalid
    
    Example:
        >>> metrics = summarize_metrics(anomaly_data, group_by="sensor")
        >>> temp_metrics = metrics["temperature"]
        >>> print(f"Temperature: {temp_metrics['mean']:.1f}°C ± {temp_metrics['std']:.1f}")
        >>> print(f"Data quality: {temp_metrics['good_quality_pct']:.1f}% good readings")
    
    CANDIDATE TODO:
    - Compute essential statistics (mean, std, min, max, count)
    - Calculate data quality metrics (null rate, quality flag distribution)
    - If anomaly detection was run, include anomaly statistics
    - Handle time-based grouping if time_window is provided
    - Consider what metrics are most valuable for industrial monitoring
    - Ensure robust handling of edge cases (all nulls, single value, etc.)
    - Document your metric choices in NOTES.md
    """
    
    if data.empty:
        raise ValueError("Input data cannot be empty.")
    
    if group_by and group_by not in data.columns:
        raise ValueError(f"Grouping column '{group_by}' not found in data.")

    # Prepare Grouping Key
    # Determine if anomaly columns exist for conditional metrics
    has_anomaly_data = 'is_anomaly' in data.columns
    
    # Define Core Aggregation Functions
    agg_funcs_list = ['mean', 'std', 'min', 'max', 'count']

    if time_window:
        if 'timestamp' not in data.columns:
            raise ValueError("Time-based grouping requires a 'timestamp' column.")
            
        # Time-based grouping: Group by time window (on index) and the group_by column
        data_grouped = data.set_index('timestamp').groupby(
            [pd.Grouper(freq=time_window)] + ([group_by] if group_by else [])
        )
        
        # Perform core aggregation on the 'value' column
        core_metrics = data_grouped['value'].agg(agg_funcs_list)
        core_metrics.columns = ['mean', 'std', 'min', 'max', 'count'] # Rename columns back to simple names

    elif group_by:
        # Simple grouping by the specified column
        data_grouped = data.groupby(group_by)
        # Perform core aggregation on the 'value' column
        core_metrics = data_grouped['value'].agg(agg_funcs_list)
        core_metrics.columns = ['mean', 'std', 'min', 'max', 'count'] # Rename columns back to simple names

    else:
        # No grouping (Overall) - calculate metrics on the whole 'value' column
        metrics = data['value'].agg(agg_funcs_list)
        
        # Manually create core_metrics DataFrame with the required index
        core_metrics = pd.DataFrame(metrics).T
        core_metrics.index = ["Overall"]
        core_metrics.columns = ['mean', 'std', 'min', 'max', 'count']


    # Calculate Custom Metrics (Quality and Anomaly)
    if not (time_window or group_by):
        # Non-grouped case: Calculate custom metrics based on core_metrics and data
        total_count = len(data) # Total records (including NaNs)
        null_count = total_count - core_metrics.loc['Overall', 'count']
        
        # Calculate custom metrics manually for the single 'Overall' row
        good_quality_pct_val = 0.0
        if 'quality' in data.columns:
            good_quality_count = (data['quality'] == 'GOOD').sum()
            good_quality_pct_val = (good_quality_count / total_count * 100) if total_count > 0 else 0.0

        anomaly_rate_val = 0.0
        if has_anomaly_data:
            anomaly_count = data['is_anomaly'].sum()
            anomaly_rate_val = (anomaly_count / total_count * 100) if total_count > 0 else 0.0
            
        # Append to core_metrics DataFrame
        core_metrics['null_count'] = null_count
        core_metrics['good_quality_pct'] = good_quality_pct_val
        core_metrics['anomaly_rate'] = anomaly_rate_val
        
        summary_df = core_metrics # Use core_metrics as the final summary_df

    else:
        # Grouped case (time_window or group_by is active)
        # data_grouped is the GroupBy object
        total_count = data_grouped['value'].size()
        null_count = total_count - core_metrics['count']
        
        # Good Quality Percentage (assuming 'quality' column exists)
        good_quality_pct = pd.Series(0.0, index=total_count.index)
        if 'quality' in data.columns:
            good_quality_count = data_grouped['quality'].apply(lambda x: (x == 'GOOD').sum())
            good_quality_pct = (good_quality_count / total_count * 100).fillna(0.0)
                
        # Anomaly Rate
        anomaly_rate = pd.Series(0.0, index=total_count.index)
        if has_anomaly_data:
            anomaly_count = data_grouped['is_anomaly'].sum()
            anomaly_rate = (anomaly_count / total_count * 100).fillna(0.0)

        # 3. Consolidate Metrics (for grouped case)
        custom_metrics = pd.DataFrame({
            'null_count': null_count,
            'good_quality_pct': good_quality_pct,
            'anomaly_rate': anomaly_rate,
            # 'total_readings' key is removed to prevent it from showing up in final metrics
            # 'total_readings': total_count 
        }, index=core_metrics.index)
        
        summary_df = pd.concat([core_metrics, custom_metrics], axis=1)

    # Format Output    
    result_dict = {}
    
    if time_window:
        # Multi-index (timestamp, sensor) or (timestamp,)
        summary_df_reset = summary_df.reset_index()
        
        for index, row in summary_df_reset.iterrows():
            # Identify the index columns based on whether group_by was used
            index_cols = summary_df_reset.columns[:(2 if group_by else 1)]
            
            time_key = row[index_cols[0]]
            group_key = row[index_cols[1]] if group_by else "Overall"
                
            time_str = time_key.strftime(f"TimeGroup_{time_window}_%Y-%m-%d %H:%M:%S")
            
            if time_str not in result_dict:
                result_dict[time_str] = {}
            
            # Drop the index columns before converting to dictionary
            result_dict[time_str][group_key] = row.drop(index_cols).to_dict()
            
    else:
        # Simple grouping by group_by column or Overall (Single-level index)
        result_dict = summary_df.T.to_dict()


    # Final cleanup: convert numpy dtypes to standard Python floats/ints for JSON compatibility
    def clean_metrics(d):
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned[k] = clean_metrics(v)
            elif isinstance(v, (np.float32, np.float64)):
                cleaned[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                cleaned[k] = int(v)
            elif pd.isna(v):
                cleaned[k] = None # Represent NaN as None
            else:
                cleaned[k] = v
        return cleaned
    
    return clean_metrics(result_dict)