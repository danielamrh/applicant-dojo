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
    # TODO: Implement this function

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
    # TODO: Implement this function
    raise NotImplementedError(
        "detect_anomalies() must be implemented by the candidate"
    )


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
    # TODO: Implement this function
    raise NotImplementedError(
        "summarize_metrics() must be implemented by the candidate"
    )
