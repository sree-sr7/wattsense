import numpy as np
import pandas as pd

class AnomalyDetector:
    """
    Robust Z-Score Anomaly Detector via Median Absolute Deviation (MAD).
    """
    def __init__(self, z_threshold=2.5, min_deviation_kwh=0.50):
        self.z_threshold = z_threshold
        self.min_deviation_kwh = min_deviation_kwh

    def detect(self, scored_df, actual_col='actual_kwh', baseline_col='expected_kwh'):
        df = scored_df.copy()
        mad_col = 'baseline_mad' if 'baseline_mad' in df.columns else None

        df['residual_kwh'] = df[actual_col] - df[baseline_col]

        if mad_col and df[mad_col].notnull().any():
            df['anomaly_score'] = df['residual_kwh'] / (1.4826 * df[mad_col])
            df['anomaly'] = (df['anomaly_score'] > self.z_threshold) & (df['residual_kwh'] > self.min_deviation_kwh)
        else:
            df['anomaly_score'] = np.nan
            df['anomaly'] = df['residual_kwh'] > self.min_deviation_kwh

        def classify(row):
            if not row['anomaly']:
                return "Normal"
            if row['timestamp'].dayofweek in [5, 6]:
                return "Weekend excess load"
            if not (8 <= row['timestamp'].hour <= 19):
                return "After-hours excess load"
            return "Daytime operational excess"

        df['anomaly_type'] = df.apply(classify, axis=1)
        return df
