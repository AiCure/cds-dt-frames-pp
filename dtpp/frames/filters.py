import pandas as pd
from dtpp.frames.framedata import FrameData
import math

class FrameThresholdFilter(FrameData):
    def __init__(self, threshold, path, offset=-math.inf, frame_start_key = 'frame_start', frame_end_key = 'frame_end', value_key = None, default_filter=False):
        """
        Input:
            threshold: the threshold for filtering a second dataset. If the filter value is below 
            threshold at a given time, the second data set should be filtered at that time point.
            path: path to a .csv or parquet file containing VAD data
            offset: offset to apply to first frame - usually used to clip instructional prompt at the beginning of a recording
            frame_start_key: name of the column containing the frame start times
            frame_end_key: name of the column containing the frame end times
            value_key: name of the column containing the value to determine whether to filter
            default_filter: if True, filter by default if the value is not present at a given time point
        """
        super().__init__(path, offset, frame_start_key, frame_end_key, value_key)
        self.threshold = threshold

    def filter_at(self, start_time, stop_time):
        """
         Returns True if data should be filtered at the given time point.
        """
        f = self.mean_value_between(start_time, stop_time) < self.threshold
        return f if not pd.isna(f) else self.default_filter