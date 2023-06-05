import pandas as pd
from dtpp.frames.framedata import FrameData

class PauseCharacteristics(FrameData):
    def __init__(self, threshold, path, offset=0.0, frame_start_key = 'frame_start', frame_end_key = 'frame_end', vad_key = 'voice_probability'):
        """
        Input:
            threshold: the threshold for voice probability below which a pause is considered to have started
            path: path to a .csv or parquet file containing VAD data
            offset: offset to apply to first frame - usually used to clip instructional prompt at the beginning of a recording
            frame_start_key: name of the column containing the frame start times
            frame_end_key: name of the column containing the frame end times
            vad_key: name of the column containing the voice activity detection probabilities
        """
        super().__init__(path, offset, frame_start_key, frame_end_key, vad_key)
        self.df_pause = None
        self.threshold = threshold

    def pause_data(self):
        if self.df_pause is not None:
            return self.df_pause
        
        pause_start = []
        pause_end = []
        pause_duration = []
        pause_started = False
        last_frame_end = None

        for index, row in self.df.iterrows():
            v = row[self.vk]
            if row[self.vk] < self.threshold:
                last_frame_end = row[self.fe]
                if not pause_started:
                    pause_start.append(row[self.fs])
                    pause_started = True
            else:
                if pause_started:
                    pause_end.append(last_frame_end)
                    pause_duration.append(last_frame_end - pause_start[-1])
                    pause_started = False
        # need to check if we are at the end of the df and pause_started is True
        if pause_started:
            pause_end.append(last_frame_end)
            pause_duration.append(last_frame_end - pause_start[-1])
                
        self.df_pause = pd.DataFrame({'pause_start': pause_start, 'pause_end': pause_end, 'pause_duration': pause_duration})
        return self.df_pause


    def pause_count(self):
        """
        Get the number of pauses in the recording.
        Input:
            threshold: the threshold for voice probability below which a pause is considered to have started (default 0.5)
        """
        return len(self.pause_data()['pause_duration'])
    
    def pause_mean(self):
        """
        Get the mean pause duration.
        """
        return self.pause_data()['pause_duration'].mean()
    
    def pause_std(self):
        """
        Get the standard deviation of pause durations.
        """
        return self.pause_data()['pause_duration'].std()
    
    def pause_min(self):
        """
        Get the minimum pause duration.
        """
        return self.pause_data()['pause_duration'].min()
    
    def pause_max(self):
        """
        Get the maximum pause duration.
        """
        return self.pause_data()['pause_duration'].max()
    
    def pause_range(self):
        """
        Get the range of pause durations.
        """
        return self.pause_max() - self.pause_min()
    