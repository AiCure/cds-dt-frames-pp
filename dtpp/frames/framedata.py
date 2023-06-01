import pandas as pd
import math

class FrameData():
    def __init__(self, path, offset=-math.inf, frame_start_key = 'frame_start', frame_end_key = 'frame_end', value_key = None):
        """
        Input:
            path: path to a .csv or parquet file containing frame based data (aka raw data)
            offset: left end temporal offset to apply to first frame - usually used to clip instructional prompt at the beginning of a recording
            frame_start_key: name of the column containing the frame start times
            frame_end_key: name of the column containing the frame end times
            value_key: string or list of the column name(s) containing the frame values
        """
        if not value_key:
            raise ValueError('value_key must be specified')

        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            raise ValueError('File must be .csv or .parquet')
        
        #remove rows from df where frame_end < offset
        self.df = df[df[frame_end_key] > offset]

        self.offset = offset
        self.fs = frame_start_key
        self.fe = frame_end_key
        self.vk = value_key

    def get_values_before(self, time):
        """
        Get the values of all frames that end before time.
        """
        return self.df[self.df[self.fe] < time][self.vk]
    
    def get_values_after(self, time):
        """
        Get the values of all frames that start after time.
        """
        return self.df[self.df[self.fs] > time][self.vk]
    
    def get_values_between(self, start_time, end_time):
        """
        Get the values of all frames that overlap with the time window defined 
        by [start_time, end_time].
        """
        ffs = start_time
        ffe = end_time
        c1 = (ffs < self.df[self.fs]) & (ffe > self.df[self.fs])
        c2 = (ffs >= self.df[self.fs]) & (ffe <= self.df[self.fe])
        c3 = (ffs < self.df[self.fe]) & (ffe > self.df[self.fe])
        return self.df[c1 | c2 | c3][self.vk]
    
    def mean_value_between(self, start_time, end_time):
        return self.get_values_between(start_time, end_time).mean() 

    def mean_value(self):
        """
        Get the mean value of all frames.
        """
        return self.df[self.vk].mean()
    
    def std_value(self):
        """
        Get the standard deviation of all frames.
        """
        return self.df[self.vk].std()
    
    def range_value(self):
        """
        Get the range of all frames.
        """
        return self.df[self.vk].max() - self.df[self.vk].min()
    
    def values(self):
        """
        Get the values of all frames.
        """
        return self.df[self.vk]
    
    def dataframe(self):
        """
        Get the underlying dataframe.
        """
        return self.df
    
class FilteredFrameData(FrameData):
    def __init__(self, data, datafilter):
        """
        Input:
            data: FrameData object
            datafilter: FrameData Filter object
        """
        self.data = data
        self.datafilter = datafilter

        fs = []
        fe = []
        v = []
        for index, row in data.dataframe().iterrows():
            if not datafilter.filter_at(row[data.fs], row[data.fe]):
                fs.append(row[data.fs])
                fe.append(row[data.fe])
                v.append(row[data.vk])

        self.offset = data.offset
        self.fs = data.fs
        self.fe = data.fe
        self.vk = data.vk
        self.df = pd.DataFrame({self.fs: fs, self.fe: fe, self.vk: v})
        
