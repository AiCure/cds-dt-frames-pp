from dtpp.frames.framedata import FrameData, FilteredFrameData
from dtpp.frames.filters import FrameThresholdFilter
import glob
import pandas as pd
import os
import librosa


def process_directory(vad_input_dir, acoustic_input_dir, output_dir, offset, input_type='csv', 
                      vad_threshold = 0.5, vad_frame_start_key = 'frame_start', vad_frame_end_key = 'frame_end', vad_key = 'voice_probability',
                      acoustic_frame_start_key = 'frame_start', acoustic_frame_end_key = 'frame_end', acoustic_value_key = None):
    
    if acoustic_value_key is None:
        raise ValueError('acoustic_value_key must be specified')

    if input_type == 'csv':
        vad_files = glob.glob(f'{vad_input_dir}/**/*.csv', recursive=True)
        files_to_process = glob.glob(f'{acoustic_input_dir}/**/*.csv', recursive=True) 
    elif input_type == 'parquet':
        vad_files = glob.glob(f'{vad_input_dir}/**/*.parquet', recursive=True)
        files_to_process = glob.glob(f'{acoustic_input_dir}/**/*.parquet', recursive=True) 
    else:
        raise ValueError('input_type must be csv or parquet')
    
    file_dict = {}
    for file in vad_files:
        url_info = file.split('/')[-1].split('.')[0]
        file_dict[url_info] = {'vad': file, 'acoustic': None}
    for file in files_to_process:
        url_info = file.split('/')[-1].split('.')[0]
        file_dict[url_info] = file_dict.get(url_info, {'vad':None})
        file_dict[url_info]['acoustic'] = file
    
    res = list()
    for url_info, d in file_dict.items():
        url_info = pd.DataFrame([{'formatted_url': url_info+'.mp4'}])
        try:
            vad_file = d.get('vad', None)
            acoustic_file = d.get('acoustic', None)
            
            if acoustic_file is None:
                continue # skip if no acoustic file

            if vad_file is None:
                print(f"Warning: no VAD file found for {url_info}")
            else:
                vad = FrameThresholdFilter(vad_threshold, vad_file, frame_start_key=vad_frame_start_key, frame_end_key=vad_frame_end_key, value_key=vad_key)
                acoustic = FrameData(acoustic_file, offset = offset, frame_start_key=acoustic_frame_start_key, frame_end_key=acoustic_frame_end_key, value_key=acoustic_value_key)
                acoustic_filtered = FilteredFrameData(acoustic, vad)
                features = pd.DataFrame([{f'{acoustic_value_key}_mean': acoustic_filtered.mean_value(),
                                          f'{acoustic_value_key}_std': acoustic_filtered.std_value(),
                                          f'{acoustic_value_key}_range': acoustic_filtered.range_value()}])
                features = pd.concat([url_info, features], axis=1)
                res.append(features)
        except Exception as e:
            continue # skip files that don't work (probably not VAD data)
    res = pd.concat(res, axis=0).to_csv(f'{output_dir}/derived_{acoustic_value_key}.csv', index=False)
    pass

if __name__ == '__main__':
    process_directory('./tests/data/vad', './tests/data/acoustic', './tests/data/output', 0.0, 
                      input_type='parquet', acoustic_frame_start_key='intensity_frame_start', acoustic_frame_end_key='intensity_frame_end', acoustic_value_key='intensity')
    # offset = librosa.get_duration(filename='/Users/aaronmasino/002_dev/cds-dt-frames-pp/tests/data/picture_description.mp3')
    # input_dir = "/Users/aaronmasino/Desktop/KAR-011-300/vad"
    # output_dir = "/Users/aaronmasino/Desktop/KAR-011-300/output"
    # process_directory(input_dir, output_dir, offset, input_type='parquet', vad_key='voice_probability')