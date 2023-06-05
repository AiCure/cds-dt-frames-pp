from dtpp.derived.vocal_activity import PauseCharacteristics
import glob
import pandas as pd
import os
import librosa


def process_directory(input_dir, output_dir, offset, input_type='csv', 
                      threshold = 0.5, frame_start_key = 'frame_start', frame_end_key = 'frame_end', vad_key = 'voice_probability'):
    if input_type == 'csv':
        files_to_process = glob.glob(f'{input_dir}/**/*.csv', recursive=True) 
    elif input_type == 'parquet':
        files_to_process = glob.glob(f'{input_dir}/**/*.parquet', recursive=True) 
    else:
        raise ValueError('input_type must be csv or parquet')
    
    res = list()
    for file in files_to_process:
        url_info = pd.DataFrame([{'formatted_url': file.split('/')[-1].split('.')[0]+'.mp4'}])
        try:
            pc = PauseCharacteristics(threshold, file, offset, frame_start_key, frame_end_key, vad_key)
            features = pd.DataFrame([{'pause_count': pc.pause_count(), 
                                      'pause_mean': pc.pause_mean(), 
                                      'pause_std': pc.pause_std(),
                                      'pause_range': pc.pause_range()}])
            features = pd.concat([url_info, features], axis=1)
            res.append(features)
        except Exception as e:
            continue # skip files that don't work (probably not VAD data)
    res = pd.concat(res, axis=0).to_csv(f'{output_dir}/derived_{vad_key}.csv', index=False)
    pass

if __name__ == '__main__':
    process_directory('./tests/data/vad', './tests/data/output', 0.0, input_type='parquet', vad_key='voice_probability')
    # offset = librosa.get_duration(filename='/Users/aaronmasino/002_dev/cds-dt-frames-pp/tests/data/picture_description.mp3')
    # input_dir = "/Users/aaronmasino/Desktop/KAR-011-300/vad"
    # output_dir = "/Users/aaronmasino/Desktop/KAR-011-300/output"
    # process_directory(input_dir, output_dir, offset, input_type='parquet', vad_key='voice_probability')