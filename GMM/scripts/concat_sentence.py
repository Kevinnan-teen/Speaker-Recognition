import glob
import os
import numpy as np
import scipy.io.wavfile as wav

#merge_files_in_a_folder # 合并音频
def merge_files(path_read_folder, path_write_wav_file):

    #files = os.listdir(path_read_folder)
    merged_signal = []
    for filename in glob.glob(os.path.join(path_read_folder, 'sentence*.wav')):
        print(filename)
        sr, signal = wav.read(filename)
        merged_signal.append(signal)
    # print(len(merged_signal))
    print(merged_signal[0].shape, merged_signal[1].shape)
    merged_signal=np.hstack(merged_signal)
    merged_signal = np.asarray(merged_signal, dtype=np.int16)
    wav.write(path_write_wav_file, sr, merged_signal)
 

#noisy train total
female_siri_path = '../speech/female/female_'
male_siri_path = '../speech/male/male_'
for i in range(9):
    path_read_folder = female_siri_path + str(i+1)
    path_write_wav_file = path_read_folder + "/merge_result.wav"
    merge_files(path_read_folder, path_write_wav_file)
for i in range(6):
    path_read_folder = male_siri_path + str(i+1)
    path_write_wav_file = path_read_folder + "/merge_result.wav"
    merge_files(path_read_folder, path_write_wav_file)

