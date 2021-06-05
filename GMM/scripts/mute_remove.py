from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
[Fs, x] = aIO.read_audio_file("../speech/female/female_1/siri_1.wav")
segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window = 1.0, weight = 0.3, plot = True)
