import params as hp
from sphfile import SPHFile
import glob
import os

 
if __name__ == "__main__":
    train_path = '../../speech/TIMIT/TRAIN/*/*/*.WAV'
    test_path  =  '../../speech/TIMIT/TEST/*/*/*.WAV'
    train_sph_files = glob.glob(train_path)
    test_sph_files = glob.glob(test_path)
    print(len(train_sph_files),"train utterences")
    print(len(test_sph_files),"test utterences")
    # for i in train_sph_files:
    #     sph = SPHFile(i)
    #     sph.write_wav(filename=i.replace(".WAV","_.wav"))
        #os.remove(i)
    # path = 'D:/pycharm_proj/corpus/data/lisa/data/timit/raw/TIMIT/TEST/*/*/*.WAV'
    # sph_files_test = glob.glob(path)
    # print(len(sph_files_test),"test utterences")
    for i in test_sph_files:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV","_.wav"))
        # os.remove(i)
    # print("Completed")
