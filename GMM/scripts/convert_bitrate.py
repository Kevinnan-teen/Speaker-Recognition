'''
转换音频采样率/比特率
'''

import os
import subprocess


def mkdirFamle():
	input_path = '../speech/male/male_'
	for i in range(6):
		output_path = input_path + str(i+1)
		os.makedirs(output_path)


def convertBitrate():
	input_path_list = []
	output_path_list = []
	for i in range(6):
		input_path = '../speech/male/out_' + str(i+1) + '/'
		output_path = '../speech/male/male_' + str(i+1) + '/'
		input_path_list.append(input_path)
		output_path_list.append(output_path)
	for i in range(6):
		for file in os.listdir(input_path_list[i]):
			input_file = input_path_list[i] + file
			output_file = output_path_list[i] + file[:-4] + '.wav'
			cmd = "ffmpeg -i " + input_file + " -ar 16000 -ac 1 " + output_file
			subprocess.call(cmd, shell=True)




if __name__=="__main__":
	#mkdirFamle()
	convertBitrate()