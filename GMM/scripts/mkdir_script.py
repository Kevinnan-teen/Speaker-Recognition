import os

def mkdir_mfcc24_gmm_model():
	root_dir = '../models/TIMIT_MFCC_24/'
	for i in range(462):
		spk_model_dir = root_dir + 'spk_' + str(i+1)
		print(spk_model_dir)
		os.makedirs(spk_model_dir)


def mkdir_mfcc_13d_7_diag_model():
	root_dir = '../models/TIMIT_MFCC_13d_7_diag/'
	for i in range(462):
		spk_model_dir = root_dir + 'spk_' + str(i+1)
		print(spk_model_dir)
		os.makedirs(spk_model_dir)


def mkdir_timit_test():
	root_dir = '../speech/TIMIT/TEST_MFCC/'
	for i in range(462):
		timit_test_dir = root_dir + 'spk_' + str(i+1)
		print(timit_test_dir)
		os.makedirs(timit_test_dir)


def justify_multiprocess_nor_success():
	root_dir = "../speech/TIMIT/TRAIN_MFCC/"
	spk_list = os.listdir(root_dir)
	print(len(spk_list))
	for spk in spk_list:
		content = os.listdir(os.path.join(root_dir, spk))
		if(len(content) == 2):
			pass
		else:
			print("error")


#mkdir_mfcc24_gmm_model()
# mkdir_timit_test()
#justify_multiprocess_nor_success()
mkdir_mfcc_13d_7_diag_model()