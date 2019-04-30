import torch
import torchvision as tv
from PIL import Image
import os

train_path = '/home/shixu/data/BR/BR_train/'
test_path = '/home/shixu/data/BR/BR_test'

def loader(path):
	images = []
	targets = []
	for dirname in os.listdir(path):
		for filename in os.listdir(path + '/' + dirname):
			fp = open('{}/{}/{}'.format(path, dirname, filename),'rb')
			images.append(Image.open(fp))
			targets.append(dirname)
			fp.close()
			print(filename)
	return images, targets

class Data(torch.utils.data.Dataset):
	def __init__(self, img, tar):
		self.img = img
		self.tar = tar

	def __getitem__(self, index):
		return self.img[index], self.tar[index]

	def __len__(self):
		return len(self.img)

	

if __name__ == '__main__':
	train_images, train_targets = loader(train_path)
#	test_images, test_targets = loader(test_path)

	train_data = Data(train_images, train_targets)
#	test_data = Data(test_images, test_targets)

	print(len(train_data))
#	print(test_data)
		
