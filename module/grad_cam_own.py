import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import torch.nn as nn
from matplotlib import pyplot as plt


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)
	
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model

		self.feature = nn.Sequential(
			self.model.feature[0],
			self.model.feature[1],

			self.model.feature[2],

			self.model.feature[3],

			self.model.feature[4],

			self.model.feature[5],

			self.model.feature[6],
			self.model.feature[7]
		)


		self.feature_extractor = FeatureExtractor(self.feature, target_layers)

		self.classifier = nn.Sequential(
			self.model.fc1,
			self.model.bn1,
			self.model.resnet.relu,
			self.model.fc2
		)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)

		output = self.model.feature[8](output)

		output = output.view(output.size(0), -1)

		output = torch.flatten(output, 1)
		output = self.classifier(output)
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255

	img = img / 128
	cam = heatmap + img
	cam = cam / np.max(cam)

	cam = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)

	return cam

	# cv2.imwrite("cam.jpg", np.uint8(255 * cam))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		

		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

      

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)


      
	
		self.model.feature.zero_grad()
		# self.feature.zero_grad()
		# self.classifier.zero_grad()
		self.model.fc1.zero_grad()
		self.model.fc2.zero_grad()
		self.model.bn1.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		### select layer
		target = features[-2]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (256, 256))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.module.features.zero_grad()
		# self.module.classifier.zero_grad()
		one_hot.backward(retain_variables=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

# def get_args():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--use-cuda', action='store_true', default=False,
# 	                    help='Use NVIDIA GPU acceleration')
# 	parser.add_argument('--image-path', type=str, default='elephant_cam.jpg',
# 	                    help='Input image path')
# 	args = parser.parse_args()
# 	args.use_cuda = args.use_cuda and torch.cuda.is_available()
# 	if args.use_cuda:
# 	    print("Using GPU for acceleration")
	    
# 	else:
# 	    print("Using CPU for computation")

# 	return args

if __name__ == '__main__':
    cfg = BaseConfig()
    fixed = '--exp_name cam --model_name resnet18f --data_name amd --polar 256 --k_fold 2 --batch_size 1 --eval_only --num_classes 2 --gpu_id 4 --pretrained /data2/wangjinhong/result/wjh/PoN/save_model/checkpoint_amd_finetune_new/PoCaco2_256a_amd_finetune1_fold2/resnet18_max_auc_0.996415770609319.pth'.split()
    args = cfg.initialize(fixed)
    use_cuda = True


    checkpoint = torch.load(args.pretrained)
    state_dict = checkpoint['net_state_dict']

    model = getattr(models, args.model_name.lower())(args)
    model.cuda()
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    grad_cam = GradCam(model=model, target_layer_names='4567', use_cuda=use_cuda)

	# Can work with any module, but it assumes that the module has a
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	# grad_cam = GradCam(model = models.vgg19(pretrained=True), \
	# 				target_layer_names = ["35"], use_cuda=args.use_cuda)

	# img = cv2.imread(args.image_path, 1)
	# img = np.float32(cv2.resize(img, (224, 224))) / 255
	# input = preprocess_image(img)

	# # If None, returns the map for the highest scoring category.
	# # Otherwise, targets the requested index.
	# target_index = None

	# mask = grad_cam(input, target_index)

	# show_cam_on_image(img, mask)

	# gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=args.use_cuda)
	# gb = gb_model(input, index=target_index)
	# utils.save_image(torch.from_numpy(gb), 'gb.jpg')

	# cam_mask = np.zeros(gb.shape)
	# for i in range(0, gb.shape[0]):
	#     cam_mask[i, :, :] = mask

	# cam_gb = np.multiply(cam_mask, gb)
	# utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
