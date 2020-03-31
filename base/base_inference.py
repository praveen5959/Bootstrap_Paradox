#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np
from time import time
from torch.nn import functional as F


#------------------------------------------------------------------------------
#   BaseInference
#------------------------------------------------------------------------------
class BaseInference(object):
	def __init__(self, model, color_f=[255,0,0], color_b=[0,0,255], kernel_sz=25, sigma=0, background_path=None):
		self.model = model
		self.color_f = color_f
		self.color_b = color_b
		self.kernel_sz = kernel_sz
		self.sigma = sigma
		self.background_path = background_path
		if background_path is not None:
			self.background = cv2.imread(background_path)[...,::-1]
			self.background = self.background.astype(np.float32)


	def load_image(self):
		raise NotImplementedError


	def preprocess(self, image, *args):
		raise NotImplementedError


	def predict(self, X):
		raise NotImplementedError


	def draw_matting(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		#mask = mask.round(
		#ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
		mask = 255*(1.0-mask)
		mask = np.expand_dims(mask, axis=2)
		mask = np.tile(mask, (1,1,3))
		mask = mask.astype(np.uint8)
		kernel = np.ones((5,5), np.uint8) 
		mask = cv2.erode(mask, kernel, iterations=4)
		ret,thresh1 = cv2.threshold(mask,125,255,cv2.THRESH_BINARY)
		edged = cv2.Canny(thresh1, 30, 200)
		#contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# Draw all contours 
# -1 signifies drawing all contours 
		#cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
		#thresh1=cv2.fillPoly(thresh1, contours, 255)
		thresh1 = cv2.dilate(thresh1, kernel, iterations=2)
		kernel = np.ones((1,1), np.uint8) 
		thresh1 = cv2.erode(thresh1, kernel, iterations=2)
		#thresh1 = cv2.dilate(thresh1, kernel, iterations=2)
		image_alpha = cv2.add(image, thresh1)
		return image_alpha
	def draw_transperency(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		#mask = mask.round()
		alpha = np.zeros_like(image, dtype=np.uint8)
		alpha[mask==1, :] = self.color_f
		alpha[mask==0, :] = self.color_b
		image_alpha = cv2.add(image, alpha)
		return mask


	def draw_background(self, image, mask):
		"""
		image (np.uint8) shape (H,W,3)
		mask  (np.float32) range from 0 to 1, shape (H,W)
		"""
		image = image.astype(np.float32)
		#mask_filtered = cv2.GaussianBlur(mask, (self.kernel_sz, self.kernel_sz), self.sigma)
		mask_filtered = np.expand_dims(mask_filtered, axis=2)
		mask_filtered = np.tile(mask_filtered, (1,1,3))

		image_alpha = image*mask_filtered + self.background*(1-mask_filtered)
		return image_alpha.astype(np.uint8)


#------------------------------------------------------------------------------
#   VideoInference
#------------------------------------------------------------------------------
class VideoInference(BaseInference):
	def __init__(self, model, video_path, input_size, use_cuda=False, draw_mode='matting',
				color_f=[255,0,0], color_b=[0,0,255], kernel_sz=25, sigma=0, background_path=None):

		# Initialize
		super(VideoInference, self).__init__(model, color_f, color_b, kernel_sz, sigma, background_path)
		self.input_size = input_size
		self.use_cuda = use_cuda
		self.draw_mode = draw_mode
		if draw_mode=='matting':
			self.draw_func = self.draw_matting
		elif draw_mode=='transperency':
			self.draw_func = self.draw_transperency
		elif draw_mode=='background':
			self.draw_func = self.draw_background
		else:
			raise NotImplementedError

		# Preprocess
		self.mean = np.array([0.485,0.456,0.406])[None,None,:]
		self.std = np.array([0.229,0.224,0.225])[None,None,:]

		# Read video
		self.video_path = video_path
		self.cap = cv2.VideoCapture(video_path)
		_, frame = self.cap.read()
		self.H, self.W = frame.shape[:2]


	def load_image(self):
		_, frame = self.cap.read()
		image = frame[...,::-1]
		return image


	def preprocess(self, image):
		image = cv2.resize(image, (self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR)
		#kernel = np.ones((3,3),np.float32)/225
		#image = cv2.filter2D(image,-1,kernel) 
		image = cv2.GaussianBlur(image,(5,5),0)
		image=cv2.medianBlur(image,5)
		image = image.astype(np.float32) / 255.0
		image = (image - self.mean) / self.std
		
		#kernel = np.ones((5,5), np.uint8) 
		#im = cv2.erode(image, kernel, iterations=2) 
		#image = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
		#gaussian_1 = cv2.GaussianBlur(image, (9,9), 10.0)
		#image = cv2.addWeighted(image, 1.5, gaussian_1, -0.5, 0, image)
		#image = cv2.Laplacian(image,cv2.CV_64F)
		X = np.transpose(image, axes=(2, 0, 1))
		X = np.expand_dims(X, axis=0)
		X = torch.tensor(X, dtype=torch.float32)
		return X


	def predict(self, X):
		with torch.no_grad():
			if self.use_cuda:
				mask = self.model(X.cuda())
				mask = F.interpolate(mask, size=(self.H, self.W), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].cpu().numpy()
			else:
				mask = self.model(X)
				print(mask.shape)
				mask = F.interpolate(mask, size=(self.H, self.W), mode='bilinear', align_corners=True)
				mask = F.softmax(mask, dim=1)
				mask = mask[0,1,...].numpy()
			return mask


	def run(self):
		while(True):
			# Read frame from camera
			start_time = time()
			image = self.load_image()
			read_cam_time = time()

			# Preprocess
			X = self.preprocess(image)
			preproc_time = time()

			# Predict
			mask = self.predict(X)
			predict_time = time()

			# Draw result
			image_alpha = self.draw_func(image, mask)
			draw_time = time()

			# Wait for interupt
			cv2.imshow('webcam', image_alpha[..., ::-1])
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			# Print runtime
			read = read_cam_time-start_time
			preproc = preproc_time-read_cam_time
			pred = predict_time-preproc_time
			draw = draw_time-predict_time
			total = read + preproc + pred + draw
			fps = 1 / total
			#print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; draw: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" % 
				#(read, preproc, pred, draw, total, fps))
