#!coding: utf-8
import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image
from skimage import img_as_float
from skimage.transform import swirl
from skimage.measure import compare_ssim
from skimage.color import rgb2gray
from scipy.stats import pearsonr
from skimage.util.shape import view_as_blocks
# skimage.filters.gaussian(image, sigma)

def crop(img, box):
	# 下面两者不一样
	# return np.asarray(img.crop(box), dtype=np.float)
	# return img_as_float(np.asarray(img.crop(box)))
	return np.asarray(img.crop(box))

def difference(img1, img2, funcs):
	# This is equivalent to np.array(a, copy=True)  
	tmp1 = np.copy(img1)
	tmp2 = np.copy(img2)
	for func in funcs:
		tmp1 = func(tmp1)
		tmp2 = func(tmp2)
	return tmp1 - tmp2

# np.sum()
# ndarray.sum Equivalent method.

def func_hist(img, bins=np.arange(0, 256 + 4, 4)):
	hist = np.histogram(img, bins=bins)
	return hist[0]

def func_fft(img):
	return np.abs(np.fft.fft2(img))

def func_sum(arr):
	return arr.flatten().sum()

def ssim(img1, img2):
	if img1.shape != img2.shape:
		raise Exception("img1 and img2 do not have same shape!", img1.shape, img2.shape)
	if len(img1.shape) == 3:
		_img1 = rgb2gray(img1)
		_img2 = rgb2gray(img2)
	return compare_ssim(_img1, _img2, data_range=_img2.max() - _img2.min())

def correlation(arr1 ,arr2):
	return pearsonr(arr1.flatten(), arr2.flatten())[0]

def pool(img, block_shape=(4,4), func=np.median):
	if len(img) == 3:
		_img = rgb2gray(img)
	# see `img` as a matrix of blocks (of shape `block_shape`)
	view = view_as_blocks(_img, block_shape)
	# collapse the last two dimensions in one
	flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
	return func(flatten_view, axis=2)

# Structural similarity index
# When comparing images, the mean squared error (MSE)–while simple to implement–is not 
# highly indicative of perceived similarity. Structural similarity aims to address this shortcoming 
# by taking texture into account.

# pillow 和 skimage 对于图像的观点是不一样的，前者是真把图像看做图像，后者就是把图像看作数组
# image = Image.open("ponzo.jpg")   # image is a PIL image 
# array = numpy.array(image)          # array is a numpy array 
# image2 = Image.fromarray(array)   # image2 is a PIL image 

# The Pearson correlation coefficient measures the linear relationship between two datasets.
# Strictly speaking, Pearson’s correlation requires that each dataset be normally distributed, 
# and not necessarily zero-mean. Like other correlation coefficients, this one varies between -1 and +1 with 0 
# implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. 
# Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.
# The p-value roughly indicates the probability of an uncorrelated system producing datasets 
# that have a Pearson correlation at least as extreme as the one computed from these datasets. 
# The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.