#!/usr/bin/python3

import random
import math
import tensorflow_datasets as tensorflow_datasets
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
import albumentations

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:  # add color Gaussian noise
        img = img + np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:  # add grayscale Gaussian noise
        img = img + np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img = img + np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).astype(absx.dtype)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).astype(absx.dtype))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = np.tile(np.reshape(left, (out_length, 1)), (1, P)) + \
              np.tile(np.reshape(np.linspace(0, P - 1, P), (1, P)), (out_length, 1))

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = np.tile(np.reshape(u, (out_length, 1)), (1, P)) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.reshape(np.sum(weights, 1), (out_length, 1))
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:,1:P-2]
        weights = weights[:,1:P-2]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:,0:P-2]
        weights = weights[:,0:P-2]
    weights = np.ascontiguousarray(weights)
    indices = np.ascontiguousarray(indices)
    sym_len_s = -np.min(indices) + 1
    sym_len_e = np.max(indices) - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h

def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h

def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k

def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf
    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_Gaussian(ksize=2 * random.randint(2, 11) + 3, theta=random.random() * np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2 * random.randint(2, 11) + 3, wd * random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    need_squeeze = True if len(img.shape) == 2 else False
    if need_squeeze:
        img = np.expand_dims(img, axis = 2)

    in_H, in_W, in_C = img.shape()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = np.zeros((in_H + sym_len_Hs + sym_len_He, in_W, in_C), dtype = np.float32)
    img_aug[sym_len_Hs:,in_H,:,:] = img

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1, dtype = np.int64)
    sym_patch_inv = sym_patch[inv_idx,:,:]
    img_aug[:sym_len_Hs, :, :] = sym_patch_inv

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1, dtype = np.int64)
    sym_patch_inv = sym_patch[inv_idx,:,:]
    img_aug[sym_len_Hs + in_H:sym_len_He,:,:] = sym_patch_inv

    out_1 = np.zeros((out_H, in_W, in_C), dtype = np.float32)
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = np.matmul(np.transpose(img_aug[idx:idx + kernel_width, :, j], (1,0,2)), weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = np.zeros((out_H, in_W + sym_len_Ws + sym_len_We, in_C), dtype = np.float32)
    out_1_aug[:,sym_len_Ws:in_W,:] = out_1

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1, dtype = np.int64)
    sym_patch_inv = sym_patch[:,inv_idx,:]
    out_1_aug[:,:sym_len_Ws,:] = sym_patch_inv

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1, dtype = np.int64)
    sym_patch_inv = sym_patch[:,inv_idx,:]
    out_1_aug[:,sym_len_Ws + in_W:sym_len_We,:] = sym_patch_inv

    out_2 = np.zeros((out_H, out_W, in_C), dtype = np.float32)
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = np.matmul(out_1_aug[:, idx:idx + kernel_width, j], weights_W[i])
    if need_squeeze:
        axis = list()
        for i,d in enumerate(out_2.shape):
          if d == 1: axis.append(i)
        np.squeeze(out_2, axis = axis)

    return out_2

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def degradation_bsrgan_variant(image, sf=4, isp_model=None):
    image = np.float32(image/255.)
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    h1, w1 = image.shape[:2]
    image = image.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop
    h, w = image.shape[:2]

    hq = image.copy()

    if sf == 4 and random.random() < scale2_prob:  # downsample1
        if np.random.rand() < 0.5:
            image = cv2.resize(image, (int(1 / 2 * image.shape[1]), int(1 / 2 * image.shape[0])),
                               interpolation=random.choice([1, 2, 3]))
        else:
            image = imresize_np(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0:
            image = add_blur(image, sf=sf)

        elif i == 1:
            image = add_blur(image, sf=sf)

        elif i == 2:
            a, b = image.shape[1], image.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                image = cv2.resize(image, (int(1 / sf1 * image.shape[1]), int(1 / sf1 * image.shape[0])),
                                   interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
                image = ndimage.filters.convolve(image, np.expand_dims(k_shifted, axis=2), mode='mirror')
                image = image[0::sf, 0::sf, ...]  # nearest downsampling
            image = np.clip(image, 0.0, 1.0)

        elif i == 3:
            # downsample3
            image = cv2.resize(image, (int(1 / sf * a), int(1 / sf * b)), interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)

        elif i == 4:
            # add Gaussian noise
            image = add_Gaussian_noise(image, noise_level1=2, noise_level2=25)

        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                image = add_JPEG_noise(image)

    # add final JPEG compression noise
    image = add_JPEG_noise(image)
    image = np.uint8((image.clip(0, 1)*255.).round())
    example = {"image":image}
    return example

def ImageNetSR(split = 'train', **kwargs):
  assert split in {'train', 'validation'}
  size = kwargs.get('size', 256)
  downscale_f = kwargs.get('downscale_f', 4)
  degradation = kwargs.get('degradation', 'bsrgan') # 'bsrgan' or 'bsrgan_light' or <interpolation_fn>
  min_crop_f = kwargs.get('min_crop_f', 0.5)
  max_crop_f = kwargs.get('max_crop_f', 1.)
  crop_method = kwargs.get('crop_method', 'random crop') # 'random crop' or 'center crop'

  def parse_function(example):
    image, label = example['image'], example['label']
    if not image.mode == 'RGB':
      image = image.convert('RGB')
    image = np.array(image).astype(np.uint8)
    min_side_len = min(image.shape[:2])
    crop_side_len = min_side_len * np.random.uniform(min_crop_f, max_crop_f, size=None)
    crop_side_len = int(crop_side_len)
    cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len) if crop_method == 'center crop' else \
              albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
    image = cropper(image = image)["image"]
    image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
    image = image_rescaler(image = image)["image"]
    degradation_process =

  ds = tfds.load('imagenet2012', split = split, shuffle_files = True)
