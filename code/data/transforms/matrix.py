# -*- coding: utf-8 -*-
"""

    The ``transformsMatrix`` module
    ========================
 
    This contains matrix-specific data augmentation transforms.
    The entire coding style approach is mimicked from Pytorch
    Here we define transforms that can be applied to any matrix.
    Note that lots of these do not make sense for spectral transforms.
    Therefore the "STFT"-specific augmentations are defined in
    a separate file. Here we define generic matrix transforms

    Example
    -------
 
    Currently implemented
    ---------------------
    
    * Rescale               : Rescale a matrix to a given size.
    * CropRandom            : Crop randomly the matrix in a sample.
    * CropCenter            : Crop the matrix from its center
    * Normalize             : Normalize an tensor with given mean and std
    * NormalizeDimension    : Normalize a tensor by first computing mean and std over a specified dimension
    * RandomCrop            : Crop a given matrix at a random location
    * RandomHorizontalFlip  : Horizontally flip a matrix randomly with proba 0.5
    * RandomVerticalFlip    : Vertically flip a matrix randomly with proba 0.5
    * RandomResizedCrop     : Crop a given matrix to random size and aspect ratio.
    * LinearTransformation  : Perform a linear transform of a tensor matrix
    * NoiseGaussian         : Adds gaussian noise to a given matrix.
    * OutliersZeroRandom    : Randomly add zeroed-out outliers (without structure)
    * FilterMeanRandom      : Perform randomized abundance filtering (under mean)
    * MaskRows              : Put random rows to zeros
    * MaskColumns           : Put random columns to zeros
    
    Comments and issues
    -------------------
    None for the moment

    Contributors
    ------------
    Philippe Esling (esling@ircam.fr)
    
"""
import numpy as np
#from skimage.transform import rescale

class Rescale(object):
    """
    Rescale a matrix to a given size.

    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of matrix edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        rescaled = rescale(data, (new_h, new_w))
        return rescaled

class CropRandom(object):
    """
    Crop randomly a matrix.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        data = data[top:(top + new_h),left:(left + new_w)]
        return data

class CropCenter(object):
    """
    Crops a given matrix at the center.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data):
        """
        Args:
            data: Data to be cropped
        Returns:
            Numpy array: Cropped data.
        """
        h, w = data.shape
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return data[int(i-th/2):int(i+th/2), int(j-tw/2):int(j+tw/2)]

class Normalize(object):
    """
    Normalize an tensor with given mean (M1,...,Mn) and std (S1,..,Sn) 
    for n channels. This transform will normalize each channel of the input 
    input[channel] = (input[channel] - mean[channel]) / std[channel]
    
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Data of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized data
        """
        retData = data.copy()
        for c in range(retData.shape[0]):
            retData[c] = (data[c] - self.mean[c]) / self.std[c]
        return retData

class NormalizeDimension(object):
    """
    Normalize a tensor by computing mean and std over a specified dimension.
    Then normalize the arrray across the given dimension.
    
    Args:
        dim (int): Dimension across which to perform normalization
    """

    def __init__(self, dim):
        self.dimension = dim

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Data of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized data
        """
        retData = data.copy()
        retData.swapaxes(0, self.dimension)
        retData = (retData - np.mean(retData, axis=0)) / np.std(retData, axis=0)
        retData.swapaxes(0, self.dimension)
        return retData

class RandomCrop(object):
    """
    Crop a given matrix at a random location.
    
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be cropped.
        Returns:
            Numpy array: Cropped matrix.
        """
        h, w = data.shape
        th, tw = self.size
        if w == tw and h == th:
            i = 0;
            j = 0;
        else:
            i = np.random.randint(0, h - th)
            j = np.random.randint(0, w - tw)
        if self.padding > 0:
            data = np.pad(data, self.padding)
        return data[i:(i + w), j:(j + h)]


class RandomHorizontalFlip(object):
    """
    Horizontally flip a given matrix randomly with a probability of 0.5.
    """

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be flipped.
        Returns:
            Numpy array: Randomly flipped matrix.
        """
        if np.random.random() < 0.5:
            return np.flipud(data)
        return data


class RandomVerticalFlip(object):
    """
    Vertically flip a given matrix randomly with a probability of 0.5.
    """

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be flipped.
        Returns:
            Numpy array: Randomly flipped matrix.
        """
        if np.random.random() < 0.5:
            return np.fliplr(data)
        return data


class RandomResizedCrop(object):
    """
    Crop a given matrix to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    
    Args:
        size: expected output size of each edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, data):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i = None
        for attempt in range(10):
            area = data.shape[0] * data.shape[1]
            target_area = np.random.uniform(0.08, 1.0) * area
            aspect_ratio = np.random.uniform(3. / 4, 4. / 3)
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            if np.random.random() < 0.5:
                w, h = h, w
            if w <= data.shape[0] and h <= data.shape[1]:
                i = np.random.randint(0, data.shape[1] - h)
                j = np.random.randint(0, data.shape[0] - w)
                break
        if (i == None):
            # Fallback
            w = min(data.shape[0], data.shape[1])
            i = (data.shape[1] - w) // 2
            j = (data.shape[0] - w) // 2
        data = data[i:(i + w), j:(j + h)]
        return np.resize(data, self.size)

class LinearTransformation(object):
    """
    Transform a tensor matrix with a square transformation matrix computed offline.
    Given transformation_matrix, will flatten the array, compute the dot
    product with the transformation matrix and reshape the tensor to its
    original shape. Applications:
    - whitening: zero-center the data, compute the data covariance matrix
                 [D x D] with np.dot(X.T, X), perform SVD on this matrix and
                 pass it as transformation_matrix.
    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        flat_tensor = tensor.copy()
        transformed_tensor = np.dot(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.shape)
        return tensor

class NoiseGaussian(object):
    """
    Adds gaussian noise to a given matrix.
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """

    def __init__(self, factor=1e-5):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data + (np.random.randn(data.shape[0], data.shape[1]) * self.factor)
        return data;
    
class OutliersZeroRandom(object):
    """
    Randomly add zeroed-out outliers (without structure)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.25):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Tensor with randomly zeroed-out outliers
        """
        dataSize = data.size
        tmpData = data.copy();
        # Add random outliers (here similar to dropout mask)
        tmpIDs = np.floor(np.random.rand(int(np.floor(dataSize * self.factor))) * dataSize)
        for i in range(tmpIDs.shape[0]):
            if (tmpIDs[i] < data.size):
                tmpData.ravel()[int(tmpIDs[i])] = 0
        return tmpData
    
class FilterMeanRandom(object):
    """
    Perform randomized abundance filtering (under mean)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.25):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data.copy()
        meanVal = np.mean(data[data > 0])
        cutThresh = np.random.rand(1) * (meanVal / 4)
        for iS in range(data.size):
            if (data.ravel()[iS] < cutThresh[0]):
                data.ravel()[iS] = 0;
        return data

class Binarize(object):
    """
    Binarize a given matrix
    """

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data[data > 0] = 1
        return data

class MaskRows(object):
    """
    Put random rows to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[0] * self.factor))) * (data.shape[0]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[0]:
                data[int(tmpIDs[i]), :] = 0
        return data

class MaskColumns(object):
    """
    Put random columns to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[1] * self.factor))) * (data.shape[1]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[1]:
                data[:, int(tmpIDs[i])] = 0
        return data