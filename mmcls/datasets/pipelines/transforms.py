import math
import random

import mmcv
import numpy as np
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from PIL import Image
import cv2
from ..builder import PIPELINES

@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self, bcsh):
        self.bcsh = bcsh
        self.trans = transforms.ColorJitter(*bcsh)

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            PIL_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
            PIL_img = self.trans(PIL_img)
            img = cv2.cvtColor(np.asarray(PIL_img),cv2.COLOR_RGB2BGR) 
            results[key] = img
        return results
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        a,b,c,d = self.bcsh
        format_string += f'brightness={a},contrast={b},saturation={c},hue={d}'
        return format_string

@PIPELINES.register_module()
class RandomCrop(object):
    """Crop the given Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            -constant: Pads with a constant value, this value is specified
                with pad_val.
            -edge: pads with the last value at the edge of the image.
            -reflect: Pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            -symmetric: Pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 pad_val=0,
                 padding_mode='constant'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        xmin = random.randint(0, height - target_height)
        ymin = random.randint(0, width - target_width)
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.padding is not None:
                img = mmcv.impad(
                    img, padding=self.padding, pad_val=self.pad_val)

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = mmcv.impad(
                    img,
                    padding=(0, self.size[0] - img.shape[0], 0,
                             self.size[0] - img.shape[0]),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = mmcv.impad(
                    img,
                    padding=(self.size[1] - img.shape[1], 0,
                             self.size[1] - img.shape[1], 0),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            xmin, ymin, height, width = self.get_params(img, self.size)
            results[key] = mmcv.imcrop(
                img,
                np.array([ymin, xmin, ymin + width - 1, xmin + height - 1]))
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Default: (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Default: (3. / 4., 4. / 3.).
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Default:
            'bilinear'.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max). "
                             f"But received {scale}")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped and resized.

        Returns:
            ndarray: Randomly cropped and resized image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            xmin, ymin, target_height, target_width = self.get_params(
                img, self.scale, self.ratio)
            img = mmcv.imcrop(
                img,
                np.array([
                    ymin, xmin, ymin + target_width - 1,
                    xmin + target_height - 1
                ]))
            results[key] = mmcv.imresize(
                img, tuple(self.size[::-1]), interpolation=self.interpolation)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@PIPELINES.register_module()
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of gray_prob.

    Args:
        gray_prob (float): Probability that image should be converted to
            grayscale. Default: 0.1.

    Returns:
        ndarray: Grayscale version of the input image with probability
            gray_prob and unchanged with probability (1-gray_prob).
            - If input image is 1 channel: grayscale version is 1 channel.
            - If input image is 3 channel: grayscale version is 3 channel
                with r == g == b.

    """

    def __init__(self, gray_prob=0.1):
        self.gray_prob = gray_prob

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.

        Returns:
            ndarray: Randomly grayscaled image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            num_output_channels = img.shape[2]
            if random.random() < self.gray_prob:
                if num_output_channels > 1:
                    img = mmcv.rgb2gray(img)[:, :, None]
                    results[key] = np.dstack(
                        [img for _ in range(num_output_channels)])
                    return results
            results[key] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gray_prob={self.gray_prob})'


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class Resize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
    """

    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        if isinstance(size, int):
            size = (size, size)
        assert size[0] > 0 and size[1] > 0
        assert interpolation in ("nearest", "bilinear", "bicubic", "area",
                                 "lanczos")

        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.interpolation = interpolation

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            h, w = img.shape[:2]
            short_side = self.size[0]
            ignore_resize = False
            if (w <= h and w == short_side) or (h <= w
                                                and h == short_side):
                ignore_resize = True
            else:
                if w < h:
                    width = short_side
                    height = int(short_side * h / w)
                else:
                    height = short_side
                    width = int(short_side * w / h)
            if not ignore_resize:
                img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    )
            results[key] = img
            results['img_shape'] = img.shape
            # img = mmcv.imresize(
            #     results[key],
            #     size=(self.width, self.height),
            #     interpolation=self.interpolation,
            #     return_scale=False)
            # results[key] = img
            # results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping, (h, w).

    Notes:
        If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                              and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_height, img_width, _ = img.shape

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class ThreeCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            image_h = img.shape[0]
            image_w = img.shape[1]
            crop_w, crop_h = self.crop_size
            assert crop_h == image_h or crop_w == image_w

            if crop_h == image_h:
                w_step = (image_w - crop_w) // 2
                offsets = list()
                offsets.append((0, 0))  # left
                offsets.append((2 * w_step, 0))  # right
                offsets.append((w_step, 0))  # middle
            elif crop_w == image_w:
                h_step = (image_h - crop_h) // 2
                offsets = list()
                offsets.append((0, 0))  # top
                offsets.append((0, 2 * h_step))  # down
                offsets.append((0, h_step))  # middle

            oversample_group = list()
            for o_w, o_h in offsets:
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1]))
                oversample_group.append(crop)
                flip_crop = mmcv.imflip(crop)
                oversample_group.append(flip_crop)
                # if is_flow and i % 2 == 0:
                #     flip_group.append(mmcv.iminvert(flip_crop))
                # else:
                #     flip_group.append(flip_crop)

                # oversample_group.extend(normal_group)
                # oversample_group.extend(flip_group)
            img_crops = np.concatenate(oversample_group, axis=0)
            results[key] = img_crops
            img_shape = img_crops.shape
        results['img_shape'] = img_shape 
        return results


@PIPELINES.register_module()
class TenCrop:
    """Crop the images into 10 crops (corner + center + flip).
    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".
    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        """Performs the TenCrop augmentation.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]

            img_h, img_w = img.shape[:2]
            crop_w, crop_h = self.crop_size

            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4

            offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]

            img_crops = list()
            crop_bboxes = list()
            for x_offset, y_offsets in offsets:
                crop = img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                flip_crop = np.flip(crop, axis=1).copy() 
                img_crops.append(crop)
                img_crops.append(flip_crop)
            img_crops = np.concatenate(img_crops, axis=0)
            results[key] = img_crops
            img_shape = img_crops.shape
        results['img_shape'] = img_shape 

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class ThreeTenCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            
            ms_imgs = []
            # for short_side in [256,288,320,352]:
            for short_side in [256,]:
                h, w = img.shape[:2]
                # short_side = self.size[0]
                ignore_resize = False
                if (w <= h and w == short_side) or (h <= w
                                                    and h == short_side):
                    ignore_resize = True
                else:
                    if w < h:
                        width = short_side
                        height = int(short_side * h / w)
                    else:
                        height = short_side
                        width = int(short_side * w / h)
                if not ignore_resize:
                    s_img = mmcv.imresize(
                        img,
                        size=(width, height),
                        interpolation='bilinear',
                        return_scale=False,
                        )
                else: 
                    s_img = img

                image_h = s_img.shape[0]
                image_w = s_img.shape[1]
                crop_w, crop_h = short_side, short_side 
                assert crop_h == image_h or crop_w == image_w

                if crop_h == image_h:
                    w_step = (image_w - crop_w) // 2
                    offsets = list()
                    offsets.append((0, 0))  # left
                    offsets.append((2 * w_step, 0))  # right
                    offsets.append((w_step, 0))  # middle
                elif crop_w == image_w:
                    h_step = (image_h - crop_h) // 2
                    offsets = list()
                    offsets.append((0, 0))  # top
                    offsets.append((0, 2 * h_step))  # down
                    offsets.append((0, h_step))  # middle

                oversample_group = list()
                for o_w, o_h in offsets:
                    crop = mmcv.imcrop(s_img, np.array(
                        [o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1]))
                    oversample_group.append(crop)
                    # flip_crop = mmcv.imflip(crop)
                    # oversample_group.append(flip_crop)
            
                img_h, img_w = oversample_group[0].shape[:2]
                crop_w, crop_h = 224,224 

                w_step = (img_w - crop_w) // 4
                h_step = (img_h - crop_h) // 4

                offsets = [
                    (0, 0),  # upper left
                    (4 * w_step, 0),  # upper right
                    (0, 4 * h_step),  # lower left
                    (4 * w_step, 4 * h_step),  # lower right
                    (2 * w_step, 2 * h_step),  # center
                ]

                for imgx in oversample_group:
                    for x_offset, y_offsets in offsets:
                        crop = imgx[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                        flip_crop = np.flip(crop, axis=1).copy() 
                        ms_imgs.append(crop)
                        ms_imgs.append(flip_crop)
                    resize_img = mmcv.imresize(imgx ,(224,224))
                    flip_resize_img = np.flip(resize_img, axis=1).copy()
                    ms_imgs.append(resize_img)
                    ms_imgs.append(flip_resize_img)
            img_crops = ms_imgs
            img_crops = np.concatenate(img_crops, axis=0)
            results[key] = img_crops
            img_shape = img_crops.shape
        results['img_shape'] = img_shape 

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str
