
import numpy as np
import math
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa



transforms = ['Identity', 'AutoContrast', 'Equalize',
'Solarize', 'Color', 'Posterize',
'Contrast', 'Brightness', 'Sharpness',
'ShearX', 'ShearY', 'TranslateX', 'TranslateY']


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

def level_to_arg(translate_const = 100):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Invert': lambda level: (),
        'Rotate': _rotate_level_to_arg,
        'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
        'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
        'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        # pylint:disable=g-long-lambda
        'TranslateX': lambda level: _translate_level_to_arg(
            level, translate_const),
        'TranslateY': lambda level: _translate_level_to_arg(
            level, translate_const),
        # pylint:enable=g-long-lambda
    }


def _enhance_level_to_arg(level):
    return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
    level = (level/_MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _translate_level_to_arg(level, translate_const):
    level = (level/_MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)



def randaugment(N = 2, M = 10):
    """Generate a set of distortions.
    Args:
    N: Number of augmentation transformations to
    apply sequentially.
    M: Magnitude for all the transformations.
    """
    sampled_ops = np.random.choice(transforms, N)
    return [(op, M) for op in sampled_ops]


def distort_image_with_randaugment(image, num_layers, magnitude=None):
    """Applies the RandAugment policy to `image`.
        RandAugment is from the paper https://arxiv.org/abs/1909.13719,
        Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        num_layers: Integer, the number of augmentation transformations to apply
            sequentially to an image. Represented as (N) in the paper. Usually best
            values will be in the range [1, 3].
        magnitude: Integer, shared magnitude across all augmentation operations.
            Represented as (M) in the paper. Usually best values are in the range
            [5, 30].
        Returns:
        The augmented version of `image`.
    """
    replace_value = [128] * 3
    tf.logging.info('Using RandAug.')

    available_ops = [
        'AutoContrast', 'Equalize', 'Posterize',
        'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
        'ShearX', 'ShearY', 'TranslateX', 'TranslateY']

    #for layer_num in range(num_layers):
        """op_to_select = tf.random_uniform(
            [], maxval=len(available_ops), dtype=tf.int32)
        random_magnitude = float(magnitude)"""
    ops = randaugment()

    #with tf.name_scope('randaug_layer_{}'.format(layer_num)):
    for (op_name, mag) in enumerate(ops):
        func, args = NAME_TO_FUNC[op_name], level_to_arg()[name](level)
        # pylint:enable=g-long-lambda
        image = lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args)
    
    return image


def solarize(image, threshold=128):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def color(image, factor):
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image, factor):
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def translate_x(image, pixels, replace):
    """Equivalent of PIL Translate in X dimension."""
    # Deprecated in tf 1.13 or 1.14 #image = contrib_image.translate(wrap(image), [-pixels, 0])
    result = tfa.image.translate(wrap(image), [-pixels, 0])
    return unwrap(result, replace)


def translate_y(image, pixels, replace):
    """Equivalent of PIL Translate in Y dimension."""
    # Deprecateed # image = contrib_image.translate(wrap(image), [0, -pixels])
    result = tfa.image.translate(wrap(image), [0, -pixels])
    return unwrap(result, replace)


def shear_x(image, level, replace):
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    # Deprecated #image = contrib_image.transform(
    #  wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
    
    result = tfa.image.shear_x(image, level, replace)
    return result


def shear_y(image, level, replace):
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    
    result = tfa.image.shear_y(image, level, replace)
    return result


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
    Args:
    image: A 3D uint8 tensor.
    Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
    """

    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.to_float(tf.reduce_min(image))
        hi = tf.to_float(tf.reduce_max(image))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.to_float(im) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    with tf.device('/cpu:0'):
        # Some augmentation that uses depth-wise conv will cause crashing when
        # training on GPU. See (b/156242594) for details.
        degenerate = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding='VALID', rate=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image):
    """Implements Equalize function from PIL using TF ops."""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


# In[ ]:





# In[ ]:




