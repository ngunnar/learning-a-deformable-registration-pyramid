import numpy as np
import tensorflow as tf

import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops

def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = tf.shape(arr)[-3:]
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0]:endind[0], startind[1]:endind[1], startind[2]:endind[2]]

def fftconv(in1, in2, mode="full"):
    """Reimplementation of ``scipy.signal.fftconv``."""
    # Reorder channels to come second (needed for fft)
    in1 = tf.transpose(in1, perm=[0, 4, 1, 2, 3])
    in2 = tf.transpose(in2, perm=[0, 4, 1, 2, 3])

    # Extract shapes
    s1 = tf.convert_to_tensor(tf.shape(in1)[-3:])
    s2 = tf.convert_to_tensor(tf.shape(in2)[-3:])
    shape = s1 + s2 - 1
    # Compute convolution in fourier space
    sp1 = tf.signal.rfft3d(in1, shape)
    sp2 = tf.signal.rfft3d(in2, shape)
    ret = tf.signal.irfft3d(sp1 * sp2, shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
        cropped.set_shape(in1.shape)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    # Reorder channels to last
    result = tf.transpose(cropped, perm=[0, 2, 3, 4, 1])
    return result

def gausssmooth(img,  std):
     # Create gaussian
    size = tf.cast(std * 5, 'int32')
    size = (size // 2) * 2 + 1
    size_f = tf.cast(size, 'float32')

    grid_x, grid_y, grid_z = tf.meshgrid(math_ops.range(size), math_ops.range(size), math_ops.range(size))
    
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')
    grid_z = tf.cast(grid_z[None, ..., None], 'float32')
    gaussian = tf.exp(-((grid_x - size_f / 2 - 0.5) ** 2 + 
                        (grid_y - size_f / 2 + 0.5) ** 2 + 
                        (grid_z - size_f / 2 + 0.5) ** 2) / std ** 2)
    gaussian = gaussian / tf.reduce_sum(gaussian)
    return fftconv(img, gaussian, 'same')

__all__ = ('dense_image_warp',)

def _interpolate_trilinear(grid,
                          query_points,
                          name='interpolate_trilinear',
                          indexing='ijk'):
    """
    Finds values for query points on a grid using trilinear interpolation.
    Args:
        grid: a 5-D float `Tensor` of shape `[batch, depth, height, width, channels]`.
        query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 3]`.
        name: a name for the operation (optional).
        indexing: whether the query points are specified as row and column (ijk),
        or Cartesian coordinates (xyz).
    Returns:
        values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
        ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ijk' and indexing != 'xyz':
        raise ValueError('Indexing mode must be \'ijk\' or \'xyz\'')

    with tf.name_scope(name):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)
        shape = grid.shape
        if len(shape) != 5:
            msg = 'Grid must be 5 dimensional. Received size: '
            raise ValueError(msg + str(grid.shape))

        batch_size, depth, height, width, channels = shape
        query_type = query_points.dtype
        grid_type = grid.dtype

        if (len(query_points.shape) != 3 or query_points.shape[2] != 3):
            msg = ('Query points must be 3 dimensional and size 3 in dim 3. Received size: ')
            raise ValueError(msg + str(query_points.shape))

        _, num_queries, _ = query_points.shape

        if depth < 2 or height < 2 or width < 2:
            msg = 'Grid must be at least batch_size x 2 x 2 x 2 in size. Received size: '
            raise ValueError(msg + str(grid.shape))

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1, 2] if indexing == 'ijk' else [2, 1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2)

        for dim in index_order:
            with tf.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.minimum(tf.maximum(min_floor, tf.floor(queries)), max_floor)
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

        if batch_size * depth * height * width > np.iinfo(np.int32).max / 8:
            error_msg = """The image size or batch size is sufficiently large
                         that the linearized addresses used by tf.gather
                         may exceed the int32 limit."""
            raise ValueError(error_msg)

        flattened_grid = tf.reshape(grid, [batch_size * depth * height * width, channels])
        batch_offsets = tf.reshape(tf.range(batch_size) * depth * height * width, [batch_size, 1])

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(z_coords, y_coords, x_coords, name):
            with tf.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + z_coords * height * width + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
            return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 8 corners around each query point
        c000 = gather(floors[0], floors[1], floors[2], 'c000')
        c001 = gather(floors[0], floors[1], ceils[2], 'c001')
        c010 = gather(floors[0], ceils[1], floors[2], 'c010')
        c011 = gather(floors[0], ceils[1], ceils[2], 'c011')
        c100 = gather(ceils[0], floors[1], floors[2], 'c100')
        c101 = gather(ceils[0], floors[1], ceils[2], 'c101')
        c110 = gather(ceils[0], ceils[1], floors[2], 'c110')
        c111 = gather(ceils[0], ceils[1], ceils[2], 'c111')
    
        # now, do the actual interpolation
        with tf.name_scope('interpolate'):
            c00 = c000 * (1 - alphas[2]) + c100*alphas[2]
            c01 = c001 * (1 - alphas[2]) + c101*alphas[2]
            c10 = c010 * (1 - alphas[2]) + c110*alphas[2]
            c11 = c011 * (1 - alphas[2]) + c111*alphas[2]

            c0 = c00 * (1 - alphas[1]) + c10 * alphas[1]
            c1 = c01 * (1 - alphas[1]) + c11 * alphas[1]

            interp = c0*(1 - alphas[0]) + c1*alphas[0]
        return interp

def dense_image_warp(image, flow, name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, k, j, i, c] is
    images[b, k - flow[b,k,j,i,0], j - flow[b, k, j, i, 1], i - flow[b, k, j, i, 2], c].
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by trilinear
    interpolation of the 8 nearest pixels around
    (b, k - flow[b, k, j, i, 0], j - flow[b, k, j, i, 1], i - flow[b, k, j, i, 2]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    Args:
        image: 5-D float `Tensor` with shape `[batch, depth, height, width, channels]`.
        flow: A 5-D float `Tensor` with shape `[batch, batch, height, width, 3]`.
        name: A name for the operation (optional).
        Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
        and do not necessarily have to be the same type.
    Returns:
        A 5-D float `Tensor` with shape`[batch, depth, height, width, channels]`
          and same type as input image.
      Raises:
        ValueError: if depth < 2 or height < 2 or width < 2 or the inputs have the wrong number
                    of dimensions.
    """
    with tf.name_scope(name):
        batch_size, depth, height, width, channels = image.shape
        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        
        grid_x, grid_y, grid_z = tf.meshgrid(tf.range(depth), 
                                            tf.range(height),
                                            tf.range(width),
                                            indexing ='ij')
        stacked_grid = tf.cast(tf.stack([grid_x, grid_y, grid_z], axis=3), flow.dtype)
    
        batched_grid = tf.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, depth * height * width, 3])
        # Compute values at the query points, then reshape the result back to the
        # image grid.    
        interpolated = _interpolate_trilinear(image, query_points_flattened)
        interpolated = tf.reshape(interpolated, [batch_size, depth, height, width, channels])
        return interpolated


import tensorflow as tf
import numpy as np
#from tfdeform.deform_util3D import dense_image_warp
#from tfdeform.convolve3D import gausssmooth


__all__ = ('random_deformation_linear',
           'random_deformation_momentum',
		   'random_deformation_momentum_sequence',
           'batch_random_deformation_momentum_sequence')

def image_gradients(image, mode='forward'):
    """Compute gradients of image."""
    if image.shape.ndims != 5:
        raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, d, h, w, dim], not %s.', image.shape)
    image_shape = tf.shape(image)
    batch_size, depth, height, width, dim = tf.unstack(image_shape)
    dz = image[:, 1:, :, :, :] - image[:, :-1, :, :, :]
    dy = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
    dx = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]

    if mode == 'forward':
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y,z) - I(x,y,z)] on the base pixel (x, y, z).
        shape = tf.stack([batch_size, 1, height, width, dim])
        dz = tf.concat([dz, tf.zeros(shape, image.dtype)], 1)
        dz = tf.reshape(dz, image_shape)
        
        shape = tf.stack([batch_size, depth, 1, width, dim])
        dy = tf.concat([dy, tf.zeros(shape, image.dtype)], 2)
        dy = tf.reshape(dy, image_shape)

        shape = tf.stack([batch_size, depth, height, 1, dim])
        dx = tf.concat([dx, tf.zeros(shape, image.dtype)], 3)
        dx = tf.reshape(dx, image_shape)
    else:
        # Return tensors with same size as original image by concatenating
        # zeros. Place the gradient [I(x+1,y,z) - I(x,y,z)] on the base pixel (x, y, z).
        shape = tf.stack([batch_size, 1, height, width, dim])
        dz = tf.concat([tf.zeros(shape, image.dtype), dz], 1)
        dz = tf.reshape(dz, image_shape)
        
        shape = tf.stack([batch_size, depth, 1, width, dim])
        dy = tf.concat([tf.zeros(shape, image.dtype), dy], 2)
        dy = tf.reshape(dy, image_shape)

        shape = tf.stack([batch_size, depth, height, 1, dim])
        dx = tf.concat([tf.zeros(shape, image.dtype), dx], 3)
        dx = tf.reshape(dx, image_shape)

    return dz, dy, dx

def jacobian(vf):
    """Compute the jacobian of a vectorfield pointwise."""
    vf0_dz, vf0_dy, vf0_dx = image_gradients(vf[..., 0:1])
    vf1_dz, vf1_dy, vf1_dx = image_gradients(vf[..., 1:2])
    vf2_dz, vf2_dy, vf2_dx = image_gradients(vf[..., 2:3])

    r1 = tf.concat([vf0_dz[..., None], vf0_dy[..., None], vf0_dx[..., None]], axis=-1)
    r2 = tf.concat([vf1_dz[..., None], vf1_dy[..., None], vf1_dx[..., None]], axis=-1)
    r3 = tf.concat([vf2_dz[..., None], vf2_dy[..., None], vf2_dx[..., None]], axis=-1)

    return tf.concat([r1, r2, r3], axis=-2)

def matmul(mat, vec):
    """Compute matrix @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 0, 1:2] * vec[..., 1:2]
    c13 = mat[..., 0, 2:3] * vec[..., 2:3]
    
    c21 = mat[..., 1, 0:1] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]
    c23 = mat[..., 1, 2:3] * vec[..., 2:3]
    
    c31 = mat[..., 2, 0:1] * vec[..., 0:1]
    c32 = mat[..., 2, 1:2] * vec[..., 1:2]
    c33 = mat[..., 2, 2:3] * vec[..., 2:3]
    
    return tf.concat([c11 + c12 + c13, c21 + c22 + c23, c31 + c32 + c33], axis=-1)

def matmul_transposed(mat, vec):
    """Compute matrix.T @ vec pointwise."""
    c11 = mat[..., 0, 0:1] * vec[..., 0:1]
    c12 = mat[..., 1, 0:1] * vec[..., 1:2]
    c13 = mat[..., 2, 0:1] * vec[..., 2:3]
    
    c21 = mat[..., 0, 1:2] * vec[..., 0:1]
    c22 = mat[..., 1, 1:2] * vec[..., 1:2]
    c23 = mat[..., 2, 1:2] * vec[..., 2:3]
    
    c31 = mat[..., 0, 2:3] * vec[..., 0:1]
    c32 = mat[..., 1, 2:3] * vec[..., 1:2]
    c33 = mat[..., 2, 2:3] * vec[..., 2:3]
    
    return tf.concat([c11 + c12 + c13, c21 + c22 + c23, c31 + c32 + c33], axis=-1)

def div(vf):
    """Compute divergence of vector field."""
    dz, _, _ = image_gradients(vf[..., 0:1], mode='backward')
    _, dy, _ = image_gradients(vf[..., 1:2], mode='backward')
    _, _, dx = image_gradients(vf[..., 2:3], mode='backward')
    return dx + dy + dz

def init(shape):
    assert len(shape) == 3
    grid_x, grid_y, grid_z = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), indexing ='ij')
    grid_x = tf.cast(grid_x[None, ..., None], 'float32')
    grid_y = tf.cast(grid_y[None, ..., None], 'float32')
    grid_z = tf.cast(grid_z[None, ..., None], 'float32')

    base_coordinates = tf.concat([grid_x, grid_y, grid_z], axis=-1)
    # Create mask to stop movement at edges
    mask = (tf.cos((grid_x - shape[0] / 2 + 1) * np.pi / (shape[0] + 2)) *
            tf.cos((grid_y - shape[1] / 2 + 1) * np.pi / (shape[1] + 2)) *
            tf.cos((grid_z - shape[2] / 2 + 1) * np.pi / (shape[2] + 2))) ** (0.25)
    return base_coordinates, mask

def batch_random_deformation_momentum_sequence(shape, std, distance, stepsize=0.1):
    r"""Create sequences of random diffeomorphic deformations.

    Parameters
    ----------
    shape : sequence of 4 ints
        Batch, depth, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        Generated deformation field for each step (Batch, step, depth, height, width)
    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """ 
    batch_size = shape[0]
    i = tf.constant(0, dtype=tf.int32)
    u0i, uji = random_deformation_momentum_sequence(shape[1:], std, distance, stepsize)
    def cond(i, u0i, uji):
        return i < batch_size - 1

    def body(i, u0i, uji):
        u1, u2 = random_deformation_momentum_sequence(shape[1:], std, distance, stepsize)
        u0i, = tf.concat([u0i, u1[None,...]], axis=0)
        uji = tf.concat([uji, u2[None,...]], axis=0)
        print(i, u0i.shape, uji.shape)
        return i + 1, u0i, uji
    
    i, u0i, uji = tf.while_loop(
        cond, body, [i, u0i, uji],
        shape_invariants=[0,tf.TensorShape([batch_size, None, *shape[1:], 3])])
    return u0i, uji
    
def random_deformation_momentum_sequence(shape, std, distance, stepsize=0.1):
    r"""Create a sequence of random diffeomorphic deformations.

    Parameters
    ----------
    shape : sequence of 3 ints
        depth, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        Generated deformation field for each step (step, depth, height, width)
    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """
    assert len(shape) == 3
    base_coordinates, mask  = init(shape)
    coordinates = tf.identity(base_coordinates)

    # Total distance is given by std * n_steps * dt, we use this
    # to work out the exact numbers.
    n_steps = tf.cast(tf.math.ceil(distance / (std * stepsize)), 'int32')
    dt = distance / (tf.cast(n_steps, 'float32') * std)

    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std ** 2

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = dt * C * tf.random.normal(shape=[1, *shape, 3])

    # Using a while loop, generate the deformation step-by-step.
    def cond(i, from_coordinates, momenta):
        return i < n_steps

    def body(i, from_coordinates, momenta):
        v = mask * gausssmooth(momenta, std)

        d1 = matmul_transposed(jacobian(momenta), v)
        d2 = matmul(jacobian(v), momenta)
        d3 = div(v) * momenta
        f_c = tf.identity(from_coordinates[-1,...][None,...])
        momenta = momenta - dt * (d1 + d2 + d3)
        v = dense_image_warp(v, f_c - base_coordinates)
        f_c = dense_image_warp(f_c, v)
        from_coordinates = tf.concat([from_coordinates, f_c], axis=0)
        return i + 1, from_coordinates, momenta

    i = tf.constant(0, dtype=tf.int32)
    i, from_coordinates, momenta = tf.while_loop(
        cond, body, [i, coordinates, momenta],
        shape_invariants=[0,tf.TensorShape([None,*shape, 3]),0])

    from_total_offset = from_coordinates - base_coordinates
    from_total_diff = np.diff(np.concatenate([base_coordinates, from_coordinates], axis=0), axis=0)

    return from_total_offset, from_total_diff

def random_deformation_momentum(shape, std, distance, stepsize=0.1):
    r"""Create a random diffeomorphic deformation.

    Parameters
    ----------
    shape : sequence of 4 ints
        Batch, depth, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.
    stepsize : float
        How large each step should be (as a propotion of ``std``).
    Returns:
        The end generated deformation field (Batch, depth, height, width)

    Notes
    -----
    ``distance`` should typically not be more than a small fraction of the
    sidelength of the image.

    The computational time is is propotional to

    .. math::
        \frac{distance}{std * stepsize}
    """
    base_coordinates, mask  = init(shape[1:])
    base_coordinates = tf.repeat(base_coordinates, shape[0], axis=0)
    coordinates = tf.identity(base_coordinates)

    # Total distance is given by std * n_steps * dt, we use this
    # to work out the exact numbers.
    n_steps = tf.cast(tf.math.ceil(distance / (std * stepsize)), 'int32')
    dt = distance / (tf.cast(n_steps, 'float32') * std)

    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std ** 2

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = dt * C * tf.random.normal(shape=[*shape, 3])

    # Using a while loop, generate the deformation step-by-step.
    def cond(i, from_coordinates, momenta):
        return i < n_steps

    def body(i, from_coordinates, momenta):
        v = mask * gausssmooth(momenta, std)

        d1 = matmul_transposed(jacobian(momenta), v)
        d2 = matmul(jacobian(v), momenta)
        d3 = div(v) * momenta
        momenta = momenta - dt * (d1 + d2 + d3)
        v = dense_image_warp(v, from_coordinates - base_coordinates)
        from_coordinates = dense_image_warp(from_coordinates, v)

        return i + 1, from_coordinates, momenta

    i = tf.constant(0, dtype=tf.int32)
    i, from_coordinates, momenta = tf.while_loop(
        cond, body, [i, coordinates, momenta])

    from_total_offset = from_coordinates - base_coordinates

    return from_total_offset

def random_deformation_linear(shape, std, distance):
    r"""Create a random deformation.

    Parameters
    ----------
    shape : sequence of 4 ints
        Batch, depth, height and width.
    std : float
        Correlation distance for the linear deformations.
    distance : float
        Expected total effective distance for the deformation.

    Notes
    -----
    ``distance`` must be significantly smaller than ``std`` to guarantee that
    the deformation is smooth.
    """
    _, mask = init(shape[1:])
    # Scale to get std 1 after smoothing
    C = np.sqrt(2 * np.pi) * std

    # Multiply by dt here to keep values small-ish for numerical purposes
    momenta = distance * C * tf.random.normal(shape=[*shape, 3])
    v = mask * gausssmooth(momenta, std)

    return v

def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0