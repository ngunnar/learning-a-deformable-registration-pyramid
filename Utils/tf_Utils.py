import numpy as np
import tensorflow as tf
#from Utils.tensorflow_addons import dense_image_warp3D
import copy

#def warp_flow(img, flow):    
#    return dense_image_warp3D(img, flow)

def draw_hsv(flow):
    fy, fx = flow[:, :, :, 0], flow[:, :, :, 1]
    v, ang = cart_to_polar_ocv(fx, fy)

    h = normalize(tf.math.multiply(ang, 180 / np.pi))
    s = tf.ones_like(h)
    v = normalize(v)

    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255

    return tf.cast(rgb, tf.uint8)

def cart_to_polar_ocv(x, y, angle_in_degrees=False):
    v = tf.math.sqrt(tf.add(tf.square(x), tf.square(y)))
    ang = atan2_ocv(y, x)
    scale = 1 if angle_in_degrees else np.pi / 180
    return v, tf.math.multiply(ang, scale)


def atan2_ocv(y, x):
    # constants
    DBL_EPSILON = 2.2204460492503131e-16
    atan2_p1 = 0.9997878412794807 * (180 / np.pi)
    atan2_p3 = -0.3258083974640975 * (180 / np.pi)
    atan2_p5 = 0.1555786518463281 * (180 / np.pi)
    atan2_p7 = -0.04432655554792128 * (180 / np.pi)

    ax, ay = tf.abs(x), tf.abs(y)
    c = tf.where(tf.greater_equal(ax, ay), tf.math.divide(ay, ax + DBL_EPSILON),
                  tf.math.divide(ax, ay + DBL_EPSILON))
    c2 = tf.math.square(c)
    angle = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c
    angle = tf.where(tf.greater_equal(ax, ay), angle, 90.0 - angle)
    angle = tf.where(tf.less(x, 0.0), 180.0 - angle, angle)
    angle = tf.where(tf.less(y, 0.0), 360.0 - angle, angle)
    return angle

def normalize(tensor, a=0, b=1):
    return tf.math.divide(tf.math.multiply(tf.math.subtract(tensor, tf.reduce_min(tensor)), b - a),
                  tf.math.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))

