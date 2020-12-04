import numpy as np
import tensorflow as tf

def calib_error(pred_dist, y_true, use_circular_region=False):
    if y_true.ndim == 3:
        y_true = y_true[0]

    if use_circular_region or y_true.shape[-1] > 1:
        cdf_vals = pred_dist.cdf(y_true, circular=True)
    else:
        cdf_vals = pred_dist.cdf(y_true)
    cdf_vals = tf.reshape(cdf_vals, (-1, 1))

    num_points = tf.cast(tf.size(cdf_vals), tf.float32)
    conf_levels = tf.linspace(0.05, 0.95, 20)
    emp_freq_per_conf_level = tf.reduce_sum(tf.cast(cdf_vals <= conf_levels, tf.float32), axis=0) / num_points

    calib_err = tf.reduce_mean(tf.abs((emp_freq_per_conf_level - conf_levels)))
    return calib_err