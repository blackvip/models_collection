import numpy as np
import tensorflow as tf

# A - truth, B - pred
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)

def get_iou_vector_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = get_iou_vector(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def faster_iou_metric(label, pred):
    metric_value = tf.py_func(get_iou_vector_batch, [label, pred], tf.float32)
    return metric_value


def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0)
    if u == 0:
        return u
    return i/u

def compute_ious(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    num_preds  = len(imgs_pred)
    scores = np.zeros((num_images, num_preds))
    for i in range(num_images):
        for j in range(num_preds):
            if imgs_true[i].sum() == imgs_pred[j].sum() == 0:
                scores[i, j] = 1
            else:
                scores[i, j] = iou(imgs_true[i], imgs_pred[j])
    return scores

def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)

def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return np.float32(sum(precisions) / len(precisions))

def threshold_mean_iou_metric(gt, predictions):
    metric_value = tf.py_func(compute_eval_metric, [gt, predictions], tf.float32)
    return metric_value

def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

metrics_dict = {
                'accuracy' : 'accuracy', 
                'mean_iou' : mean_iou,
                'faster_iou_metric' : faster_iou_metric,
                'threshold_mean_iou_metric' : threshold_mean_iou_metric,
               }