import numpy as np

iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0)
    if u == 0:
        return u
    return i/u

def iou_metric_batch(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = iou(imgs_true[i], imgs_pred[i])
    return scores.mean()

def mean_iou(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            # scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
            scores[i] = np.mean([iou(imgs_true[i], (threshold<=imgs_pred[i]).astype('uint8')) for threshold in iou_thresholds])
            
    return scores.mean()

def dsb_mean_iou(true_masks, pred_masks):
    """
    Due to https://www.kaggle.com/wcukierski/example-metric-implementation
    """

    def update_shape(shape):
        shape_list = list(shape)
        if 1 in shape_list:
            shape_list.remove(1)
        return tuple(shape_list)
    true_masks = true_masks.reshape(update_shape(true_masks.shape))
    pred_masks = pred_masks.reshape(update_shape(pred_masks.shape))

    y_pred = np.sum((pred_masks.T*np.arange(1, len(pred_masks)+1)).T, axis=0)
    y_true = np.sum((true_masks.T*np.arange(1, len(true_masks)+1)).T, axis=0)
    #
    num_pred = y_pred.max()
    num_true = y_true.max()
    if num_pred < 1:
        num_pred = 1
        y_pred[0, 0] = 1
    # Compute intersection between all objects
    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(num_true, num_pred))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=num_true)[0]
    area_pred = np.histogram(y_pred, bins=num_pred)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp*1.0 / (tp + fp + fn)
        else:
            p = 0

        prec.append(p)

    return np.mean(prec)


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(
        false_positives), np.sum(false_negatives)
return tp, fp, fn

