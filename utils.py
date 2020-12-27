import keras.models
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import f1_score as f_score, iou_score as jaccard_score

from deeplabv3plus import BilinearUpsampling
from metrics import iou_metric_all, iou_metric_fronts, iou_metric_hot, iou_metric_cold, iou_metric_stationary, \
    iou_metric_occlusion, iou_metric_binary, weighted_categorical_crossentropy, mixed_loss_gen


def load_model(model):
    return keras.models.load_model(model, custom_objects={
        "weighted_jaccard_loss": jaccard_loss,
        "iou_metric_all": iou_metric_all,
        "iou_metric_fronts": iou_metric_fronts,
        "iou_metric_hot": iou_metric_hot,
        "iou_metric_cold": iou_metric_cold,
        "iou_metric_stationary": iou_metric_stationary,
        "iou_metric_occlusion": iou_metric_occlusion,
        "iou_metric_binary": iou_metric_binary,
        "weighted_iou_score": jaccard_score,
        "weighted_f_score": f_score,
        "BilinearUpsampling": BilinearUpsampling,
    })
