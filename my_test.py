import tensorflow as tf
import yaml
import os
from official.vision.modeling.heads.dense_prediction_heads import RetinaNetHead


from official.vision.serving import detection
from official.core.config_definitions import ExperimentConfig, TaskConfig, TrainerConfig, RuntimeConfig, base_config
from official.vision.configs import retinanet
from official.vision.tasks import RetinaNetTask
import dataclasses
import numpy as np
import pandas as pd
# Load YAML file

#pipeline_config = "models/coco_spinenet49_mobile_tpu.yaml"
pipeline_config = "models/retinanet/resnet50fpn_coco_tfds_tpu.yaml"
with tf.io.gfile.GFile(pipeline_config, 'r') as yaml_file:
    yaml_string = yaml_file.read()
model_yaml = yaml.safe_load(yaml_string)

# Parse YAML string
@dataclasses.dataclass
class RetinanetExperimentConfig(base_config.Config):
    task: TaskConfig = retinanet.RetinaNetTask()
    trainer: TrainerConfig = TrainerConfig()
    runtime: RuntimeConfig = RuntimeConfig()

cfg = RetinanetExperimentConfig.from_yaml(pipeline_config)
cfg.task.model.num_classes = 91 #80: 819, 720
det = detection.DetectionModule(cfg, batch_size=1, input_image_size=[384, 384])
ckpt = tf.compat.v2.train.Checkpoint(model=det.model)
print(ckpt.restore("models/retinanet/ckpt-33264").expect_partial())
x1 = np.expand_dims(tf.keras.preprocessing.image.load_img("zidane.jpg", target_size=(384, 384)), 0)
x2 = np.expand_dims(tf.keras.preprocessing.image.load_img("james-resly-XOxsVJmCyxk-unsplash.jpg", target_size=(384, 384)), 0)
x = np.concatenate([x1,x2], axis=0,dtype=np.float32)
res = det.serve(x1)
import matplotlib.pyplot as plt
import cv2
im_res = np.array(x[0])
bbox = np.array(res["detection_boxes"][0][0], dtype=np.int32)
cv2.rectangle(im_res,
              (bbox[1],bbox[0]), (bbox[3],bbox[2]),
              color=(0,0,0))
plt.imshow(np.array(im_res, dtype=np.uint8))
import tensorflow as tf


class ModelWrapper():
    def __init__(self, model, HEIGHT, WIDTH, nb_classes, threshold=0.25):
        self.model = model
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.nb_classes = nb_classes
        self.threshold = threshold

    '''
    def call(self, x):
        return self.__call__(x)
    '''

    def __call__(self, x):
        tf.print("batch shape", x.shape)

        preds = self.model.serve(x)

        preds = tf.concat([preds["detection_boxes"],
                           tf.expand_dims(preds["detection_scores"], axis=-1),
                           tf.expand_dims(tf.cast(preds["detection_classes"], tf.float32), axis=-1)], axis=-1)
        tf.print("preds shape", preds.shape)
        y = self.get_boxes_constant_size_map(preds)
        tf.print("printing output------------")
        tf.print(tf.shape(y))
        # tf.print(y[0])
        return y

    def get_boxes(self, preds):
        batch_results = []
        for pred in preds:
            boxes = pred[0] * tf.constant([self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT], dtype=tf.float32)
            results = tf.concat([boxes, tf.expand_dims(pred[2], axis=-1),
                                 tf.cast(tf.one_hot(tf.cast(pred[1], tf.int32), self.nb_classes, 1), tf.float32)],
                                axis=-1)

            batch_results.append(results)
        return batch_results

    def get_boxes_constant_size_map(self, preds):
        def pred_loop(pred):
            # tf.print(pred)
            # tf.print(pred[0])
            # tf.print(pred[0].shape)

            # pred = pred[pred[..., 5] > self.threshold]
            boxes = pred[..., :4] * tf.constant([self.WIDTH, self.HEIGHT, self.WIDTH, self.HEIGHT], dtype=tf.float32)
            scores = pred[..., 5]
            cls = pred[..., 4]
            one_hot_cls = tf.cast(tf.one_hot(tf.cast(cls, tf.int32), self.nb_classes, 1), tf.float32)

            results = tf.concat([boxes, tf.expand_dims(scores, axis=-1),
                                 one_hot_cls], axis=-1)

            # tf.stack(boxes, scores, )
            # print(results.shape)
            # batch_results.append(results)
            return results

        batch_results = tf.map_fn(pred_loop, preds, infer_shape=False)
        return batch_results

    @property
    def layers(self):
        return self.model.layers

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @staticmethod
    def from_config(conf_wrapper):
        model_wrapper = conf_wrapper["model_wrapper"]
        config = conf_wrapper["config"]
        keras_model = model_wrapper.model.__class__.from_config(config)
        keras_model.decode_predictions = model_wrapper.model.decode_predictions
        return ModelWrapper(keras_model, model_wrapper.HEIGHT, model_wrapper.WIDTH,
                            model_wrapper.nb_classes, model_wrapper.threshold)

    def get_config(self):
        return {"config": self.model.get_config(), "model_wrapper": self}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)
import os
from pathlib import Path
import numpy as np

from xplique.attributions import SobolAttributionMethod
from xplique.attributions import IntegratedGradients
from xplique.attributions import SmoothGrad, Saliency, DeconvNet, GradientInput, GuidedBackprop
from object_detector import BoxIouCalculator,ImageObjectDetectorScoreCalculator, YoloObjectFormater

import matplotlib.pyplot as plt
from xplique.plots.image import _normalize, _clip_percentile
from xplique.metrics import Deletion, Insertion, MuFidelity, AverageStability

class Explainer:
    methods = {
        "ig":IntegratedGradients,
        "saliency":Saliency,
        "deconvnet":DeconvNet,
        "gradient_input":GradientInput,
        "guided_backprop": GuidedBackprop,
        "sobol": SobolAttributionMethod,
        "smoothgrad": SmoothGrad
    }
    metrics = {
        "deletion": Deletion,
        "insertion": Insertion,
        "mufidelity": MuFidelity,
        "average_stability": AverageStability
    }
    def __init__(self, model, nb_classes=20):
        self.model = model
        self.score_calculator = ImageObjectDetectorScoreCalculator(
            YoloObjectFormater(), BoxIouCalculator())
        self.explainer_wrapper = None
        self.nb_classes = nb_classes
        self.last_params = None

    def apply(self, method_name, preds, img, params):
        self.last_params = params.copy()
        self.last_img = img
        wrapper = ModelWrapper(self.model, img.shape[1], img.shape[2], self.nb_classes)
        params["operator"] = self.score_calculator.tf_batched_score
        self.explainer_wrapper = self.methods[method_name](wrapper, **self.last_params)
        self.last_expl = self.explainer_wrapper.explain(img, wrapper.get_boxes(preds))
        self.last_params["method"] = method_name
        return self.last_expl

    def score(self, method, explanation, img, preds, params={}):
        wrapper = ModelWrapper(self.model, img.shape[1], explanation.shape[2], self.nb_classes)
        #operator_batched = operator_batching()
        if method != "average_stability":
            params["operator"] = self.score_calculator.tf_batched_score
            metric = self.metrics[method](wrapper, np.array(img), wrapper.get_boxes(preds), **params)
            return metric(explanation)

        metric = self.metrics[method](wrapper, np.array(img), wrapper.get_boxes(preds), **params)
        return metric(self.explainer_wrapper)

    def visualize(self, alpha=0.5, cmap="jet", clip_percentile=0.5, figsize=(10,10)):
        image = self.last_img[-1]
        expl = self.last_expl[-1]
        plt.figure(figsize=figsize)
        if image is not None:
            image = _normalize(image)
            plt.imshow(image)
        if expl.shape[-1] == 3:
            expl = np.mean(expl, -1)
        if clip_percentile:
            expl = _clip_percentile(expl, clip_percentile)
        if not np.isnan(np.max(np.min(expl))):
            expl = _normalize(expl)

        plt.imshow(expl, cmap=cmap, alpha=alpha)
        plt.axis('off')
"""
method = "saliency"
params = {
    "batch_size": 16
}
"""
method = "smoothgrad"
params = {
    "batch_size": 16,
    "nb_samples": 16,
    "noise": 0.069
}
pred = np.zeros((1,6), dtype=np.float32)
pred[...,:4] = bbox
pred[...,4] = res["detection_scores"][0][0]
pred[...,5] = res["detection_classes"][0][0]
explainer = Explainer(det)
explanation = explainer.apply(method, pred, x1, params)
explainer.visualize()
plt.show()