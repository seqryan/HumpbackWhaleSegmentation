from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

import logging
import random
import cv2
import json
import os
import numpy as np

from PIL import Image, ImageStat

# while running on local environment replace cv2_imshow with cv2.imshow()
from google.colab.patches import cv2_imshow

setup_logger()


def is_grayscale(path):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum)/3 == stat.sum[0]:  # check the avg with any element value
        return True  # if grayscale
    else:
        return False  # else its colour


def get_whale_dicts(img_dir, annotation_file):
    json_file = os.path.join(img_dir, annotation_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['_via_img_metadata'].values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        image = cv2.imread(filename)
        if (image is None):
            continue

        # uncomment  to skip grayscale images
        # if(is_grayscale(filename)):
        #     continue

        height, width = image.shape[:2]
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        has_annottaions = False  # only consider images with anotations
        for anno in annos:
            assert not anno["region_attributes"]
            has_annottaions = True
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs

        # only consider images with anotations
        if has_annottaions:
            dataset_dicts.append(record)
    return dataset_dicts


def register_dataset(dataset_dir, annotation_file, preview_annotations=False):
    # register dataset
    for d in ["train", "test"]:
        if "whale_" + d not in DatasetCatalog:  # skip if already registered
            data_path = os.path.join(dataset_dir, d)
            DatasetCatalog.register(
                "whale_" + d, lambda d=d: get_whale_dicts(data_path, annotation_file))
            MetadataCatalog.get("whale_" + d).set(thing_classes=["whale"])
    whale_metadata = MetadataCatalog.get("whale_train")

    if preview_annotations:
        data_path = os.path.join(dataset_dir, 'train')
        dataset_dicts = get_whale_dicts(data_path, annotation_file)
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=whale_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2_imshow(out.get_image()[:, :, ::-1])
    return whale_metadata


def train(weights_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("whale_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.OUTPUT_DIR = weights_dir
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # only has one class (whale).
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg


def load_model(weights_dir):
    """
    weights_dir: directory where model_final.pth is stored
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("whale_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final.pth")
    cfg.OUTPUT_DIR = weights_dir
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # only has one class (whale).
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    return cfg


def infer(dataset_dir, annotation_file, cfg, whale_metadata, num_samples=5):
    """
    cfg: configuration returned from the train function
    whale_metadata: metadata variable returned from register_dataset function
    """

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    data_path = os.path.join(dataset_dir, 'test')
    dataset_dicts = get_whale_dicts(data_path, annotation_file)
    for d in random.sample(dataset_dicts, num_samples):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        foreground = extract_subject_rgba(im, outputs)
        cv2_imshow(foreground)
        # combined = np.concatenate([im, foreground[:, :, :3]], axis=1)
        # cv2_imshow(combined)


def save_segmented_images(dataset_dir, save_dir, cfg, whale_metadata):
    """
    save_dir: directory where the segmented images will be saved
    cfg: configuration returned from the train function
    whale_metadata: metadata variable returned from register_dataset function
    """
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    for mode in ['train', 'test']:
        logging.info('Saving %s images ...' % mode)
        segmented_dir = os.path.join(save_dir, mode)
        try:
            os.mkdir(segmented_dir)
        except FileExistsError as e:
            logging.error('%s already exisits in %s ' % (mode, save_dir))

        mode_dir = os.path.join(dataset_dir, mode)
        for index, file_name in enumerate(os.listdir(mode_dir)):
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                image_path = os.path.join(mode_dir, file_name)
                im = cv2.imread(image_path)

                # file_name = file_name.split('.')[0] + '.png'
                segmented_path = os.path.join(segmented_dir, file_name)

                outputs = predictor(im)
                foreground = extract_subject_rgba(im, outputs)
                cv2.imwrite(segmented_path, foreground)
            if index % 1000 == 0:
                logging.debug('Completed %d segmentations' % (index + 1))


def extract_subject(im, output):
    """
    im: image containing the subject_mask
    output: prediction result of detectron2 for im as input
    """

    # Class IDs. It is possible the whale object is segmented into multiple regions
    class_ids = np.array(output["instances"].pred_classes.cpu())

    # Create a blank canvas to create mask
    background = np.zeros(im.shape[:2])
    subject_mask = np.zeros(im.shape[:2], dtype=bool)

    for class_index in class_ids:
        # for each region extract mask
        mask_tensor = output["instances"].pred_masks[class_index]
        region_mask = mask_tensor.cpu()

        assert subject_mask.shape == region_mask.shape  # ensure mask is correct
        subject_mask = np.logical_or(subject_mask, region_mask)

    # can be used as a binary BW mask
    bin_mask = np.where(subject_mask, 255, background).astype(np.uint8)

    # === Add a fourth channel to the original image array === #

    # Split into RGB (technically BGR in OpenCV) channels
    b, g, r = cv2.split(im.astype("uint8"))

    # Create alpha channel array of ones
    # Then multiply by 255 to get the max transparency value
    a = np.ones(subject_mask.shape, dtype="uint8") * 255

    # Rejoin with alpha channel that's always 1, or non-transparent
    rgba = [b, g, r, a]
    # Both of the lines below accomplish the same thing
    im_4ch = cv2.merge(rgba, 4)

    # === Extract pixels using mask === #

    # Create 4-channel blank background
    bg = np.zeros(im_4ch.shape)

    # Create 4-channel mask
    mask = np.stack([subject_mask, subject_mask, subject_mask, subject_mask], axis=2)

    # Copy color pixels from the original color image where mask is set
    foreground = np.where(mask, im_4ch, bg).astype(np.uint8)

    return foreground


def extract_subject_rgba(im, output):
    """
    im: image containing the subject_mask
    output: prediction result of detectron2 for im as input
    """

    # Class IDs. It is possible the whale object is segmented into multiple regions
    class_ids = np.array(output["instances"].pred_classes.cpu())

    # Create a blank canvas to create mask
    subject_mask = np.zeros(im.shape[:2], dtype=bool)

    for class_index in class_ids:
        # for each region extract mask
        mask_tensor = output["instances"].pred_masks[class_index]
        region_mask = mask_tensor.cpu()

        assert subject_mask.shape == region_mask.shape  # ensure mask is correct
        subject_mask = np.logical_or(subject_mask, region_mask)

    # Create alpha channel array of ones
    # Then multiply by 255 to get the max transparency value
    alpha = np.ones(im.shape, dtype="uint8")
    zer = np.zeros(im.shape, dtype="uint8")
    subject_mask = np.expand_dims(subject_mask, axis=2)
    alpha = np.where(subject_mask, alpha, zer).astype(np.uint8)

    bg = np.array([255, 255, 255])
    foreground = ((bg * (1 - alpha)) + (im * alpha)).astype(np.uint8)

    return foreground


def mask_subject(cfg, whale_metadata):
    """
    cfg: configuration returned from the train function
    whale_metadata: metadata variable returned from register_dataset function
    """

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_whale_dicts("dataset/test")
    for d in random.sample(dataset_dicts, 15):
        im = cv2.imread(d["file_name"])
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=whale_metadata,
                       scale=0.5,
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       instance_mode=ColorMode.SEGMENTATION
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])
