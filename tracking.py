import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import scipy.io
# import predict
from pathlib import Path
from spine_tracking.metadata import utils as metadata_utils

from spine_tracking.dendritic_spine_detection.utils import CentroidTracker
from collections import OrderedDict
from typing import List, Tuple

# models/research/object_detection muss im PYTHONPATH sein

# parser = argparse.ArgumentParser(description='Track spines in the whole stack',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('-i', '--images', required=False,
#                     help='Path to input images')
# parser.add_argument('-t', '--threshold',
#                     help='Threshold for detection', default=0.5, type=float)
# parser.add_argument('-a', '--appeared',
#                     help='appeared counter', default=0, type=int)
# parser.add_argument('-d', '--disappeared',
#                     help='disappeared counter', default=3, type=int)
# parser.add_argument('-th', '--theta',
#                     help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
# parser.add_argument('-ta', '--tau',
#                     help='Threshold for tau (tracking threshold)', default=0.5, type=float)
# parser.add_argument('-m', '--model',
#                     help='Path to model you want to analyze with')
# parser.add_argument('-c', '--csv', required=False,
#                     help='Single file or folder of csv files for previous prediction.'
#                     'If this flag is set, no model prediction will be executed')
# parser.add_argument('-s', '--save-images', action='store_true',
#                     help='Activate this flag if images should be saved')
# parser.add_argument('-o', '--output', required=False,
#                     help='Path where tracking images and csv should be saved, default: output/tracking/MODEL')
# parser.add_argument('-f', '--file-save',
#                     help="Name of tracked data csv file", default="data_tracking.csv")
# parser.add_argument('-mc', '--metric', default='iom',
#                     help='Metric which should be used for evaluating. Currently available: iom, iou.'
#                     'Own metric can be implemented as lambda function which takes two arguments and returns one.')
# parser.add_argument('-uo', '--use-offsets', action='store_true',
#                     help='whether offsets should be used or not')


def draw_boxes(img: np.ndarray, objects: OrderedDict) -> np.ndarray:
    """Draw boxes onto image

    Args:
        img (np.ndarray): image input to draw on
        objects (OrderedDict): Dictionary of objects of format (cX, cY, w, h, conf)

    Returns:
        np.ndarray: output image with drawn boxes
    """
    for key in objects:
        # w, h = 512, 512
        cX, cY, width, height, conf = objects[key]
        x1, x2 = int(cX-width/2), int(cX+width/2)
        y1, y2 = int(cY-height/2), int(cY+height/2)
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)

        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

        # green filled rectangle for text
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x1+20, y1-10), color, thickness=-1)

        # text
        img = cv2.putText(img, str(key), (x1+1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    return img


def csv_to_boxes(df: pd.DataFrame) -> Tuple[List]:
    """Convert a dataframe into the relevant detection information

    Args:
        df (pd.DataFrame): Dataframe of interest

    Returns:
        Tuple[List]: Tuple containing boxes, scores, num detections 
    """
    scores = df['score'].values
    boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values

    # if 'width' in df.columns and 'height' in df.columns:
    #     w = df['width'].values[0]
    #     h = df['height'].values[0]
    # else:
    #     w = default_w
    #     h = default_h
    
    # # divide by image size
    # boxes[:, 0] /= w
    # boxes[:, 1] /= h
    # boxes[:, 2] /= w
    # boxes[:, 3] /= h

    num_detections = len(scores)
    return boxes, scores, num_detections

def main(args):
    # args = parser.parse_args()
    MAX_DIFF = args.tau
    IOM_THRESH = args.theta
    THRESH = args.threshold
    MIN_APP = args.appeared
    MAX_DIS = args.disappeared
    METRIC = args.metric
    MAX_VOL = 2000

    if args.images is None:
        raise ValueError('You need specify input images or input tif stack!')

    # save_folder: folder where tracking csv file will be saved
    # folder: name of folder which is used in csv file for generating filename-column
    if args.model is not None:
        model_name = args.model.split(
            "/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join('output/tracking', model_name)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    img_output_path = os.path.join(args.output, 'images')

    if args.file_save is None:
        input_inspired_filename = metadata_utils.remove_iterative_params_from_filename(os.path.basename(args.images)[:-4])
        csv_output_path = os.path.join(args.output,'csvs', f'data_tracking_{input_inspired_filename}.csv')
    else:
        csv_output_path = os.path.join(args.output, 'csvs', args.file_save)
    if args.save_images and not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    # to get some annotations on the first images too, make the same backwards
    all_imgs = sorted(glob.glob(args.images))

    all_dicts = []
    total_boxes = []
    total_scores = []
    nr_imgs = len(list(all_imgs))
    objects = dict()

    # if it's just a single csv file, load all data before iterating over images
    if args.csv is not None:
        all_csv_files = glob.glob(args.csv)
        if len(all_csv_files) == 0:
            raise ValueError(
                'No csv files with valid prediction data are available.')
        csv_path = args.csv

    # get all boxes and scores at the start if prediction is necessary:
    if args.csv is None:
        csv_path = args.images.replace('images', 'csvs').replace('png', 'csv')
        all_csv_files = glob.glob(csv_path)
        if len(all_csv_files) == 0:
            raise ValueError(
                'No csv files with valid prediction data are available.')

    all_csv_paths = sorted(all_csv_files)
    # all_csv_paths = list(Path().rglob(csv_path))


    ct = CentroidTracker(maxDisappeared=MAX_DIS, minAppeared=MIN_APP,
                         maxDiff=MAX_DIFF, iomThresh=IOM_THRESH, maxVol=MAX_VOL, metric=METRIC)

    # use given prediction for all images, if csv is available
    for img in all_imgs:
        orig_img = os.path.basename(img)
        if len(all_csv_paths) > 1:
            csv_path = [
                elem for elem in all_csv_paths if orig_img[:-4] == os.path.basename(elem)[:-4]]
            if len(csv_path) == 0:
                # no corresponding csv file for this image
                continue
            else:
                csv_path = csv_path[0]
            try:
                new_df = pd.read_csv(csv_path)
                boxes, scores, num_detections = csv_to_boxes(
                    new_df)
            except:
                continue
        else:
            try:
                new_df = pd.read_csv(args.csv)

                # load only data from interesting image
                new_df = new_df[new_df.apply(lambda row: os.path.splitext(
                    orig_img)[0] in row['filename'], axis=1)]  # axis=1 for looping through rows

                boxes, scores, num_detections = csv_to_boxes(
                    new_df)
            except:
                continue

        # boxes = boxes[0]

        # scores = scores[0]
        # num_detections = int(num_detections[0])

        img_path = '/' + img.split('../')[-1]
        image_np = cv2.imread(img_path)
        h, w = image_np.shape[:2]
        # Real tracking part!
        rects = np.array([[boxes[i][0], boxes[i][1],
                           boxes[i][2], boxes[i][3], scores[i]] for i in range(num_detections)
                          if scores[i] >= THRESH])

        objects = ct.update(rects)  # y1, x1, y2, x2 - format

        # Start with non-empty lists
        boxes = []
        scores = []

        # DO NOT USE absolute path for images!
        total_path = os.path.join(img_output_path, img.split('/')[-1])
        for key in objects:
            orig_dict = {'filename': total_path,
                         'width': w, 'height': h, 'class': 'spine'}

            # Making boxes, scores correct
            cX, cY, width, height, conf = objects[key]
            x1, x2 = (cX-width/2)/w, (cX+width/2)/w
            y1, y2 = (cY-height/2)/h, (cY+height/2)/h
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

            orig_dict.update({'id': key, 'ymin': round(y1*h, 2), 'ymax': round(y2*h, 2), 'xmin': round(x1*w, 2),
                              'xmax': round(x2*w, 2), 'score': conf})

            all_dicts.append(orig_dict)

        boxes = np.array(boxes)
        scores = np.array(scores)
        total_boxes.append(boxes)
        total_scores.append(scores)

        if args.save_images:
            image_np = cv2.imread(img)
            image_np = draw_boxes(image_np, objects)
            cv2.imwrite(img_output_path + '/' + img.split('/')[-1], image_np)

    # delete all double elements
    all_dicts = [dict(tup)
                 for tup in {tuple(set(elem.items())) for elem in all_dicts}]
    df = pd.DataFrame(all_dicts, columns=[
                      'id', 'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.sort_values(by='filename', inplace=True)
    df.to_csv(csv_output_path, index=False)

    # count real spines (does NOT correspond to max_key, but to number of keys!)
    nr_all_ind = len(df.groupby('id'))
    print(f"Nr of spines found: {nr_all_ind}")

    print('[INFO] Written predictions to '+csv_output_path+'.')

def main_explicit_args(matching_filename: str, files_dir: str, csv_path: str, threshold: float, appeared: int, disappeared: int, theta: float, 
                        tau: float, save_images: bool, output: str, metric: str, use_offsets: bool,
                        model: str):
    
    csv = os.path.join(csv_path, matching_filename)
    csv = csv.replace('images', 'csvs').replace('png', 'csv')

    images = os.path.join(files_dir, matching_filename)
    images = images.replace('csvs', 'images').replace('csv', 'png')

    args = argparse.Namespace(threshold=threshold, appeared=appeared, disappeared=disappeared,
                                theta=theta, tau=tau, save_images=save_images, output=output,
                                metric=metric, use_offsets=use_offsets, model=model, 
                                file_save=None, csv=csv, images=images)

    main(args)
                        
                                        