# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import copy
import logging
import numpy as np
import time
from ydocr.cls import TextClassifier
from ydocr.det import TextDetector
from ydocr.rec import TextRecognizer
# import tools.infer.utility as utility
from predict_system import sorted_boxes,get_rotate_crop_image
from ydocr.structure.predict_structure import TableStructurer
from ydocr.structure.matcher import TableMatch
import onnxruntime as ort
from ydocr.utility import get_model_data,get_model_data_from_path,get_table_character_dict



# from ppstructure.table.table_master_match import TableMasterMatcher
# from ppstructure.utility import parse_args
# import ppstructure.table.predict_structure as predict_strture




def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_

# args = parse_args()
# table_model = utility.create_predictor(args, 'table', logger)

character_dict = get_table_character_dict()
table_model_file = 'table_model_ch.onnx'

class TableSystem(object):
    def __init__(self,  box_thresh=0.5, unclip_ratio=1.6, rec_model_path=None, det_model_path=None,table_model_path=None,
                 ort_providers=None):


        self.text_detector = TextDetector(box_thresh=box_thresh, unclip_ratio=unclip_ratio,
                                          det_model_path=det_model_path, ort_providers=ort_providers)
        self.text_recognizer = TextRecognizer(rec_model_path=rec_model_path, ort_providers=ort_providers)

        self.table_structurer = TableStructurer()
        # if args.table_algorithm in ['TableMaster']:
        #     self.match = TableMasterMatcher()
        # else:
        self.match = TableMatch(filter_ocr_result=True)

        # self.benchmark = args.benchmark
        # self.predictor, self.input_tensor, self.output_tensors, self.config = table_model
        model_data = get_model_data(table_model_file) if table_model_path is None else get_model_data_from_path(table_model_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_data, so, providers=ort_providers)
        self.output_tensors = None
        self.predictor, self.input_tensor = sess, sess.get_inputs()[0]

    def __call__(self, img, return_ocr_result_in_table=True):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()

        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        if return_ocr_result_in_table:
            result['boxes'] = dt_boxes  #[x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()

        time_dict['match'] = toc - tic
        result['html'] = pred_html
        # if self.benchmark:
        #     self.autolog.times.end(stamp=True)
        # end = time.time()
        # time_dict['all'] = end - start
        # if self.benchmark:
        #     self.autolog.times.stamp()
        return result, time_dict

    def _structure(self, img):
        # if self.benchmark:
        #     self.autolog.times.start()
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse



    def _ocr(self, img):
        h, w = img.shape[:2]
        ori_im = img.copy()
        # if self.benchmark:
        #     self.autolog.times.stamp()
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        # logger.debug("dt_boxes num : {}, elapse : {}".format(
        #     len(dt_boxes), det_elapse))
            

        #  get_rotate_crop_image 用这种方式crop保证table的识别和ocr的识别结果一致 如下方式太过简单
        # img_crop_list = []
        # for i in range(len(dt_boxes)):
        #     det_box = dt_boxes[i]
        #     x0, y0, x1, y1 = expand(2, det_box, img.shape)
        #     text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
        #     img_crop_list.append(text_rect)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)

        # 格式转换为了表格后续的操作
        if dt_boxes is None:
            return None, None     
        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)

        dt_boxes = np.array(r_boxes)
        # logger.debug("rec_res num  : {}, elapse : {}".format(
        #     len(rec_res), rec_elapse))
        return dt_boxes, rec_res, det_elapse, rec_elapse

def to_excel(html_table, excel_path):
    from tablepyxl import tablepyxl
    tablepyxl.document_to_xl(html_table, excel_path)

def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


