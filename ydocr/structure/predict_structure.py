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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import json
from ydocr.utility import get_model_data, get_character_dict, get_model_data_from_path
from preprocess import preprocess_op
from postprocess import TableLabelDecode
import onnxruntime as ort

# import tools.infer.utility as utility

# from ppocr.postprocess import build_post_process

# from ppocr.utils.utility import get_image_file_list, check_and_read
#  from ppocr.utils.visual import draw_rectangle
# from ppstructure.utility import parse_args


character_dict = 'ydocr/model/table_structure_dict_ch.txt'
table_model_file = 'ydocr/model/table_model_ch.onnx'

def build_pre_process_list(args):
    resize_op = {'ResizeTableImage': {'max_len': args.table_max_len, }}
    pad_op = {
        'PaddingTableImage': {
            'size': [args.table_max_len, args.table_max_len]
        }
    }
    normalize_op = {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'mean': [0.485, 0.456, 0.406] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
    to_chw_op = {'ToCHWImage': None}
    keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
    if args.table_algorithm not in ['TableMaster']:
        pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]
    else:
        pre_process_list = [
            resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
        ]
    return pre_process_list

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

# args = parse_args()

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

# table_model = utility.create_predictor(args, 'table', logger)
class TableStructurer(object):
    def __init__(self,ort_providers=None):
        # pre_process_list = build_pre_process_list(args)
        if ort_providers is None:
            ort_providers = ['CPUExecutionProvider']

        # postprocess_params = {
        #     'name': 'TableLabelDecode',
        #     "character_dict_path": args.table_char_dict_path,
        #     'merge_no_span_structure': args.merge_no_span_structure
        # }
        
        self.preprocess_op = preprocess_op
        self.postprocess_op = TableLabelDecode(character_dict_path = character_dict,merge_no_span_structure=True)
        model_data = get_model_data(table_model_file) if table_model_file is None else get_model_data_from_path(table_model_file)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(model_data, so, providers=ort_providers)
        self.output_tensors = None
        self.predictor, self.input_tensor = sess, sess.get_inputs()[0]

    def __call__(self, img):
        starttime = time.time()
        # ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        input_dict = {}
        input_dict[self.input_tensor.name] = img
        outputs = self.predictor.run(self.output_tensors, input_dict)
        
        # self.input_tensor.copy_from_cpu(img)
        # self.predictor.run()
        # outputs = []
        # for output_tensor in self.output_tensors:
        #     output = output_tensor.copy_to_cpu()
        #     outputs.append(output)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime
        return (structure_str_list, bbox_list), elapse

def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image

def main():
    
    # image_file_list = get_image_file_list(args.image_dir)
    table_structurer = TableStructurer()
    image_file = r'img_file/table_ch_result3.jpg'
    img = cv2.imread(image_file)
    structure_res, elapse = table_structurer(img)
    print(structure_res)
    structure_str_list, bbox_list = structure_res
    # bbox_list_str = json.dumps(bbox_list.tolist())
    img = draw_boxes(img, bbox_list)
    cv2.imwrite('table_structure.png',img)
    #         img_save_path = os.path.join(args.output,
    #                                      os.path.basename(image_file))
    # count = 0
    # total_time = 0
    # # os.makedirs(args.output, exist_ok=True)
    # with open(
    #         os.path.join(args.output, 'infer.txt'), mode='w',
    #         encoding='utf-8') as f_w:
    #     for image_file in image_file_list:
    #         img, flag, _ = check_and_read(image_file)
    #         if not flag:
    #             img = cv2.imread(image_file)
    #         if img is None:
    #             logger.info("error in loading image:{}".format(image_file))
    #             continue
    #         structure_res, elapse = table_structurer(img)
    #         structure_str_list, bbox_list = structure_res
    #         bbox_list_str = json.dumps(bbox_list.tolist())
    #         logger.info("result: {}, {}".format(structure_str_list,
    #                                             bbox_list_str))
    #         f_w.write("result: {}, {}\n".format(structure_str_list,
    #                                             bbox_list_str))

    #         if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
    #             img = draw_rectangle(image_file, bbox_list)
    #         else:
    #             img = utility.draw_boxes(img, bbox_list)
    #         img_save_path = os.path.join(args.output,
    #                                      os.path.basename(image_file))
    #         cv2.imwrite(img_save_path, img)
    #         logger.info("save vis result to {}".format(img_save_path))
    #         if count > 0:
    #             total_time += elapse
    #         count += 1
    #         logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    main()
