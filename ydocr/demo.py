
from ydocr.predict_system import TextSystem,order_onrow
import cv2
import numpy as np


def cv_imread(img_path):
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),1)
    cv_img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return cv_img



def main():
    text_sys = TextSystem()
    # print(text_sys.ocr_single_line(cv2.imread('test3.png')))
    img = cv_imread(r"D:\影刀测试\OCR通用\lQDPJxaoSHdVGl5ozQGAsFWhRyEJ4V_HAxTIAuRAhwA_384_104.jpg")
    # img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    res = text_sys.detect_and_ocr(img, box_thresh=0.5, unclip_ratio=2.0)
    print(res)
    # print('\n')

    res = order_onrow(res)
    print(res)
    # draw_img = draw_ocr_box_result(img, res, 0.5, 'WeiRuanYaHei-1.ttf')
    # cv2.imshow('test', draw_img)
    # cv2.waitKey()


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()

