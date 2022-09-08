
from ydocr.predict_system import TextSystem,order_onrow
import cv2
import numpy as np
from urllib.request import urlopen


def cv_imread(img_path):
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),1)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img


def cv_imread_url(img_url):
    resp = urlopen(img_url)
    img = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img 


def main():
    text_sys = TextSystem()
    # print(text_sys.ocr_single_line(cv2.imread('test3.png')))

    # url 读取 
    img = cv_imread_url('https://cdn.nlark.com/yuque/0/2022/png/22909407/1662605393255-ed70d655-8d39-4145-8632-1a5cafe0a56a.png')
    # img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    res = text_sys.detect_and_ocr(img, box_thresh=0.5, unclip_ratio=2.0)
    print(res)
    # print('\n')

    res = order_onrow(res)
    print(res)



if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()


