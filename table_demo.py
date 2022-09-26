
from ydocr.predict_table import TableSystem,to_excel
import cv2
import numpy as np
from urllib.request import urlopen
import os

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
    # 1.初始化模型
    table_sys = TableSystem()

    # 2. 读取图片
    #   url 读取 
    # img = cv_imread_url('https://cdn.nlark.com/yuque/0/2022/png/22909407/1662605393255-ed70d655-8d39-4145-8632-1a5cafe0a56a.png')
    #   本地文件读取
    image_file = r"D:\yd_project\OCR\测试图片\6098829-a4381bccd545b368.png"
    img = cv_imread(image_file)
    # 3. 表格识别
    pred_res, _ = table_sys(img)
    
    pred_html = pred_res['html'] 
    print(pred_res['html'])
    # 4. 设置保存路径
    excel_dir = 'save'
    excel_filename = os.path.split('image_file')[-1]
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)
    excel_filename = os.path.join(excel_dir,excel_filename[:-4]+'.xlsx')
    # 转换成excel
    to_excel(pred_html,excel_filename)

if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()

