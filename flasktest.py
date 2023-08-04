# coding=utf-8
# =============================================
# @Time      : 2022-07-20 10:04
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import os
import requests
import glob
from tqdm import tqdm
import time,json


def pic_post(images):
    postdata = {
        "input": (r'image.png', images, 'image/png')
    }
    st = time.time()
    try:
        result = requests.post('http://127.0.0.1:99884/predict', files=postdata)
        et = time.time()
        print(f"request_ok_time:{round(et - st, 2)}s")
        if result.status_code == 200:
            obj = json.loads(result.text)
            with open('./test.txt','w',encoding='utf-8') as w:
                w.write(json.dumps(obj['predictions']))
    except Exception as e:
        et = time.time()
        print(f"exception request_exception_time:{round(et - st, 2)}s")
        print(f"报错：{e}")



if __name__ == '__main__':

    file_path = './datasets/train/image/3.png'

    file_path = [file_path] if os.path.isfile(file_path) else glob(f"{file_path}/*")

    for i, file in enumerate(tqdm(file_path)):
        img = open(file, "rb")
        results = pic_post(img)