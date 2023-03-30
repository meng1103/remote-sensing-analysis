# 导入常用的库
import os
import time

import matplotlib.pyplot as plt
import paddlers as pdrs
from PIL import Image
# 导入flask库的Flask类和request对象
from flask import Flask, render_template, request
from paddle.dataset.image import cv2

app = Flask(__name__)

# 将导出模型所在目录传入Predictor的构造方法中
change_detection_predictor = pdrs.deploy.Predictor('./inference_model_pp')
terrain_classification_predictor = pdrs.deploy.Predictor('./inference_model_tc')
target_detection_predictor = pdrs.deploy.Predictor('./inference_model_td')
target_extraction_predictor = pdrs.deploy.Predictor('./inference_model_te')


def proportion(imge):
    count = 0
    sum = 0
    for i in range(len(imge)):
        for j in range(len(imge[i])):
            sum += 1
            if imge[i][j] > 0:
                count += 1
    return count / sum


# path = "./predict_image/change_detection_predict.png"
def photo_2(path):
    img = Image.open(path)
    Img = img.convert('L')
    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    threshold = 200

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    # 图片二值化
    photo = Img.point(table, '1')
    photo.save(path)


def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


# 首页
@app.route('/')
def index_page():
    return render_template('front_page.html')


# 目标提取
@app.route('/target_extraction')
def target_extraction():
    return render_template('target_extraction.html')


@app.route("/target_extraction_result", methods=['POST', 'GET'])
def target_extraction_result():
    received_file = request.files['input_image']
    print(received_file)
    imageFileName = received_file.filename
    print('imageFileName_a', imageFileName)
    if received_file:
        received_dirPath = './static/upload_target_extraction'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
    imageFilePath = os.path.join(received_dirPath, imageFileName)
    received_file.save(imageFilePath)
    print('image file saved to %s' % imageFilePath)
    test_path = (imageFilePath)
    result = target_extraction_predictor.predict(img_file=test_path)
    imge = result[0]['bbox']
    for i in range(len(imge)):
        imge[i] = int(imge[i])
    img = cv2.imread(test_path)
    box_color = (255, 0, 255)
    cv2.rectangle(img, (imge[0], imge[1]), (imge[2], imge[3]), color=box_color, thickness=2)
    # pro = proportion(result)  # 计算占比
    plt.imshow(img)
    img_path = './static/result_images/terrain_target_extraction_predict.png'
    plt.savefig(img_path)
    img_result = return_img_stream(img_path)
    return render_template('target_extraction_result.html', img_result=img_result)


# 变化检测----------------------------------------------------------------------------------------------------------------------
@app.route('/change_detection')
def change_detection():
    return render_template('change_detection.html')


# 使用pp_predict_image这个API服务时的调用函数
@app.route("/change_detection_result", methods=['POST', 'GET'])
def change_detection_result():
    startTime = time.time()
    received_file = request.files['input_image']
    b_file = request.files['input_image_b']
    print(received_file)
    print(b_file)
    imageFileName_a = received_file.filename
    imageFileName_b = b_file.filename
    print('imageFileName_a', imageFileName_a)
    print('imageFileName_b', imageFileName_b)

    if b_file:
        received_dirPath_a = './static/upload_change_detection/images_A'
        received_dirPath_b = './static/upload_change_detection/images_B'
        if not os.path.isdir(received_dirPath_a):
            os.makedirs(received_dirPath_a)
        if not os.path.isdir(received_dirPath_b):
            os.makedirs(received_dirPath_b)
        imageFilePath_a = os.path.join(received_dirPath_a, imageFileName_a)
        imageFilePath_b = os.path.join(received_dirPath_b, imageFileName_b)
        received_file.save(imageFilePath_a)
        b_file.save(imageFilePath_b)
        print('image file saved to %s' % imageFilePath_a)
        print('image file saved to %s' % imageFilePath_b)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        test_path = (imageFilePath_a, imageFilePath_b)
        img = change_detection_predictor.predict(img_file=test_path)
        result = img[0]['label_map']
        pro = proportion(result)  # 计算占比
        plt.imshow(result)
        img_path = './static/result_images/change_detection_predict.png'
        plt.savefig(img_path)
        photo_2(img_path)  # 转黑白并按照原路径保存
        img_result = return_img_stream(img_path)
        usedTime = time.time() - startTime
        print('完成对接收图片的预测，总共耗时%.2f秒' % usedTime)
        return render_template("change_detection_result.html", img_result=img_result, pro=pro)
    else:
        return 'failed'


# 目标检测---------------------------------------------------------------------------------------------------------------------------
@app.route('/target_detection')
def target_detection():
    return render_template('target_detection.html')


@app.route("/target_detection_result", methods=['POST', 'GET'])
def target_detection_result():
    received_file = request.files['input_image']
    print(received_file)
    imageFileName = received_file.filename
    print('imageFileName_a', imageFileName)
    if received_file:
        received_dirPath = './static/upload_target_detection'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
    imageFilePath = os.path.join(received_dirPath, imageFileName)
    received_file.save(imageFilePath)
    print('image file saved to %s' % imageFilePath)
    test_path = (imageFilePath)
    result = target_detection_predictor.predict(img_file=test_path)
    imge = result['label_map']
    # pro = proportion(result)  # 计算占比
    plt.imshow(imge)
    img_path = './static/result_images/terrain_target_detection_predict.png'
    plt.savefig(img_path)
    # photo_2(img_path)  # 转黑白并按照原路径保存
    img_result = return_img_stream(img_path)
    return render_template('target_detection_result.html', img_result=img_result)


# 地物分类--------------------------------------------------------------------------------------------------------------
@app.route('/terrain_classification')
def terrain_classification():
    return render_template('terrain_classification.html')


@app.route("/terrain_classification_result", methods=['POST', 'GET'])
def terrain_classification_result():
    received_file = request.files['input_image']
    print(received_file)
    imageFileName = received_file.filename
    print('imageFileName_a', imageFileName)
    if received_file:
        received_dirPath = './static/upload_terrain_classification'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
    imageFilePath = os.path.join(received_dirPath, imageFileName)
    received_file.save(imageFilePath)
    print('image file saved to %s' % imageFilePath)
    test_path = (imageFilePath)
    result = terrain_classification_predictor.predict(img_file=test_path)
    imge = result['label_map']
    # pro = proportion(result)  # 计算占比
    plt.imshow(imge)
    img_path = './static/result_images/terrain_classification_predict.png'
    plt.savefig(img_path)
    photo_2(img_path)  # 转黑白并按照原路径保存
    img_result = return_img_stream(img_path)
    return render_template('terrain_classification_result.html', img_result=img_result)


# 主函数
if __name__ == "__main__":
    app.run()
