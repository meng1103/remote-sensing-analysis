import cv2
import matplotlib.pyplot as plt
import paddlers as pdrs

# 将导出模型所在目录传入Predictor的构造方法中
predictor = pdrs.deploy.Predictor('./inference_model_te')
# img_file参数指定输入图像路径
# result = predictor.predict(img_file='test.jpg')

test_path = 'playground_8.jpg'
# 目标类别
CLASS = 'playground'
INPUT_SIZE = 608

result = predictor.predict(img_file=test_path)

# print(result)
# # imge = result[0]['bbox']
# # print(imge)
# img_q = [x['bbox'] for x in result]
# sum = 0
image = result[0]['bbox']
# print(image)

for i in range(len(image)):
    image[i] = int(image[i])

img = cv2.imread(test_path)
box_color = (255, 0, 255)
cv2.rectangle(img, (image[0], image[1]), (image[2], image[3]), color=box_color, thickness=2)
plt.imshow(img)
# plt.show()
plt.savefig('predictc.png')
