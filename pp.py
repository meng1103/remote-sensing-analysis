import matplotlib.pyplot as plt
import paddlers as pdrs

# 将导出模型所在目录传入Predictor的构造方法中
from PIL import Image

predictor = pdrs.deploy.Predictor('./inference_model_pp')
# img_file参数指定输入图像路径
a_path = './data/A'
b_path = './data/B'
test_path = ('./data/A/test_101.png',
             './data/B/test_101.png')
result = predictor.predict(img_file=test_path)
imge = result[0]['label_map']
# image = Image.fromarray(imge)
# plt.imshow(imge)
print(imge)

# plt.savefig('./predict1.jpg')
# plt.show()
def proportion(imge):
    count = 0
    sum = 0
    for i in range(len(imge)):
        for j in range(len(imge[i])):
            sum += 1
            if imge[i][j] > 0:
                count += 1
    return count/sum

pro = proportion(imge)
print(pro)