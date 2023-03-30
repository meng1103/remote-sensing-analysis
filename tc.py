import matplotlib.pyplot as plt
import paddlers as pdrs

# 将导出模型所在目录传入Predictor的构造方法中
predictor = pdrs.deploy.Predictor('./inference_model_tc')
# img_file参数指定输入图像路径
# result = predictor.predict(img_file='test.jpg')

test_path = ('pp_test_B.png')

result = predictor.predict(img_file=test_path)
# imge = result[0]['label_map']
# # imge = result[0]['score_map']
# # print(imge.shape)
# # print(imge)
print(result)
# imge = result['label_map']
# print(imge)
# plt.imshow(imge)
# plt.show()
# plt.savefig('predictc_1.png')
