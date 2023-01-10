import cv2
from google.cloud import vision
import os
import matplotlib.pyplot as plt
from PIL import Image

# please enter your google vision api key
YOUR_SERVICE = 'xxxxxx.json'

# cv2 read image
img = cv2.imread('./6.jpg')
cv2.imshow("original_image", img)
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉成灰階
# cv2.imshow("gray_image", image)
image = cv2.medianBlur(image, 7)                 # 模糊化，去除雜訊
image = cv2.Laplacian(image, -1, 1, 5)        # 偵測邊緣
cv2.imshow('laplacian', image)

ret, binary = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary_image", binary)

# 用侵蝕膨脹來增大白色範圍
# 輸入影像進行處理
# 1.初始化卷積核,根據實際任務指定大小,不一定非要3x3
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 6))  # 形成一個9x3的矩陣遮罩(捲積核)
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))   # 形成一個5x5的矩陣遮罩(捲積核)
dilation = cv2.dilate(binary, rectKernel, iterations = 3)
# tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, rectKernel)  #使用方法: cv2.morphologyEx(src,op,kernel)
cv2.imshow('dilation', dilation)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("contours 数量：", len(contours))

#創一個list存抓到的邊緣
contourslist = []

for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    # print(contours[i])
    # print('----------------------------')
#     print(f'i:{i},x:{x},y:{y},w:{w},h:{h}')
    if w > 360:
        contourslist.append((x, y, w, h))
a = sorted(contourslist, key=lambda s: s[-2])  # x為list內每個元素，依照x[1]的元素進行比較
# print(contours)
print("a=", a)

# 描出外輪廓
external_img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 255), 5)
cv2.imshow("external_img", external_img)

# 存成新圖片, 先用y确定高，再用x确定宽 ((x, y, w, h))
c= img.copy()
# cv2.imshow("ccccccc", c)
new_image = c[a[-1][1]:(a[-1][1])+(a[-1][3]), a[-1][0]:(a[-1][0]+a[-1][2])]    # 先用y确定高，再用x确定宽
cv2.imshow("cut_img", new_image)

image = cv2.resize(new_image, (300, 190))
print("after resize=", image.shape)
cv2.imwrite("external.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
# -------------------------------------

# 讀入external.jpg
YOUR_PIC = './external.jpg'
print('YOUR_PIC type=', type(YOUR_PIC))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = YOUR_SERVICE
client = vision.ImageAnnotatorClient()

# one-shot upload
with open(YOUR_PIC, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)


response = client.text_detection(image=image)

# 看vision ai text辨識結果
# print('response=', response)

# 將抓到資料存進response_list
response_list = []
for i in response.text_annotations:
    response_list.append(i)

# print(response_list)

# 過濾掉str type, 只留下數字資料存入just_int
just_int = []
for j in response_list:
    try:
        int(j.description)
        just_int.append(j)
    except ValueError as error:
        pass
# print('just_int=', just_int)


im = Image.open(YOUR_PIC)
plt.imshow(im)

# 建立一個data list來放卡號
data = []

#  卡號辨識、存取
for text in just_int:
    a = [(v.x, v.y) for v in text.bounding_poly.vertices]
    a.append(a[0])
    x, y = zip(*a)
    if 50 > max(x)-min(x) > 21 and (max(y)-min(y))> 6:
        # print('max_x:', max(x), 'min_x', min(x), 'max_y:', max(y), 'min_y:', min(y))
        # print(f'x_length = {max(x)-min(x)}')
        # print(f'y_length = {max(y)-min(y)}')
        # plt.plot(x, y, color='blue')
        # print("結果:", text.description)
        data.append(text.description)

# 印出卡號
card_num_spec = ''
card_num_spec = card_num_spec + data[0] + data[1]
print("card number:", card_num_spec)
# plt.show()
