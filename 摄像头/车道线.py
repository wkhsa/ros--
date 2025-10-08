import cv2
import numpy as np

# 读取原图
img = cv2.imread("5_1.jpg")

# 获取图像的高度和宽度
height, width = img.shape[:2]

# 将图像的上半部分裁剪掉，只保留下半部分
img = img[height//2:, :]

# 中值滤波
o = cv2.medianBlur(img, 5)

# 转换为灰度图
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)

# 使用Canny算法检测边缘
binary = cv2.Canny(gray, 50, 150)

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 15, minLineLength=100, maxLineGap=18)

# 遍历所有直线并绘制
for line in lines:
    x1, y1, x2, y2 = line[0]  # 读取直线两个端点的坐标
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
cv2.imshow("canny", binary)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()