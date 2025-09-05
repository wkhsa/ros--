import cv2                      #图像处理
from pyzbar.pyzbar import decode#识别二维码和条形码
import webbrowser               #打开网址
img = cv2.imread("b51f8882e9ccd31b7521c1b7f7fbc67.jpg")#设置二维码图片
cap = cv2.VideoCapture(0)          #打开摄像头，因为我只有一个摄像头，所以是0

data=['link']#列表无法为空的，所以加了个link'，存储识别数据

while True:     #循环
    success,img =cap.read()#img是看摄像头图片每帧是否加载到，success是看内存是否加载到
    QR_code = decode(img)       #降识别的信息转换，因为一开始识别是以字节的形式，要转成字符串
    #print(QR_code)

    for QR in QR_code:
        QR_data=QR.data.decode('utf-8')

        if QR_data != data[-1]:
            data.append(QR_data)
            webbrowser.open(QR_data)
            print(QR_data)


        point = QR.rect
        #print(point)
        #图矩形框与添加文字
        cv2.rectangle(img,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(200,2,200),5)#框的位置大小，颜色和厚度

        cv2.putText(img,QR_data,(point[0],point[1]-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)#将识别的内容显示出来
    cv2.imshow("QR",img)#此处不要缩进，不然识别
    if  cv2.waitKey(1) & 0xFF == 27:
            break


