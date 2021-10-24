# 导入 opencv 库，并使用 namespace: cv
import cv2 as cv

def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)

# 使用 opencv 导入图片，括号中可以输入绝对路径或者相对路径。
# 在 vscode 中导入文件夹，可以帮助你快速的找到文件的相对路径
# 但如果报错的话，则最好使用图片的据对路径。
# 使用 img 变量保存导入的图片。
img = cv.imread('C:/Users/Daoming Chen/Documents/GitHub/cnblog_style/pics/1.png')
# imshow 用来展示图片，其两个参数分别为：'重新赋予的图片名称'，变量

img_resize = rescaleFrame(img)
cv.imshow('pic1', img_resize)
# 等待按键，当按下任意键后，结束程序。
cv.waitKey(0)


# capture = cv.VideoCapture('')

# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('video',frame)
#     # if d is pressed after 20s 
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()