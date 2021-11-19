from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np

import ui


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # link button click function
        self.Q1_1.clicked.connect(self.load_img)
        self.Q1_2.clicked.connect(self.color_seperation)
        self.Q1_3.clicked.connect(self.color_trans)
        self.Q1_4.clicked.connect(self.blending)
        self.Q2_1.clicked.connect(self.gaussian_blur)
        self.Q2_2.clicked.connect(self.bilateral_filter)
        self.Q2_3.clicked.connect(self.median_filter)
        self.Q4_1.clicked.connect(self.resize_img)
        self.Q4_2.clicked.connect(self.translation)
        self.Q4_3.clicked.connect(self.rotate_and_scale)
        self.Q4_4.clicked.connect(self.shearing)

        self.img_1 = cv2.imread("Dataset/Q1_Image/Sun.jpg")
        self.img_2 = cv2.imread("Dataset/Q2_Image/Lenna_whiteNoise.jpg")
        self.img_4 = cv2.imread("Dataset/Q4_Image/SQUARE-01.png")

    # Q 1.1
    def load_img(self):
        img = self.img_1
        cv2.imshow("Q 1.1", img)
        print(f"Height: {img.shape[0]}\nWidth: {img.shape[1]}")

    # Q 1.2
    def color_seperation(self):
        img_B = self.img_1.copy()
        img_B[:, :, 1] = 0
        img_B[:, :, 2] = 0

        img_G = self.img_1.copy()
        img_G[:, :, 0] = 0
        img_G[:, :, 2] = 0

        img_R = self.img_1.copy()
        img_R[:, :, 1] = 0
        img_R[:, :, 0] = 0

        cv2.imshow("Q 1.2_B", img_B)
        cv2.moveWindow("Q 1.2_B", self.geometry().x() + 300, self.geometry().y())
        cv2.imshow("Q 1.2_G", img_G)
        cv2.moveWindow("Q 1.2_G", self.geometry().x() + 300 + img_G.shape[1], self.geometry().y())
        cv2.imshow("Q 1.2_R", img_R)
        cv2.moveWindow("Q 1.2_R", self.geometry().x() + 300 + + img_G.shape[1] * 2, self.geometry().y())

    # Q 1.3
    def color_trans(self):
        img_1 = self.img_1.copy()
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Q 1.3_1", img_1)
        cv2.moveWindow("Q 1.3_1", self.geometry().x() + 300, self.geometry().y())
        img_2 = self.img_1.copy()
        gray = img_2.mean(axis=2)
        # img_2 = np.stack([gray for _ in range(3)], axis=2)
        img_2[:, :, 0] = gray
        img_2[:, :, 1] = gray
        img_2[:, :, 2] = gray

        cv2.imshow("Q 1.3_2", img_2)
        cv2.moveWindow("Q 1.3_2", self.geometry().x() + 300 + img_2.shape[1], self.geometry().y())

    # Q 1.4
    def blending(self):
        img_1 = cv2.imread("Dataset/Q1_Image/Dog_Strong.jpg")
        img_2 = cv2.imread("Dataset/Q1_Image/Dog_Weak.jpg")

        win_name = "Q 1.4"
        cv2.namedWindow(win_name, 0)
        cv2.imshow(win_name, cv2.addWeighted(img_1, 1, img_2, 0, 0))    # initial img

        def callback(v):
            beta = v / 255
            alpha = 1 - beta

            img = cv2.addWeighted(img_1, alpha, img_2, beta, 0)
            cv2.imshow(win_name, img)

        cv2.createTrackbar("blend", win_name, 0, 255, callback)

    def blur(self, filter):
        if filter != 3:
            img = self.img_2.copy()
        else:
            img = cv2.imread("Dataset/Q2_Image/Lenna_pepperSalt.jpg")

        win_name = f"Q 2.{filter}_original"
        cv2.imshow(win_name, img)
        cv2.moveWindow(win_name, self.geometry().x() + 300, self.geometry().y())

        if filter == 1:
            img = cv2.GaussianBlur(img, [5, 5], 273)
        elif filter == 2:
            img = cv2.bilateralFilter(img, 9, 90, 90)
        elif filter == 3:
            img_ = cv2.medianBlur(img, 5)
            img = cv2.medianBlur(img, 3)
            win_name = "Q 2.3_result_2"
            cv2.imshow(win_name, img_)
            cv2.moveWindow(win_name, self.geometry().x() + 300 + img.shape[1] * 2, self.geometry().y())

        win_name = f"Q 2.{filter}_result"
        cv2.imshow(win_name, img)
        cv2.moveWindow(win_name, self.geometry().x() + 300 + img.shape[1], self.geometry().y())

    # Q 2.1
    def gaussian_blur(self):
        self.blur(1)

    # Q 2.2
    def bilateral_filter(self):
        self.blur(2)

    # Q 2.3
    def median_filter(self):
        self.blur(3)

    # Q 4.1
    def resize_img(self):
        img = self.img_4
        img = cv2.resize(img, [256, 256])
        cv2.imshow("Q 4.1", img)
        self.img_4 = img

    # Q 4.2
    def translation(self):
        img = self.img_4

        trans_x = 0
        trans_y = 60
        trans_mat = np.array([[1.0, 0, trans_x], [0.0, 1, trans_y]])
        img = cv2.warpAffine(img, trans_mat, (400, 300))

        win_name = "Q 4.2"
        cv2.imshow(win_name, img)

        self.img_4 = img

    # Q 4.3
    def rotate_and_scale(self):
        img = self.img_4
        center = (128, 188)
        angle = 10
        scale = 0.5
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, rot_mat, (400, 300))
        cv2.imshow("Q 4.3", img)

        self.img_4 = img

    # Q 4.4
    def shearing(self):
        img = self.img_4

        srcTri = np.array([[50.0, 50], [200, 50], [50, 200]], dtype=np.float32)
        dstTri = np.array([[10.0, 100], [200, 50], [100, 250]], dtype=np.float32)

        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        img = cv2.warpAffine(img, warp_mat, (400, 300))
        cv2.imshow("Q 4.4", img)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())