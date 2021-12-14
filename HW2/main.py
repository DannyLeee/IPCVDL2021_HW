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
        self.Q2_1.clicked.connect(self.corner_detection)
        self.Q2_2.clicked.connect(self.find_intrinsic)
        self.Q2_3.clicked.connect(self.find_extrinsic)
        self.Q2_4.clicked.connect(self.find_distortion)
        self.Q2_5.clicked.connect(self.undistorted)

        # load image
        self.img2 = []
        for idx in range(1, 16):
            self.img2 += [cv2.imread(f'Dataset/Q2_Image/{idx}.bmp')]

        # global variable
        self.corners = []  # Q1

    # Q 2.1
    def corner_detection(self):
        # col or row numbers of corners
        pattern_size = (11, 8)

        self.corners = []
        for idx in range(0, 15):
            img = self.img2[idx]
            is_found, corners = cv2.findChessboardCorners(img, pattern_size)
            cv2.drawChessboardCorners(img, pattern_size, corners, is_found)
            self.corners += [corners]

            win_name = "Q 2.1"
            cv2.namedWindow(win_name, 0)
            cv2.resizeWindow(win_name, 512, 512)
            cv2.moveWindow(win_name, self.geometry().x() + 300, self.geometry().y())
            cv2.imshow(win_name, img)
            cv2.waitKey(500)  # wait for 500 ms

        cv2.waitKey(500)
        cv2.destroyWindow(win_name)

        # ref: https://docs.opencv.org/4.5.3/dc/dbb/tutorial_py_calibration.html
        # only need "pattern" to find distortion matrix then the intrinsic matrix (camera matrix) will also appear
        # so, the object points can use "pattern" form to describe the chess board

        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = [objp for _ in range(15)]  # 3d point in real world space
        _, self.intrinsic_mtx, self.dist_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, self.corners, self.img2[0].shape[:-1], None, None)

    # Q 2.2
    def find_intrinsic(self):
        result = f"Q 2.2 intrinsic matrix:\n{self.intrinsic_mtx}"
        self.result.setText(result)

    # Q 2.3
    def find_extrinsic(self):
        idx = self.img_idx.value() - 1
        R, _ = cv2.Rodrigues(self.rvecs[idx])
        result = np.append(R, self.tvecs[idx], axis=1)

        str_result = f"Q 2.3 extrinsic matrix of image {idx + 1}\n{result}"
        self.result.setText(str_result)

    # Q 2.4
    def find_distortion(self):
        result = f"Q 2.4 distortion matrix:\n{self.dist_coeffs}"
        self.result.setText(result)

    # Q 2.5
    def undistorted(self):
        for idx in range(0, 15):
            img = self.img2[idx]
            h, w = img.shape[:2]

            img_ = cv2.undistort(img, self.intrinsic_mtx, self.dist_coeffs)

            win_name = "Q 2.5"
            cv2.namedWindow(win_name, 0)
            cv2.resizeWindow(win_name, 1024, 512)
            cv2.moveWindow(win_name, self.geometry().x() + 300, self.geometry().y())
            result = np.zeros((h, w + w, 3), dtype="uint8")
            result[0:h, 0:w] = img
            result[0:h, w:] = img_
            cv2.imshow(win_name, result)
            cv2.waitKey(500)  # wait for 500 ms

        cv2.waitKey(500)
        cv2.destroyWindow(win_name)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())