# original implementation around qt5: https://qiita.com/odaman68000/items/c8c4093c784bff43d319

import os
import sys
from time import sleep, time
import multiprocessing
import queue
from threading import Thread
import traceback

import numpy
import cv2
import onnxruntime
from  PyQt5 import QtCore, QtGui, QtWidgets
import label

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480


def main():
    model_file = sys.argv[1]

    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()

    proc_args = (model_file, in_q, out_q)
    window_proc = multiprocessing.Process(target=run_app, args=proc_args)
    window_proc.start()

    try:
        detect(*proc_args)
    except:
        wintow_proc.to_continue = False
        in_q.put(None)
        out_q.put(None)
    window_proc.join()


def run_app(model_file, in_q, out_q):
    app = QtWidgets.QApplication(sys.argv)
    window = ImageWidget(model_file, in_q, out_q)
    window.show()

    try:
        app.exec_()
    except Exception as e:
        print(e)
    finally:
        app.closeAllWindows()


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, model_file, in_q, out_q):
        super().__init__()
        if model_file.endswith('pascal_voc3.onnx'):
            self.drawer = label.DrawLabel('pascal_voc3', CAPTURE_WIDTH, CAPTURE_HEIGHT, 7, 7)
        elif model_file.endswith('urban_objects.onnx'):
            self.drawer = label.DrawLabel('urban_objects', CAPTURE_WIDTH, CAPTURE_HEIGHT, 7, 7, 0.3)
        self.to_continue = True
        self.image = None
        self.in_q = in_q
        self.out_q = out_q
        self.image_q = queue.Queue()
        self.capture_th = Thread(target=self.capture)
        self.capture_th.start()
        self.draw_th = Thread(target=self.draw)
        self.draw_th.start()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.image is None:
            painter.setPen(QtCore.Qt.black)
            painter.setBrush(QtCore.Qt.black)
            painter.drawRect(0, 0, self.width(), self.height())
            return
        pixmap = self.create_QPixmap(self.image)
        painter.drawPixmap(0, 0, self.image.shape[1], self.image.shape[0], pixmap)

    def set_image(self, image):
        self.image = image
        self.update()

    def create_QPixmap(self, image):
        qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 4, QtGui.QImage.Format_ARGB32_Premultiplied)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def draw(self):
        cnt = 0
        t = time()
        start_time = t

        while self.to_continue:
            pred = self.out_q.get(timeout=10)
            image = self.image_q.get(timeout=10)
            if pred is None:
                break

            self.drawer.draw_rect(image, pred)
            self.set_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGRA))
            cnt += 1
            if cnt % 30 == 0:
                n = time()
                print('{} FPS'.format(30 / (n - t)))
                t = n

        print('average {} FPS'.format(cnt / (time() - start_time)))

    def closeEvent(self, event):
        event.accept()
        self.to_continue = False
        self.capture_th.join()
        self.draw_th.join()


    def capture(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        buffer_idx = 0
        while self.to_continue:
            _, orig_img = camera.read()
            if self.in_q.qsize() > 0:
                continue
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(orig_img, (224, 224))
            x = img.transpose(2, 0, 1).reshape(1, 3, 224, 224).astype(numpy.float32)
            self.in_q.put(x)
            self.image_q.put(orig_img)

        print('release camera')
        camera.release()
        self.in_q.put(None)


def detect(model_file, in_q, out_q):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../detector/trained', model_file))
    detector = onnxruntime.InferenceSession(model_path, onnxruntime.SessionOptions())
    input_name = detector.get_inputs()[0].name
    output_name = detector.get_outputs()[0].name

    while True:
        x = in_q.get(timeout=10)
        if x is None:
            out_q.put(None)
            break

        pred = detector.run([output_name], {input_name: x})[0][0]
        out_q.put(pred)


if __name__ == '__main__':
    main()
