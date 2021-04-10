# coding=utf-8
"""人脸检测---dlib"""
import dlib
import cv2
import numpy as np


class Detector:
    def __init__(self, dir_predictor_landmarks):
        # 人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        # face landmark检测器
        self.predictor = dlib.shape_predictor(dir_predictor_landmarks)

    def get_rects(self, img):
        if len(img.shape) != 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        rects = self.detector(img_gray, 1)
        return rects

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for idx in range(0, 68):
            coords[idx] = (shape.part(idx).file, shape.part(idx).y)
        return coords

    def rect_to_bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        return (x, y, w, h)

    def detect_face(self, img, debug=False):
        """
        人脸检测
        :param img: 3通道图片, bgr格式
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = self.detector(gray, 1)
        if len(face_rects) == 0:
            return None

        result = []
        for (i, rect) in enumerate(face_rects):
            result_dic = {}
            shape = self.predictor(gray, rect)
            shape = self.shape_to_np(shape)
            left_eye = list(((shape[36] + shape[39]) / 2.0).astype(np.int))
            right_eye = list(((shape[42] + shape[45]) / 2.0).astype(np.int))
            nose = list(shape[30])
            left_mouth = list(shape[48])
            right_mouth = list(shape[54])
            (x, y, w, h) = self.rect_to_bb(rect)

            result_dic['bbox'] = [x, y, x + w, y + h]
            result_dic['area'] = w * h
            result_dic['left_eye'] = left_eye
            result_dic['right_eye'] = right_eye
            result_dic['nose'] = nose
            result_dic['left_mouth'] = left_mouth
            result_dic['right_mouth'] = right_mouth
            result_dic['width'] = w
            result_dic['height'] = h
            if debug:
                cv2.circle(img, tuple(left_eye), 1, (0, 0, 255), -1)
                cv2.circle(img, tuple(right_eye), 2, (0, 0, 255), -1)
                cv2.circle(img, tuple(nose), 3, (0, 0, 255), -1)
                cv2.circle(img, tuple(left_mouth), 4, (0, 0, 255), -1)
                cv2.circle(img, tuple(right_mouth), 5, (0, 0, 255), -1)
            result.append(result_dic)

        return result


if __name__ == "__main__":
    dir_predictor_landmarks = 'shape_predictor_68_face_landmarks.dat'
    image = cv2.imread('images/family.jpg')
    detector = Detector(dir_predictor_landmarks)

    results = detector.detect_face(image)
    # 检测结果显示
    for each_result in results:
        cv2.circle(image, tuple(each_result['left_eye']), 1, (0, 0, 255), -1)
        cv2.circle(image, tuple(each_result['right_eye']), 2, (0, 0, 255), -1)
        cv2.circle(image, tuple(each_result['nose']), 3, (0, 0, 255), -1)
        cv2.circle(image, tuple(each_result['left_mouth']), 4, (0, 0, 255), -1)
        cv2.circle(image, tuple(each_result['right_mouth']), 5, (0, 0, 255), -1)
    cv2.imshow('result', image)
    cv2.waitKey(0)

