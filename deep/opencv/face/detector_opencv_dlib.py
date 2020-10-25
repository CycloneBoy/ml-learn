# coding=utf-8
import numpy as np
import dlib
import cv2
import cv2gpu


class Detector:
    def __init__(self, dir_predictor_landmarks, dir_haarcascade_xml):
        if cv2gpu.is_cuda_compatible():
            cv2gpu.init_gpu_detector(dir_haarcascade_xml)
        else:
            print('cannot use opencv gpu---init failed')
        self.predictor = dlib.shape_predictor(dir_predictor_landmarks)

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for idx in range(0, 68):
            coords[idx] = (shape.part(idx).x, shape.part(idx).y)
        return coords

    def detect_face(self, img, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_copy = gray.copy()
        gray_copy = gray_copy.astype(np.float)

        face_rects = cv2gpu.find_faces(gray_copy, gray_copy.shape[0], gray_copy.shape[1])
        if len(face_rects) == 0:
            return None

        result = []
        for (x, y, w, h) in face_rects:
            result_dic = {}
            rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
            shape = self.predictor(gray, rect)
            shape = self.shape_to_np(shape)
            left_eye = list(((shape[36] + shape[39]) / 2.0).astype(np.int))
            right_eye = list(((shape[42] + shape[45]) / 2.0).astype(np.int))
            nose = list(shape[30])
            left_mouth = list(shape[48])
            right_mouth = list(shape[54])
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
    dir_haarcascade_xml = 'haarcascade_frontalface_default_cuda.xml'
    detector = Detector(dir_predictor_landmarks, dir_haarcascade_xml)

    img = cv2.imread('images/family.jpg')
    result = detector.detect_face(img)

    max_result = max(result, key=lambda x: x['area'])

    cv2.circle(img, tuple(max_result['left_eye']), 1, (0, 0, 255), -1)
    cv2.circle(img, tuple(max_result['right_eye']), 2, (0, 0, 255), -1)
    cv2.circle(img, tuple(max_result['nose']), 3, (0, 0, 255), -1)
    cv2.circle(img, tuple(max_result['left_mouth']), 4, (0, 0, 255), -1)
    cv2.circle(img, tuple(max_result['right_mouth']), 5, (0, 0, 255), -1)

    cv2.rectangle(img, (max_result['bbox'][0], max_result['bbox'][1]), (max_result['bbox'][2], max_result['bbox'][3]),
                  (0, 0, 255), 2)
    cv2.imshow('result', img)
    cv2.waitKey(0)
