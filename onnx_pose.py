import cv2
import time
import numpy as np
import onnxruntime as ort


class MoveNetPose:
    # COCO skeleton (17 keypoints)
    SKELETON = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (11, 12),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    def __init__(self, model_path, video_path):
        # Load model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        print("Model Input Shape:", self.session.get_inputs()[0].shape)
        print("Model Output Shape:", self.session.get_outputs()[0].shape)

        # Video
        self.cap = cv2.VideoCapture(video_path)

        # State
        self.ptime = 0

    def preprocess(self, frame):
        """Prepare frame for MoveNet"""
        img = cv2.resize(frame, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, img):
        """Run ONNX inference"""
        start = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        inference_time = (time.time() - start) * 1000

        output = outputs[0]

        if len(output.shape) == 4:
            keypoints = output[0][0]
        else:
            keypoints = output[0]

        return keypoints, inference_time

    def draw_pose(self, frame, keypoints):
        """Draw keypoints and skeleton"""
        h, w, _ = frame.shape
        display = frame.copy()

        points = []

        for kp in keypoints:
            y_norm, x_norm, score = kp

            x = int(x_norm * w)
            y = int(y_norm * h)

            if score > 0.2:
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                points.append((x, y))
            else:
                points.append(None)

        # Skeleton
        for pair in self.SKELETON:
            p1 = points[pair[0]]
            p2 = points[pair[1]]

            if p1 is not None and p2 is not None:
                cv2.line(display, p1, p2, (255, 0, 0), 2)

        return display

    def draw_metrics(self, frame, inference_time):
        """Draw FPS + inference time"""
        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if self.ptime != 0 else 0
        self.ptime = ctime

        cv2.putText(frame, f'FPS: {int(fps)}',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.putText(frame, f'Inference: {int(inference_time)} ms',
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            img = self.preprocess(frame)
            keypoints, infer_time = self.infer(img)

            display = self.draw_pose(frame, keypoints)
            display = self.draw_metrics(display, infer_time)

            cv2.imshow("MoveNet Video Pose", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    model_path = "Movenet.onnx"
    video_path = r"E:\libraries\Handtracking\WhatsApp Video 2026-02-16 at 1.58.29 PM.mp4"

    app = MoveNetPose(model_path, video_path)
    app.run()


if __name__ == "__main__":
    main()