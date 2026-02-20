import cv2
import time
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self, video_path):
        # MediaPipe init
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Video
        self.cap = cv2.VideoCapture(video_path)

        # State vars
        self.ptime = 0
        self.frame_count = 0
        self.save_image_done = False

    def process_frame(self, frame):
        """Run pose inference on a frame"""
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_infer = time.time()
        results = self.pose.process(rgb)
        inference_time = (time.time() - start_infer) * 1000

        # Draw landmarks
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame, results, inference_time

    def draw_metrics(self, frame, inference_time):
        """Draw FPS and inference time"""
        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if self.ptime != 0 else 0
        self.ptime = ctime

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f'Infer Time: {int(inference_time)} ms', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def save_sample(self, frame, results):
        """Save first detected pose frame"""
        if results.pose_landmarks and not self.save_image_done:
            cv2.imwrite("pose_sample.jpg", frame)
            self.save_image_done = True
            print("Sample image saved as pose_sample.jpg")

    def run(self):
        print("Press 'q' to quit...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, results, infer_time = self.process_frame(frame)
            frame = self.draw_metrics(frame, infer_time)

            self.save_sample(frame, results)

            # Count frames (first 150)
            if self.frame_count < 150:
                self.frame_count += 1

            cv2.imshow("MediaPipe Pose", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    video_path = r"E:\libraries\Handtracking\WhatsApp Video 2026-02-16 at 1.58.29 PM.mp4"
    app = PoseEstimator(video_path)
    app.run()


if __name__ == "__main__":
    main()