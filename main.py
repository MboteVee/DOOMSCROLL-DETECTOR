import cv2
import mediapipe as mp
import time
import winsound


# ==========================================
# 1. SETTINGS
# ==========================================

LOOKING_DOWN_THRESHOLD = 20
RESET_TIME = 3
ALERT_REPEAT_INTERVAL = 20
CALIBRATION_TIME = 3

MODEL_PATH = "face_landmarker.task"


# ==========================================
# 2. INITIALIZE MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)


# ==========================================
# 3. INITIALIZE VARIABLES
# ==========================================

looking_down_start = None
looking_down_seconds = 0

neutral_start = None

last_alert_time = 0

calibration_start = time.time()
calibration_values = []
neutral_baseline = None

status = "CALIBRATING"


# ==========================================
# 4. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ==========================================
# 5. HELPER FUNCTION: FORMAT TIMER
# ==========================================

def format_time(seconds):

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02d}:{remaining_seconds:02d}"


# ==========================================
# 6. HELPER FUNCTION: SOUND ALERT
# ==========================================

def play_alert():

    try:
        winsound.MessageBeep(
            winsound.MB_ICONEXCLAMATION
        )

    except Exception:
        print("\a")


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp for video processing
        timestamp_ms = int(
            time.monotonic() * 1000
        )

        # Detect facial landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        current_time = time.time()

        # ======================================
        # 8. CHECK IF FACE IS DETECTED
        # ======================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Eye landmarks
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Nose landmark
            nose = landmarks[1]

            # Calculate eye-line position
            eye_y = (
                left_eye.y + right_eye.y
            ) / 2

            # Calculate distance between eyes
            eye_distance = abs(
                left_eye.x - right_eye.x
            )

            # Prevent division by zero
            if eye_distance > 0:

                # Nose position relative to eyes
                nose_position = (
                    nose.y - eye_y
                ) / eye_distance

                # ==================================
                # 9. CALIBRATION
                # ==================================

                if neutral_baseline is None:

                    calibration_values.append(
                        nose_position
                    )

                    elapsed_calibration = (
                        current_time - calibration_start
                    )

                    status = "CALIBRATING"

                    if elapsed_calibration >= CALIBRATION_TIME:

                        neutral_baseline = (
                            sum(calibration_values)
                            / len(calibration_values)
                        )

                        print(
                            "Calibration completed!"
                        )

                else:

                    # ==================================
                    # 10. DETECT DOWNWARD ORIENTATION
                    # ==================================

                    difference = (
                        nose_position
                        - neutral_baseline
                    )

                    looking_down = (
                        difference > 0.18
                    )

                    # ==================================
                    # 11. TIMER LOGIC
                    # ==================================

                    if looking_down:

                        neutral_start = None

                        if looking_down_start is None:

                            looking_down_start = (
                                current_time
                            )

                        looking_down_seconds = (
                            current_time
                            - looking_down_start
                        )

                    else:

                        if neutral_start is None:

                            neutral_start = (
                                current_time
                            )

                        neutral_duration = (
                            current_time
                            - neutral_start
                        )

                        # Reset after a sustained
                        # period of looking up

                        if neutral_duration >= RESET_TIME:

                            looking_down_start = None

                            looking_down_seconds = 0

                            last_alert_time = 0

                    # ==================================
                    # 12. DOOM SCROLLING DETECTION
                    # ==================================

                    if looking_down_start is not None:

                        looking_down_seconds = (
                            current_time
                            - looking_down_start
                        )

                    else:

                        looking_down_seconds = 0

                    if (
                        looking_down_seconds
                        >= LOOKING_DOWN_THRESHOLD
                    ):

                        status = "WARNING"

                        # Sound alert
                        if (
                            current_time - last_alert_time
                            >= ALERT_REPEAT_INTERVAL
                        ):

                            play_alert()

                            last_alert_time = (
                                current_time
                            )

                    else:

                        status = "NORMAL"

                    # ==================================
                    # 13. DISPLAY TIMER
                    # ==================================

                    timer_text = format_time(
                        looking_down_seconds
                    )

                    cv2.putText(
                        frame,
                        f"Looking down: {timer_text}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    # ==================================
                    # 14. DISPLAY WARNING
                    # ==================================

                    if status == "WARNING":

                        warning_text = (
                            "Looks like you're giving "
                            "much attention to your phone!"
                        )

                        cv2.putText(
                            frame,
                            warning_text,
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            "Take a break!",
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                    else:

                        cv2.putText(
                            frame,
                            "Status: NORMAL",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

        else:

            # ======================================
            # 15. FACE NOT DETECTED
            # ======================================

            status = "NO FACE DETECTED"

            cv2.putText(
                frame,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Treat absence as a break
            if neutral_start is None:

                neutral_start = current_time

            if (
                current_time - neutral_start
                >= RESET_TIME
            ):

                looking_down_start = None

                looking_down_seconds = 0

                last_alert_time = 0


        # ==========================================
        # 16. DISPLAY VIDEO
        # ==========================================

        cv2.imshow(
            "Doom Scrolling Detector",
            frame
        )


        # ==========================================
        # 17. EXIT PROGRAM
        # ==========================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# 18. CLEAN UP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Doom Scrolling Detector stopped.")




    

        
        