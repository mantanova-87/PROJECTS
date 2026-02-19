import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
import time

# === Constants ===
FPS = 30
MEASURE_DURATION = 10
MEASURE_FRAMES = MEASURE_DURATION * FPS
MAX_SHIFT = 10
BRIGHTNESS_LOW = 50
BRIGHTNESS_HIGH = 200
MIN_FACE_HEIGHT = 80
MAX_FACE_HEIGHT = 180

# === Filters ===
def bandpass(signal, low=0.7, high=4.0, fs=30, order=3):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)

def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def normalize_lighting(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def apply_skin_mask(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

# === Signal Extraction ===
def extract_rppg_chrom(rgb_buffer):
    rgb = np.stack(rgb_buffer)
    r = rgb[:, :, :, 0].mean(axis=(1, 2))
    g = rgb[:, :, :, 1].mean(axis=(1, 2))
    b = rgb[:, :, :, 2].mean(axis=(1, 2))
    X = 3 * r - 2 * g
    Y = 1.5 * r + g - 1.5 * b
    S = X - Y
    S = (S - np.mean(S)) / np.std(S)
    return S

def estimate_hr(signal, fps):
    signal = bandpass(signal)
    signal = smooth_signal(signal)
    std = np.std(signal)
    if std < 0.15:
        return "No Pulse Detected", False
    peaks, _ = find_peaks(signal, distance=fps / 2.5, prominence=0.2, width=2)
    if len(peaks) >= 2:
        bpm = len(peaks) * (60 / (MEASURE_FRAMES / fps))
        return f"HR: {int(bpm)} bpm", True
    freqs = np.fft.rfftfreq(len(signal), d=1/fps)
    fft_mag = np.abs(np.fft.rfft(signal))
    peak_freq = freqs[np.argmax(fft_mag)]
    bpm = peak_freq * 60
    return f"HR (FFT): {int(bpm)} bpm", True

# === Environment Checks ===
def is_lighting_good(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return BRIGHTNESS_LOW < brightness < BRIGHTNESS_HIGH

def is_face_stable(center, prev_center):
    if prev_center is None:
        return True
    dx = abs(center[0] - prev_center[0])
    dy = abs(center[1] - prev_center[1])
    return dx < MAX_SHIFT and dy < MAX_SHIFT

def is_distance_good(face_height):
    if face_height < MIN_FACE_HEIGHT:
        return "Too far — come closer", False
    elif face_height > MAX_FACE_HEIGHT:
        return "Too close — move back", False
    else:
        return "Perfect distance — measuring...", True

# === Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

buffer = []
hr_text = ""
frame_times = []
measuring = False
hr_calculated = False
prev_center = None
env_text = "Checking environment..."

print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_times.append(time.time())
    fps_est = 1 / (frame_times[-1] - frame_times[-2]) if len(frame_times) > 2 else FPS

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        lm = face.landmark
        points = [lm[10], lm[338], lm[297], lm[332], lm[1], lm[284], lm[424], lm[356]]
        x_coords = [int(p.x * w) for p in points]
        y_coords = [int(p.y * h) for p in points]
        x1, x2 = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
        y1, y2 = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
        roi = rgb[y1:y2, x1:x2]

        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            continue

        roi_resized = cv2.resize(roi, (64, 64))
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_height = y2 - y1

        lighting_ok = is_lighting_good(roi_resized)
        stable = is_face_stable(center, prev_center)
        distance_msg, distance_ok = is_distance_good(face_height)
        prev_center = center

        if not lighting_ok:
            env_text = "Too dark — adjust lighting"
            buffer.clear()
            measuring = False
            hr_calculated = False
        elif not stable:
            env_text = "Face moving — hold still"
            buffer.clear()
            measuring = False
            hr_calculated = False
        elif not distance_ok:
            env_text = distance_msg
            buffer.clear()
            measuring = False
            hr_calculated = False
        else:
            env_text = distance_msg
            roi_normalized = normalize_lighting(roi_resized)
            roi_skin = apply_skin_mask(roi_normalized)
            buffer.append(roi_skin)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("ROI", cv2.cvtColor(roi_skin, cv2.COLOR_RGB2BGR))

            if not measuring and not hr_calculated:
                buffer.clear()
                measuring = True
                hr_text = "Measuring Heart Rate..."

            if measuring and len(buffer) >= MEASURE_FRAMES:
                signal = extract_rppg_chrom(buffer[-MEASURE_FRAMES:])
                hr_text, valid = estimate_hr(signal, FPS)
                print(f"[RESULT] {hr_text}")  # ✅ Print to terminal
                measuring = False
                hr_calculated = True
                time.sleep(5)
                break
    else:
        buffer.clear()
        measuring = False
        hr_calculated = False
        hr_text = ""
        env_text = "No face detected"

    cv2.putText(frame, env_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if hr_text:
        cv2.putText(frame, hr_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps_est:.1f}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Smart rPPG HR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()