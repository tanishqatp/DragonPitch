import threading
import time
from collections import deque
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

MODEL_PATH = "hrnet_pose.tflite"
DELEGATE_PATH = "libQnnTFLiteDelegate.so"

def load_interpreter(model_path, use_npu=True):
    if use_npu:
        try:
            delegate = tf.lite.experimental.load_delegate(
                DELEGATE_PATH, options={"backend_type": "htp"}
            )
            interp = tf.lite.Interpreter(
                model_path=model_path, experimental_delegates=[delegate]
            )
            interp.allocate_tensors()
            print("Loaded QNN HTP (NPU) delegate.")
            return interp, "npu"
        except (ValueError, OSError) as e:
            print(f"NPU failed ({e}), falling back to CPU.")
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, "cpu"

interpreter_npu, backend = load_interpreter(MODEL_PATH, use_npu=True)
input_details  = interpreter_npu.get_input_details()
output_details = interpreter_npu.get_output_details()
scale, zero_point = output_details[0]["quantization"]
IN_H, IN_W = input_details[0]["shape"][1], input_details[0]["shape"][2]


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
tracked_joints = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

PREROLL_SECONDS          = 1.0
POST_TRIGGER_SECONDS     = 3.0
RESULT_DISPLAY_SECONDS   = 5.0
HIP_CONF_THRESHOLD       = 0.4
VELOCITY_BASELINE_SAMPLES = 15
VELOCITY_MULTIPLIER      = 4.0
MIN_TRIGGER_VELOCITY     = 20.0

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

def get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h):
    j = COCO_KEYPOINT_NAMES.index(name)
    jh = heatmaps[:, :, j].astype(np.float64)
    y, x = np.unravel_index(np.argmax(jh), jh.shape)
    conf = (float(jh[y, x]) - zero_point) * scale

    # Sub-pixel refinement via parabolic interpolation around the peak bin.
    # Raw argmax only has heatmap-grid precision (can be tens of pixels on a
    # high-res source frame), which was causing large spurious frame-to-frame
    # jitter in downstream angle calculations even when confidence was high.
    def _refine(center, lo, hi):
        denom = lo - 2 * center + hi
        if abs(denom) < 1e-6:
            return 0.0
        return 0.5 * (lo - hi) / denom

    dx = dy = 0.0
    if 0 < x < jh.shape[1] - 1:
        dx = _refine(jh[y, x], jh[y, x - 1], jh[y, x + 1])
    if 0 < y < jh.shape[0] - 1:
        dy = _refine(jh[y, x], jh[y - 1, x], jh[y + 1, x])

    px = (x + dx) * img_w / hm_w
    py = (y + dy) * img_h / hm_h
    return (px, py), conf

def get_joint_xy_conf_filtered(name, heatmaps, img_w, img_h, hm_w, hm_h, prev_xy=None, dt=None,
                                 max_speed_px_per_sec=4000):
    """Same as get_joint_xy_conf, but rejects a detection that implies an
    impossible jump from the previous frame (fast motion blur can make the
    model confidently lock onto background instead of the joint). Falls back
    to the raw detection with confidence zeroed out, so downstream code
    treats it as low-confidence rather than trusting a bad position."""
    xy, conf = get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h)
    if prev_xy is not None and dt is not None and dt > 0:
        speed = np.linalg.norm(np.array(xy) - np.array(prev_xy)) / dt
        if speed > max_speed_px_per_sec:
            return xy, 0.0  # flag as untrustworthy; position kept for reference only
    return xy, conf

def joint_angle(a, b, c):
    """Angle at point b, formed by points a-b-c, in degrees."""
    a, b, c = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_pose_signals(video_path):
    """Runs the full clip through the model once, returns per-frame joint positions + confidences."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracked_joints = [
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    data = {"timestamps": []}
    for name in tracked_joints:
        data[f"{name}_xy"] = []
        data[f"{name}_conf"] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IN_W, IN_H))
        model_input = np.expand_dims(resized, axis=0).astype(np.uint8)

        interpreter_npu.set_tensor(input_details[0]["index"], model_input)
        interpreter_npu.invoke()
        heatmaps = interpreter_npu.get_tensor(output_details[0]["index"])[0]

        hm_h, hm_w, _ = heatmaps.shape
        img_h, img_w = frame.shape[0], frame.shape[1]

        for name in tracked_joints:
            xy, conf = get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h)
            data[f"{name}_xy"].append(xy)
            data[f"{name}_conf"].append(conf)

        data["timestamps"].append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    for key in data:
        data[key] = np.array(data[key])
    data["fps"] = fps
    return data

def smooth(signal, window=3):
    if len(signal) < window:
        return signal
    return np.convolve(signal, np.ones(window)/window, mode="same")


def compute_onset_by_velocity(values, timestamps, confidences, confidence_threshold=0.4,
                                velocity_threshold_multiplier=3.0, smoothing_window=3):
    valid = confidences >= confidence_threshold
    if valid.sum() < 5:
        return None

    t = timestamps[valid]
    v = smooth(values[valid], smoothing_window)
    velocity = np.abs(np.diff(v)) / np.diff(t)
    v_t = t[1:]

    n_ref = max(3, int(0.15 * len(velocity)))
    resting_mean = velocity[:n_ref].mean()
    resting_std = velocity[:n_ref].std() + 1e-6
    threshold = resting_mean + velocity_threshold_multiplier * resting_std

    for ti, vi in zip(v_t[n_ref:], velocity[n_ref:]):
        if vi > threshold:
            return ti
    return None

'''def compute_onset_by_angle_change(angles, timestamps, confidences, angle_change_threshold=15.0,
                                    confidence_threshold=0.5, reference_frames=5):
    valid = confidences >= confidence_threshold
    if valid.sum() < reference_frames + 1:
        return None

    t = timestamps[valid]
    a = angles[valid]

    reference_angle = np.median(a[:reference_frames])  # "resting" position, from a few real early frames

    for ti, ai in zip(t, a):
        if abs(ai - reference_angle) > angle_change_threshold:
            return ti
    return None'''


def compute_settle_time_after(values, timestamps, confidences, after_time,
                                confidence_threshold=0.4, smoothing_window=3):
    valid = (confidences >= confidence_threshold) & (timestamps >= after_time)
    if valid.sum() < 5:
        return None

    t = timestamps[valid]
    v = smooth(values[valid], smoothing_window)
    velocity = np.abs(np.diff(v)) / np.diff(t)
    v_t = t[1:]

    quiet_threshold = np.percentile(velocity, 20)
    peak_idx = np.argmax(velocity)
    for i in range(peak_idx, len(velocity)):
        if velocity[i] <= quiet_threshold * 1.5:
            return v_t[i]
    return None


def body_scale(data, confidence_threshold=0.4):
    conf_mask = (data["left_shoulder_conf"] >= confidence_threshold) & \
                (data["right_shoulder_conf"] >= confidence_threshold)
    if conf_mask.sum() < 3:
        return None
    widths = np.linalg.norm(data["left_shoulder_xy"][conf_mask] - data["right_shoulder_xy"][conf_mask], axis=1)
    return np.median(widths)

def sequencing_score(data):
    t = data["timestamps"]

    hip_angle = np.array([
        joint_angle(data["left_hip_xy"][i], data["right_hip_xy"][i], data["right_knee_xy"][i])
        for i in range(len(t))
    ])
    hip_conf = np.minimum(np.minimum(data["left_hip_conf"], data["right_hip_conf"]), data["right_knee_conf"])

    shoulder_angle = np.array([
        joint_angle(data["left_shoulder_xy"][i], data["right_shoulder_xy"][i], data["right_elbow_xy"][i])
        for i in range(len(t))
    ])
    shoulder_conf = np.minimum(np.minimum(data["left_shoulder_conf"], data["right_shoulder_conf"]), data["right_elbow_conf"])

    hip_onset = compute_onset_by_velocity(hip_angle, t, hip_conf)
    shoulder_onset = compute_onset_by_velocity(shoulder_angle, t, shoulder_conf)

    if hip_onset is None or shoulder_onset is None:
        return {"metric": "sequencing", "message": "Not enough confident data to score.", "verdict": None}

    gap = shoulder_onset - hip_onset
    verdict = gap > 0.02
    return {
        "metric": "sequencing",
        "hip_onset": hip_onset,
        "shoulder_onset": shoulder_onset,
        "gap_seconds": gap,
        "verdict": verdict,
        "message": "Great sequencing \u2014 hips rotating before shoulders." if verdict
                   else "Throw is becoming arm-dominant \u2014 shoulders rotating too early relative to hips.",
    }

def get_frame_joint_data(heatmaps, img_w, img_h, hm_w, hm_h):
    """All tracked joints for one frame, in the same shape the metric functions expect."""
    out = {}
    for name in tracked_joints:
        xy, conf = get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h)
        out[f"{name}_xy"] = xy
        out[f"{name}_conf"] = conf
    return out

def buffer_to_data_dict(frames, fps_estimate):
    """Converts a list of per-frame dicts (from the live loop) into the same
    `data` structure extract_pose_signals() produces, so the existing metric
    functions work unchanged."""
    data = {"timestamps": np.array([f["t"] for f in frames])}
    for name in tracked_joints:
        data[f"{name}_xy"] = np.array([f[f"{name}_xy"] for f in frames])
        data[f"{name}_conf"] = np.array([f[f"{name}_conf"] for f in frames])
    data["fps"] = fps_estimate
    return data

# ──────────────────────────────────────────────
# Shared Session State
# ──────────────────────────────────────────────
session_lock   = threading.Lock()
session_active = threading.Event()   # .set() = running  |  .clear() = stopped
camera_thread  = None
session_state  = {
    "running": False,
    "pitch_count": 0,
}

# ──────────────────────────────────────────────
# Camera Loop (background thread)
# ──────────────────────────────────────────────
def camera_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        socketio.emit("session_error", {"message": "Camera not available"})
        return

    state         = "watching"
    preroll       = deque()
    capture_buffer       = []
    capture_trigger_time = None
    result_shown_until   = 0.0
    pitch_count   = 0
    pitch_history = []
    velocity_history = deque(maxlen=VELOCITY_BASELINE_SAMPLES)
    prev_hip_angle = None
    prev_t         = None
    t0 = time.time()

    print("[EVK] Camera loop started.")
    socketio.emit("session_started", {"message": "Session started"})

    try:
        while session_active.is_set():          # ← stops when watch hits /stop
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("DragonPitch - Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                session_active.clear()

            now = time.time() - t0
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IN_W, IN_H))
            model_input = np.expand_dims(resized, axis=0).astype(np.uint8)

            interpreter_npu.set_tensor(input_details[0]["index"], model_input)
            interpreter_npu.invoke()
            heatmaps = interpreter_npu.get_tensor(output_details[0]["index"])[0]
            hm_h, hm_w, _ = heatmaps.shape
            img_h, img_w   = frame.shape[0], frame.shape[1]

            joint_data   = get_frame_joint_data(heatmaps, img_w, img_h, hm_w, hm_h)
            frame_record = {"t": now, **joint_data}

            hip_angle = joint_angle(
                joint_data["left_hip_xy"],
                joint_data["right_hip_xy"],
                joint_data["right_knee_xy"]
            )
            hip_conf = min(
                joint_data["left_hip_conf"],
                joint_data["right_hip_conf"],
                joint_data["right_knee_conf"]
            )

            # ── State Machine (your existing logic, unchanged) ──
            if state == "watching":
                preroll.append(frame_record)
                while preroll and now - preroll[0]["t"] > PREROLL_SECONDS:
                    preroll.popleft()

                if hip_conf >= HIP_CONF_THRESHOLD and prev_hip_angle is not None:
                    velocity = abs(hip_angle - prev_hip_angle) / max(now - prev_t, 1e-6)
                    if len(velocity_history) >= 8:
                        baseline_mean = np.mean(velocity_history)
                        baseline_std  = np.std(velocity_history) + 1e-6
                        threshold     = baseline_mean + VELOCITY_MULTIPLIER * baseline_std
                        if velocity > threshold and velocity > MIN_TRIGGER_VELOCITY:
                            state = "capturing"
                            capture_buffer       = list(preroll)
                            capture_trigger_time = now
                            print(f"[{now:.2f}s] Pitch detected, capturing...")
                    velocity_history.append(velocity)

                if hip_conf >= HIP_CONF_THRESHOLD:
                    prev_hip_angle = hip_angle
                    prev_t = now

            elif state == "capturing":
                capture_buffer.append(frame_record)
                if now - capture_trigger_time >= POST_TRIGGER_SECONDS:
                    span         = capture_buffer[-1]["t"] - capture_buffer[0]["t"]
                    fps_estimate = len(capture_buffer) / span if span > 0 else 24.0
                    clip_data    = buffer_to_data_dict(capture_buffer, fps_estimate)
                    seq          = sequencing_score(clip_data)

                    pitch_count += 1

                    # Update shared state + emit to any connected clients (Streamlit later)
                    with session_lock:
                        session_state["pitch_count"] = pitch_count

                    socketio.emit("pitch_detected", {
                        "pitch_number": pitch_count,
                        "sequencing":   seq,
                    })
                    print(f"=== PITCH #{pitch_count} | SEQ: {seq['message']} ===")

                    state = "showing_results"
                    result_shown_until = now + RESULT_DISPLAY_SECONDS
                    velocity_history.clear()
                    prev_hip_angle = None
                    prev_t = None
                    preroll.clear()
                    capture_buffer = []

            elif state == "showing_results":
                if now > result_shown_until:
                    state = "watching"

    finally:
        cap.release()
        print(f"[EVK] Session ended. Total pitches: {pitch_count}")
        socketio.emit("session_stopped", {
            "message":     "Session stopped",
            "pitch_count": pitch_count,
        })
        with session_lock:
            session_state["running"]     = False
            session_state["pitch_count"] = pitch_count

# ──────────────────────────────────────────────
# Flask Routes  (Watch calls these)
# ──────────────────────────────────────────────
@app.route("/start", methods=["POST"])
def start_session():
    global camera_thread
    with session_lock:
        if session_state["running"]:
            return jsonify({"status": "already_running"}), 200

        session_state["running"]     = True
        session_state["pitch_count"] = 0

    session_active.set()
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    return jsonify({"status": "started"}), 200


@app.route("/stop", methods=["POST"])
def stop_session():
    with session_lock:
        if not session_state["running"]:
            return jsonify({"status": "not_running"}), 200

    session_active.clear()         # signals the camera loop to exit
    return jsonify({
        "status":      "stopped",
        "pitch_count": session_state["pitch_count"],
    }), 200


@app.route("/status", methods=["GET"])
def get_status():
    with session_lock:
        return jsonify(session_state), 200


# ──────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[EVK] DragonPitch server starting on port 5000 (backend: {backend})")
    socketio.run(app, host="0.0.0.0", port=5000)