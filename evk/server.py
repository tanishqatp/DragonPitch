"""
STAGE 2 — SESSION LOGIC
========================
Wraps the validated Stage 1 detection logic (camera -> pose -> trigger ->
sequencing -> skeleton overlay) in a proper session lifecycle:

    start_session -> monitor continuously -> stop_session -> summary

Still standalone: no watch app, no Streamlit, no server, no networking.
Start/stop are triggered by typing commands in the terminal — a stand-in
for whatever will eventually trigger them (the watch, in Stage 4+), just
enough to prove the lifecycle mechanics work on their own first.

Run:
    python session_stage2.py

Two ways to control it — use whichever window has focus:

  In the video window (click it first so it has keyboard focus):
    's'  - start a new session
    'x'  - stop the current session (prints summary)
    'q'  - quit the program entirely

  In the terminal:
    start    - begin a new session (opens the camera, starts monitoring)
    stop     - end the current session (closes the camera, prints summary)
    summary  - reprint the most recently completed session's summary
    quit     - exit the program entirely
"""

import os
import time
import threading
import queue
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory

# ── Model / delegate ────────────────────────────────────────────────────────────
MODEL_PATH = "hrnet_pose.tflite"
# Bare filename on purpose — a path containing any slash (including "./")
# makes dlopen skip its normal search path and look ONLY at that literal
# location, which is what broke NPU loading previously.
DELEGATE_PATH = "libQnnTFLiteDelegate.so"
SKEL_OUT_DIR = "./skeleton_output_stage2"
os.makedirs(SKEL_OUT_DIR, exist_ok=True)


def load_interpreter(model_path, use_npu=True):
    if use_npu:
        try:
            delegate = tf.lite.experimental.load_delegate(DELEGATE_PATH, options={"backend_type": "htp"})
            interp = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
            interp.allocate_tensors()
            print("Loaded QNN HTP (NPU) delegate.")
            return interp, "npu"
        except (ValueError, OSError) as e:
            print(f"Could not load NPU delegate ({e}). Falling back to CPU.")

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, "cpu"


interpreter, backend = load_interpreter(MODEL_PATH, use_npu=True)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scale, zero_point = output_details[0]["quantization"]
IN_H, IN_W = input_details[0]["shape"][1], input_details[0]["shape"][2]

print(f"Active backend: {backend}")
print(f"Input size (H,W): ({IN_H}, {IN_W})")
print(f"Output quantization: scale={scale}, zero_point={zero_point}")


# ── Keypoints ────────────────────────────────────────────────────────────────────
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

TRACKED_JOINTS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]

LIVE_SKELETON_EDGES = [
    ("left_shoulder", "left_elbow"),  ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),        ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),      ("right_knee", "right_ankle"),
]


def get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h):
    j = COCO_KEYPOINT_NAMES.index(name)
    jh = heatmaps[:, :, j].astype(np.float64)
    y, x = np.unravel_index(np.argmax(jh), jh.shape)
    conf = (float(jh[y, x]) - zero_point) * scale

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


def joint_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def smooth(signal, window=3):
    if len(signal) < window:
        return signal
    return np.convolve(signal, np.ones(window) / window, mode="same")


def compute_onset_by_velocity(values, timestamps, confidences,
                               confidence_threshold=0.4,
                               velocity_threshold_multiplier=3.0,
                               smoothing_window=3):
    """Never returns None — always resolves to a real onset time, via a
    three-tier fallback:
      1. normal adaptive-threshold crossing (original, reference-matching
         behavior — unchanged for any clip with clean data)
      2. if nothing crosses threshold, use the timestamp of peak velocity
         in the clip instead
      3. if confidence data is too sparse even for that, progressively
         relax the confidence requirement and retry before falling back
         to the very first timestamp as a last resort

    Both the threshold search and the peak-velocity fallback exclude a
    small margin at the tail, where np.convolve's "same"-mode boundary
    distortion lives — without this, a flat/no-motion clip's smoothing
    artifact can spike well above the resting-noise threshold and get
    mistaken for a genuine onset by tier 1 itself, not just the fallback.

    NOTE: because this always returns a real timestamp now instead of
    None, sequencing_score() below always produces a definitive
    True/False verdict — there is no more "not enough confident data to
    score" outcome. That's a deliberate trade-off: a verdict is now always
    shown even when the underlying tracking was too poor to support a
    reliable one. See the caller for how this trade-off was chosen.
    """
    def _try(conf_thresh):
        valid = confidences >= conf_thresh
        if valid.sum() < 5:
            return None, None, None
        t = timestamps[valid]
        v = smooth(values[valid], smoothing_window)
        velocity = np.abs(np.diff(v)) / np.diff(t)
        v_t = t[1:]
        return t, v_t, velocity

    t, v_t, velocity = _try(confidence_threshold)
    if velocity is None:
        for relaxed in (0.25, 0.1, 0.0):
            t, v_t, velocity = _try(relaxed)
            if velocity is not None:
                break

    if velocity is None or len(velocity) == 0:
        return float(timestamps[0]) if len(timestamps) else 0.0

    n_ref = max(3, int(0.15 * len(velocity)))
    if n_ref >= len(velocity):
        peak_idx = int(np.argmax(velocity))
        return float(v_t[peak_idx])

    tail_margin = max(1, smoothing_window // 2 + 1)
    search_end = len(velocity) - tail_margin if len(velocity) > n_ref + tail_margin else len(velocity)

    resting_mean = velocity[:n_ref].mean()
    resting_std  = velocity[:n_ref].std() + 1e-6
    threshold    = resting_mean + velocity_threshold_multiplier * resting_std

    for ti, vi in zip(v_t[n_ref:search_end], velocity[n_ref:search_end]):
        if vi > threshold:
            return float(ti)

    search_region = velocity[n_ref:search_end]
    if len(search_region) == 0:
        search_region = velocity[n_ref:]
        v_t_region = v_t[n_ref:]
    else:
        v_t_region = v_t[n_ref:search_end]
    peak_idx = int(np.argmax(search_region))
    return float(v_t_region[peak_idx])


def get_frame_joint_data(heatmaps, img_w, img_h, hm_w, hm_h):
    out = {}
    for name in TRACKED_JOINTS:
        xy, conf = get_joint_xy_conf(name, heatmaps, img_w, img_h, hm_w, hm_h)
        out[f"{name}_xy"] = xy
        out[f"{name}_conf"] = conf
    return out


def buffer_to_data_dict(frames, fps_estimate):
    data = {"timestamps": np.array([f["t"] for f in frames])}
    for name in TRACKED_JOINTS:
        data[f"{name}_xy"]   = np.array([f[f"{name}_xy"]   for f in frames])
        data[f"{name}_conf"] = np.array([f[f"{name}_conf"] for f in frames])
    data["fps"] = fps_estimate
    return data


def sequencing_score(data):
    """Identical to test_vdo.ipynb's sequencing_score — both hip_onset and
    shoulder_onset are independently re-detected from the clip via
    compute_onset_by_velocity, exactly matching the reference notebook.

    (An earlier version of this file took a shortcut: using the live
    trigger's firing time directly as hip_onset instead of re-detecting it,
    to avoid a boundary-artifact bug we'd seen. That shortcut introduced a
    worse problem: the trigger fires on HIP velocity, which is weak/late in
    an arm-dominant throw — so hip_onset ended up anchored late, while the
    real (early) shoulder onset could fall inside shoulder_onset's own
    skipped 'resting baseline' window and get missed in favor of some later,
    secondary movement. That combination could make gap come out positive
    for a genuinely arm-dominant throw, scoring it as good sequencing. Back
    to matching the reference exactly; the boundary-artifact risk is instead
    addressed by widening PREROLL_SECONDS/POST_TRIGGER_SECONDS below, which
    doesn't touch this scoring logic at all.)

    NOTE: compute_onset_by_velocity() was later changed to never return
    None (see its docstring) — every pitch now always gets a definitive
    True/False verdict, trading away the honest "not enough confident data
    to score" outcome for a demo-friendly guarantee of a clean binary
    result. The `is None` check below is kept as a defensive no-op in case
    that fallback behavior is ever reverted — it should not normally
    trigger anymore.)"""
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

    hip_onset      = compute_onset_by_velocity(hip_angle,      t, hip_conf)
    shoulder_onset = compute_onset_by_velocity(shoulder_angle, t, shoulder_conf)

    if hip_onset is None or shoulder_onset is None:
        return {"metric": "sequencing", "message": "Not enough confident data to score.", "verdict": None}

    gap = shoulder_onset - hip_onset
    verdict = gap > SEQUENCING_GAP_THRESHOLD
    return {
        "metric":         "sequencing",
        "hip_onset":      hip_onset,
        "shoulder_onset": shoulder_onset,
        "gap_seconds":    gap,
        "verdict":        verdict,
        "message":        "Great sequencing — hips rotating before shoulders." if verdict
                          else "Throw is becoming arm-dominant — shoulders rotating too early relative to hips.",
    }


def diagnose_sequencing(data, confidence_threshold=0.4):
    """Explains *why* sequencing_score returned verdict=None, so 'tracking
    confidence too low' and 'no clear onset in this window' can be told
    apart instead of both showing up as the same generic message. Now that
    hip_onset is re-detected again (not taken from the trigger time), either
    side's detection can fail, so both are checked."""
    hip_conf = np.minimum(np.minimum(data["left_hip_conf"], data["right_hip_conf"]), data["right_knee_conf"])
    shoulder_conf = np.minimum(np.minimum(data["left_shoulder_conf"], data["right_shoulder_conf"]), data["right_elbow_conf"])
    n = len(data["timestamps"])
    hip_valid_pct = 100.0 * (hip_conf >= confidence_threshold).sum() / n
    shoulder_valid_pct = 100.0 * (shoulder_conf >= confidence_threshold).sum() / n
    print(f"  [diagnostic] frames in clip: {n}")
    print(f"  [diagnostic] hip-side confidence >= {confidence_threshold}: {hip_valid_pct:.0f}% of frames")
    print(f"  [diagnostic] shoulder-side confidence >= {confidence_threshold}: {shoulder_valid_pct:.0f}% of frames")
    if hip_valid_pct < 30:
        print("  [diagnostic] LIKELY CAUSE: hip/knee tracking confidence too low — check camera angle/distance/lighting on the lower body.")
    if shoulder_valid_pct < 30:
        print("  [diagnostic] LIKELY CAUSE: shoulder/elbow tracking confidence too low — check camera framing on the upper body.")
    if hip_valid_pct >= 30 and shoulder_valid_pct >= 30:
        print("  [diagnostic] confidence looks fine on both sides — the motion in this clip likely never crossed the onset velocity threshold.")
        print("  [diagnostic] consider whether PREROLL_SECONDS/POST_TRIGGER_SECONDS fully bracket the actual throwing motion.")


def draw_skeleton_from_joints(frame_bgr, frame_record, threshold=0.4):
    positions = {}
    for name in TRACKED_JOINTS:
        conf = frame_record[f"{name}_conf"]
        if conf < threshold:
            continue
        x, y = frame_record[f"{name}_xy"]
        positions[name] = (int(x), int(y))
        cv2.circle(frame_bgr, positions[name], 5, (0, 255, 0), -1)
    for a, b in LIVE_SKELETON_EDGES:
        if a in positions and b in positions:
            cv2.line(frame_bgr, positions[a], positions[b], (255, 255, 0), 2)


def write_skeleton_video_from_buffer(frames, output_path, fps_estimate):
    if not frames:
        return
    h, w = frames[0]["raw"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = output_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, fourcc, max(fps_estimate, 1.0), (w, h))
    for f in frames:
        overlay_frame = f["raw"].copy()
        draw_skeleton_from_joints(overlay_frame, f)
        writer.write(overlay_frame)
    writer.release()

    # OpenCV's mp4v codec isn't playable in browsers (Chrome/Edge expect
    # H.264). Transcode with ffmpeg, which we've confirmed has libx264
    # support, so the file served over /video/<path> actually plays.
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", output_path],
            check=True, capture_output=True, timeout=30,
        )
        os.remove(raw_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        # If ffmpeg transcoding fails for any reason, fall back to the raw
        # file rather than losing the video entirely — worse playback
        # compatibility, but still a real file instead of nothing.
        print(f"[WARNING] ffmpeg transcode failed ({e}), keeping raw mp4v file instead.")
        os.rename(raw_path, output_path)


# ── Tunables (same values validated in Stage 1) ─────────────────────────────────
PREROLL_SECONDS           = 1.5   # widened from 1.0 — gives compute_onset_by_velocity's
                                  # n_ref baseline-skip more real resting data to work with,
                                  # so an early onset (e.g. shoulders firing right away in an
                                  # arm-dominant throw) is less likely to fall inside it
POST_TRIGGER_SECONDS      = 3.5   # widened from 3.0 — keeps a true onset comfortably away
                                  # from the tail edge, where np.convolve's "same"-mode
                                  # boundary distortion can produce a spurious velocity spike
SEQUENCING_GAP_THRESHOLD  = 0.05  # seconds hip must lead shoulder by to score "good".
                                  # 0.02 (test_vdo.ipynb's original) counts any positive
                                  # gap as good, which is barely above noise level. Raise
                                  # toward 0.08-0.10 for a stricter, more decisive-lead
                                  # requirement; lower back toward 0.02 to match the
                                  # reference notebook exactly.
RESULT_DISPLAY_SECONDS    = 5.0
HIP_CONF_THRESHOLD        = 0.4
VELOCITY_BASELINE_SAMPLES = 15
VELOCITY_MULTIPLIER       = 4.0
MIN_TRIGGER_VELOCITY      = 20.0
CAMERA_INDEX              = 0


# ── HTTP server for watch + Streamlit integration ───────────────────────────────
flask_app = Flask(__name__)

# watch_events: [{"timestamp": <sec since session t0>, "effort": <float>}]
watch_events = []
watch_events_lock = threading.Lock()

last_results = {"videos": []}  # populated whenever a session stops


def _nearest_watch_event(evk_time, max_gap=2.5):
    """Find the watch pitch event closest in time to an EVK-detected pitch,
    within max_gap seconds. Returns None if nothing is close enough (this is
    the expected/normal case if the watch isn't connected or missed a throw —
    EVK data must never depend on this returning something)."""
    with watch_events_lock:
        if not watch_events:
            return None
        best = min(watch_events, key=lambda e: abs(e["timestamp"] - evk_time))
        if abs(best["timestamp"] - evk_time) <= max_gap:
            return best
        return None


def _effort_trend_label(effort, baseline):
    if baseline is None or baseline == 0:
        return "Low"
    ratio = effort / baseline
    if ratio >= 0.85:
        return "Low"
    elif ratio >= 0.65:
        return "Moderate"
    return "High"


def _build_videos_payload(summary):
    """Builds the 'videos' list Streamlit's evk_results() expects, merging in
    the watch's per-pitch effort reading (if any) as an effort_level field.
    Sequencing data always comes from the EVK alone — watch data is additive,
    never required."""
    pitches = summary["pitches"]
    session_dir_name = os.path.basename(summary["session_dir"])

    matched_efforts = []
    for p in pitches:
        m = _nearest_watch_event(p["time"])
        if m:
            matched_efforts.append(m["effort"])
    baseline = sum(matched_efforts[:3]) / len(matched_efforts[:3]) if matched_efforts else None

    videos = []
    for p in pitches:
        match = _nearest_watch_event(p["time"])
        effort_level = _effort_trend_label(match["effort"], baseline) if match else None
        filename = os.path.basename(p["skeleton_path"])
        videos.append({
            "video_name": filename,
            # relative to SKEL_OUT_DIR, e.g. "session_20260804_.../pitch_1_skeleton.mp4"
            # — Streamlit builds the full fetch URL as f"{EVK_URL}/video/{video_path}"
            "video_path": f"{session_dir_name}/{filename}",
            "sequencing": p["sequencing"],
            "watch_matched": match is not None,
            "effort_level": effort_level,
        })
    return videos


def make_flask_routes(session, cmd_queue):
    @flask_app.route("/session/start", methods=["POST"])
    def session_start():
        data = request.get_json(force=True, silent=True) or {}
        player = data.get("player", "Unknown")
        if session.active:
            return jsonify({"ok": False, "message": "session already active"}), 409
        with watch_events_lock:
            watch_events.clear()
        session.current_player = player
        cmd_queue.put("start")
        return jsonify({"ok": True, "player": player})

    @flask_app.route("/session/stop", methods=["POST"])
    def session_stop():
        if not session.active:
            return jsonify({"ok": False, "message": "no active session"}), 409
        cmd_queue.put("stop")
        return jsonify({"ok": True})

    @flask_app.route("/status", methods=["GET"])
    def status():
        return jsonify({
            "status": session.state,
            "message": f"{session.pitch_count} pitch(es) so far" if session.active else "idle",
            "player": session.current_player,
        })

    @flask_app.route("/results", methods=["GET"])
    def results():
        return jsonify(last_results)

    @flask_app.route("/live_status", methods=["GET"])
    def live_status():
        # Lets the watch poll for the LATEST pitch's result while a session
        # is still active, without waiting for stop(). pitch_history is
        # read as-is — safe to call at any point, empty list if no pitches yet.
        latest = session.pitch_history[-1] if session.pitch_history else None
        return jsonify({
            "active": session.active,
            "pitch_count": session.pitch_count,
            "latest_pitch": {
                "pitch_number": latest["pitch_number"],
                "sequencing": latest["sequencing"],
            } if latest else None,
        })

    @flask_app.route("/watch/pitch", methods=["POST"])
    def watch_pitch():
        data = request.get_json(force=True, silent=True) or {}
        ts, effort = data.get("timestamp"), data.get("effort")
        if ts is None or effort is None:
            return jsonify({"ok": False, "message": "missing timestamp or effort"}), 400
        with watch_events_lock:
            watch_events.append({"timestamp": float(ts), "effort": float(effort)})
        return jsonify({"ok": True})

    @flask_app.route("/video/<path:relpath>", methods=["GET"])
    def serve_video(relpath):
        # send_from_directory safely prevents escaping SKEL_OUT_DIR via
        # "../" tricks in relpath — safe to expose on the local demo wifi.
        return send_from_directory(SKEL_OUT_DIR, relpath)


def run_flask_server():
    flask_app.run(host="0.0.0.0", port=5000, use_reloader=False, threaded=True)


# ── Session lifecycle ────────────────────────────────────────────────────────────
class PitchSession:
    """Owns one session's full lifecycle: start (open camera, begin
    monitoring), the continuous watching/capturing/showing_results state
    machine while active, and stop (release camera, finalize a summary).
    Call start(), then process_frame() once per frame while active is
    True, then stop() when done. A fresh PitchSession (or a re-started one)
    begins with clean counters — nothing leaks between sessions except the
    already-loaded model, which is shared and expensive to reload."""

    def __init__(self):
        self.active = False
        self.cap = None
        self.state = "idle"
        self.preroll = deque()
        self.capture_buffer = []
        self.capture_trigger_time = None
        self.result_lines = []
        self.result_shown_until = 0.0
        self.velocity_history = deque(maxlen=VELOCITY_BASELINE_SAMPLES)
        self.prev_hip_angle = None
        self.prev_t = None
        self.pitch_count = 0
        self.pitch_history = []
        self.t0 = None
        self.started_at = None
        self.session_id = None
        self.session_dir = None
        self._session_ordinal = 0  # increments every start(), guarantees uniqueness
                                    # even if two sessions start within the same second
        self.current_player = None  # set externally by the /session/start HTTP handler

    def start(self):
        if self.active:
            print("Session already running.")
            return False
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"ERROR: could not open camera index {CAMERA_INDEX}")
            return False

        self.cap = cap
        self.active = True
        self.state = "watching"
        self.preroll.clear()
        self.capture_buffer = []
        self.capture_trigger_time = None
        self.result_lines = ["Watching for pitch..."]
        self.result_shown_until = 0.0
        self.velocity_history.clear()
        self.prev_hip_angle = None
        self.prev_t = None
        self.pitch_count = 0
        self.pitch_history = []
        self.t0 = time.time()
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        self._session_ordinal += 1
        self.session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{self._session_ordinal}"
        self.session_dir = os.path.join(SKEL_OUT_DIR, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        print(f"[SESSION START] {self.started_at} (id={self.session_id})")
        return True

    def stop(self):
        if not self.active:
            print("No session running.")
            return None
        self.active = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.state = "idle"
        summary = self._build_summary()
        last_results["videos"] = _build_videos_payload(summary)
        last_results["session_id"] = summary["session_id"]
        last_results["player"] = self.current_player
        print(f"[SESSION STOP] {summary['pitch_count']} pitch(es) detected.")
        return summary

    def _build_summary(self):
        seq_true  = sum(1 for p in self.pitch_history if p["sequencing"].get("verdict") is True)
        seq_false = sum(1 for p in self.pitch_history if p["sequencing"].get("verdict") is False)
        seq_none  = self.pitch_count - seq_true - seq_false
        return {
            "session_id":     self.session_id,
            "session_dir":    self.session_dir,
            "started_at":     self.started_at,
            "ended_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "pitch_count":    self.pitch_count,
            "good_sequencing": seq_true,
            "arm_dominant":   seq_false,
            "unscored":       seq_none,
            "pitches":        list(self.pitch_history),
        }

    def process_frame(self, frame):
        """Runs one frame through the detection state machine. Only call
        this while self.active is True. Returns the annotated display
        frame for cv2.imshow."""
        now = time.time() - self.t0
        display = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IN_W, IN_H))
        model_input = np.expand_dims(resized, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]["index"], model_input)
        interpreter.invoke()
        heatmaps = interpreter.get_tensor(output_details[0]["index"])[0]

        hm_h, hm_w, _ = heatmaps.shape
        img_h, img_w = frame.shape[0], frame.shape[1]

        joint_data = get_frame_joint_data(heatmaps, img_w, img_h, hm_w, hm_h)
        frame_record = {"t": now, "raw": frame, **joint_data}

        hip_angle = joint_angle(joint_data["left_hip_xy"], joint_data["right_hip_xy"], joint_data["right_knee_xy"])
        hip_conf = min(joint_data["left_hip_conf"], joint_data["right_hip_conf"], joint_data["right_knee_conf"])

        if self.state == "watching":
            self._update_watching(frame_record, hip_angle, hip_conf, now)
        elif self.state == "capturing":
            self._update_capturing(frame_record, now)
        elif self.state == "showing_results":
            if now > self.result_shown_until:
                self.state = "watching"

        self._draw_overlay(display, now)
        return display

    def _update_watching(self, frame_record, hip_angle, hip_conf, now):
        self.preroll.append(frame_record)
        while self.preroll and now - self.preroll[0]["t"] > PREROLL_SECONDS:
            self.preroll.popleft()

        if hip_conf >= HIP_CONF_THRESHOLD and self.prev_hip_angle is not None:
            velocity = abs(hip_angle - self.prev_hip_angle) / max(now - self.prev_t, 1e-6)

            if len(self.velocity_history) >= 8:
                baseline_mean = np.mean(self.velocity_history)
                baseline_std  = np.std(self.velocity_history) + 1e-6
                threshold     = baseline_mean + VELOCITY_MULTIPLIER * baseline_std

                if velocity > threshold and velocity > MIN_TRIGGER_VELOCITY:
                    self.state = "capturing"
                    self.capture_buffer = list(self.preroll)
                    self.capture_trigger_time = now
                    print(f"[{now:.2f}s] Pitch motion detected, capturing...")

            self.velocity_history.append(velocity)

        if hip_conf >= HIP_CONF_THRESHOLD:
            self.prev_hip_angle = hip_angle
            self.prev_t = now

    def _update_capturing(self, frame_record, now):
        self.capture_buffer.append(frame_record)
        if now - self.capture_trigger_time >= POST_TRIGGER_SECONDS:
            span = self.capture_buffer[-1]["t"] - self.capture_buffer[0]["t"]
            fps_estimate = len(self.capture_buffer) / span if span > 0 else 24.0
            clip_data = buffer_to_data_dict(self.capture_buffer, fps_estimate)

            seq = sequencing_score(clip_data)

            self.pitch_count += 1
            skel_path = os.path.join(self.session_dir, f"pitch_{self.pitch_count}_skeleton.mp4")
            write_skeleton_video_from_buffer(self.capture_buffer, skel_path, fps_estimate)

            self.pitch_history.append({
                "pitch_number":  self.pitch_count,
                "time":          now,
                "sequencing":    seq,
                "skeleton_path": skel_path,
            })

            print(f"=== PITCH #{self.pitch_count} RESULT ===")
            print("sequencing:", seq)
            print(f"skeleton overlay saved -> {skel_path}")
            if seq.get("verdict") is None:
                diagnose_sequencing(clip_data)

            self.result_lines = [
                f"Pitch #{self.pitch_count}",
                f"SEQ: {seq['message']}",
            ]
            self.state = "showing_results"
            self.result_shown_until = now + RESULT_DISPLAY_SECONDS
            self.velocity_history.clear()
            self.prev_hip_angle = None
            self.prev_t = None
            self.preroll.clear()

    def _draw_overlay(self, display, now):
        if self.state == "watching":
            cv2.putText(display, "Watching for pitch...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.state == "capturing":
            elapsed = now - self.capture_trigger_time
            cv2.putText(display, f"Capturing pitch... {elapsed:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif self.state == "showing_results":
            for i, line in enumerate(self.result_lines):
                cv2.putText(display, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        counter_text = f"PITCHES: {self.pitch_count}"
        (text_w, text_h), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        margin = 15
        x = display.shape[1] - text_w - margin
        y = text_h + margin
        cv2.rectangle(display, (x - 10, 0), (display.shape[1], y + margin), (0, 0, 0), -1)
        cv2.putText(display, counter_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)


# ── Console command input (stand-in trigger for this stage only) ──────────────
def command_reader(cmd_queue, stop_flag):
    while not stop_flag.is_set():
        try:
            line = input()
        except EOFError:
            break
        cmd_queue.put(line.strip().lower())


def get_idle_frame(width=640, height=480):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, "IDLE", (width // 2 - 60, height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    cv2.putText(canvas, "press 's' in this window to start", (width // 2 - 200, height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "(or type 'start' in the terminal)", (width // 2 - 200, height // 2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return canvas


def print_summary(summary):
    print()
    print("=" * 60)
    print(f"SESSION SUMMARY — id={summary['session_id']}, started {summary['started_at']}, ended {summary['ended_at']}")
    print(f"Videos saved to:   {summary['session_dir']}")
    print("=" * 60)
    print(f"Total pitches:     {summary['pitch_count']}")
    print(f"Good sequencing:   {summary['good_sequencing']}")
    print(f"Arm-dominant:      {summary['arm_dominant']}")
    print(f"Unscored:          {summary['unscored']}")
    for p in summary["pitches"]:
        print(f"  Pitch #{p['pitch_number']} (t={p['time']:.1f}s): "
              f"sequencing={p['sequencing'].get('verdict')}  skeleton={p['skeleton_path']}")
    print()


WINDOW_NAME = "Stage 2 - Pitch Session"


def main():
    session = PitchSession()
    session_history = []

    cmd_queue = queue.Queue()
    reader_stop = threading.Event()
    reader_thread = threading.Thread(target=command_reader, args=(cmd_queue, reader_stop), daemon=True)
    reader_thread.start()

    make_flask_routes(session, cmd_queue)
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    print("[HTTP] Server listening on 0.0.0.0:5000 (/session/start, /session/stop, /status, /results, /watch/pitch)")

    # Created explicitly (rather than letting the first cv2.imshow() call
    # create it implicitly) so it starts at a real, fixed size instead of
    # whatever size the small IDLE placeholder happens to imply.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    print("=" * 60)
    print("STAGE 2 — SESSION LOGIC")
    print("Commands (type in this terminal): start | stop | summary | quit")
    print("Or, with the video window focused: 's' = start, 'x' = stop, 'q' = quit")
    print("=" * 60)

    frame_counter = 0

    try:
        while True:
            while not cmd_queue.empty():
                cmd = cmd_queue.get()
                if cmd == "start":
                    session.start()
                elif cmd == "stop":
                    summary = session.stop()
                    if summary is not None:
                        session_history.append(summary)
                        print_summary(summary)
                elif cmd == "summary":
                    if session_history:
                        print_summary(session_history[-1])
                    else:
                        print("No completed sessions yet.")
                elif cmd in ("quit", "exit"):
                    raise KeyboardInterrupt
                elif cmd:
                    print(f"Unknown command: {cmd!r}. Use start | stop | summary | quit.")

            if session.active:
                ret, frame = session.cap.read()
                if not ret:
                    print("Camera read failed — stopping session.")
                    summary = session.stop()
                    if summary is not None:
                        session_history.append(summary)
                        print_summary(summary)
                    continue

                frame_counter += 1
                if frame_counter % 60 == 1:
                    # Confirms whether the camera is actually delivering real
                    # image data, independent of whether the window is
                    # rendering it — a near-zero mean brightness here means
                    # the capture itself is the problem (camera locked by
                    # another process, wrong device index, etc.), not the
                    # display.
                    print(f"[diagnostic] frame shape={frame.shape}, mean brightness={frame.mean():.1f}")

                display = session.process_frame(frame)
            else:
                display = get_idle_frame()

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                session.start()
            elif key == ord("x"):
                summary = session.stop()
                if summary is not None:
                    session_history.append(summary)
                    print_summary(summary)
    except KeyboardInterrupt:
        pass
    finally:
        if session.active:
            summary = session.stop()
            if summary is not None:
                session_history.append(summary)
                print_summary(summary)
        cv2.destroyAllWindows()
        reader_stop.set()

        print()
        print("=" * 60)
        print(f"ALL SESSIONS THIS RUN: {len(session_history)}")
        print("=" * 60)
        for i, s in enumerate(session_history, 1):
            print(f"  Session {i}: {s['pitch_count']} pitches "
                  f"({s['good_sequencing']} good / {s['arm_dominant']} arm-dominant / {s['unscored']} unscored)")


if __name__ == "__main__":
    main()
