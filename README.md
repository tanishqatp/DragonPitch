# DragonPitch: real-time hip-before-shoulder pitch sequencing on the Dragonwing IQ-8275 EVK on the Hexagon NPU

A pitcher throws in front of a webcam. A Dragonwing IQ-8275 EVK watches their hips and shoulders on its NPU and scores whether their sequencing is correct. A Pixel Watch on their wrist tracks effort. A laptop turns all of it into spoken-language coaching feedback from a local LLM. No cloud involved anywhere.

- **Hardware:** Qualcomm Dragonwing IQ-8275 EVK · Pixel Watch 4 (Snapdragon W5 Gen 2) · Snapdragon X Elite laptop
- **Difficulty:** Intermediate
- **Time:** ~1–2 hours, assuming the EVK and watch are already flashed/paired
- **Stack:** Python (Flask, TFLite + QNN, OpenCV) · Kotlin/Compose (Wear OS) · Streamlit + Ollama

Two Qualcomm silicon families (Dragonwing and Snapdragon), three tiers of compute, one pipeline — no single device does everything, and none of them talk to the cloud to do it.

## What you'll build

- A live pose-estimation pipeline on the EVK's Hexagon NPU that decides, in real time, whether a pitcher's hips are rotating before their shoulders.
- A Wear OS watch app that starts/stops a session and reports live effort back to the EVK.
- A Streamlit dashboard that pulls both together and asks a local LLM to write the pitcher a coaching note.

## Before you start

You'll need:

- [ ] A **Dragonwing IQ-8275 EVK**, flashed with Ubuntu, with a USB webcam attached.
- [ ] Qualcomm's **QAIRT SDK** installed on the EVK — the stock image does *not* ship `libQnnTFLiteDelegate.so` / `libQnnHtp.so`, so pose inference will silently fall back to CPU (or fail) until this is installed.
- [ ] A **Pixel Watch 4** (or another Wear OS 3+ device) and Android Studio, to build and side-load the watch app.
- [ ] A laptop (ideally Snapdragon X Elite, but any machine works) with **Python 3.10+**, **[Ollama](https://ollama.com)** installed, and the `gemma3` model pulled (`ollama pull gemma3`).
- [ ] All three devices on the **same local WiFi network** — there's no cloud hop, so they need to be able to reach each other directly.

## Repo structure

```
DragonPitch/
├── evk/
│   └── server.py           Flask server + pose pipeline (runs on the EVK)
├── watch/
│   ├── MainActivity.kt      Wear OS session UI
│   └── SessionViewModel.kt  Session state + peak-acceleration effort tracking
├── analytics/
│   └── streamlit_app.py    Coaching-analytics dashboard (runs on the laptop)
└── requirements.txt
```

Three independently-runnable pieces, one per device.

## Step 1: bring up the EVK pipeline

On the EVK:

```bash
pip install -r requirements.txt
```

Make sure the QAIRT SDK's delegate libraries are on the path (`libQnnTFLiteDelegate.so`, `libQnnHtp.so`, and the Hexagon firmware skel libraries) — consult Qualcomm's QAIRT SDK / AI Runtime SDK documentation if they aren't already installed.

Run the server:

```bash
cd evk
python server.py
```

It listens on `0.0.0.0:5000`. Confirm it's up from another machine on the same network:

```bash
curl http://<evk-ip>:5000/status
```

## Step 2: point the laptop app at the EVK

Edit the config block near the top of `analytics/streamlit_app.py`:

```python
EVK_URL    = "http://YOUR_EVK_IP:5000"  # put your Dragonwing IQ-8275 EVK's local network IP here
OLLAMA_URL = "http://localhost:11434/api/generate"
```

Then, with Ollama running and `gemma3` pulled:

```bash
cd analytics
streamlit run streamlit_app.py
```

## Step 3: build and install the watch app

Open the `watch/` folder in Android Studio, update the EVK IP the same way:

```kotlin
private val evkBaseUrl = "http://YOUR_EVK_IP:5000"  // put your Dragonwing IQ-8275 EVK's local network IP here
```

then build and install onto a Pixel Watch 4 (or any Wear OS 3+ device) over ADB or Wi-Fi debugging.

## Step 4: run a session

1. Start a session from the watch.
2. Pitch in front of the EVK's camera.
3. Watch the Streamlit app for the live sequencing verdict, skeleton overlay, and effort readout.
4. Ask for AI coaching feedback in the Streamlit UI once a few pitches are in.

## How the sequencing metric works

The core metric checks whether **hip rotation begins before shoulder rotation** — a scale-invariant, joint-angle representation (hip: left_hip→right_hip→right_knee; shoulder: left_shoulder→right_shoulder→right_hip) that stays meaningful regardless of camera distance. An earlier version tried to use wrist/elbow angle for the "arm" side of the comparison, but the arm has no genuine quiet baseline during a windup (glove adjustments, rocking motion, the arm passing behind the body all move it continuously) — so that signal was dropped in favor of the more stable, more spec-accurate hip/shoulder comparison.

By design, `compute_onset_by_velocity()` in `evk/server.py` always returns a definitive True/False verdict rather than an honest "not enough confident data" outcome — a deliberate demo-friendly tradeoff, not an oversight.

## Performance

| Metric | Result |
|---|---|
| NPU pose inference (per frame) | ~6.4 ms |
| Preprocess + decode (per frame) | ~2 ms combined |
| Camera capture (per frame) | ~40.1 ms (~85% of total frame time — the actual bottleneck) |
| Effective live pipeline rate | ~23.7 FPS (camera capped at 30 FPS) |
| NPU vs. CPU pose output delta | ~1 heatmap pixel across all load-bearing joints (consistent with INT8 vs. float32 quantization noise, not a bug) |
| Watch pitch classifier (trained, unshipped) | 91% accuracy, int8/uint8 quantized, 5 KB |

At ~24 FPS (one sample every ~42 ms), the pipeline has enough temporal resolution to tell which of hip/shoulder rotation starts first, though not to resolve very small sub-frame timing differences.

## Troubleshooting / things that came up during development

**NPU inference gives slightly different joint coordinates than CPU.** Expected — run the same input through both interpreters and compare; a delta of about 1 heatmap pixel across load-bearing joints is normal INT8 (NPU) vs. float32 (CPU) quantization noise, not a decode bug.

**The `libQnnTFLiteDelegate.so` path breaks NPU loading.** Load it by bare filename, not a path containing a slash (even `./`) — a path with a slash makes `dlopen` skip its normal search path and look only at that literal location.

**Watch/EVK pitch-to-effort correlation sometimes misfires.** Watch and EVK clocks aren't synchronized; correlation relies on a 2.5-second nearest-timestamp tolerance window, which can misfire under poor WiFi conditions.

**Gap timing or skeleton clips showing as zero / missing in the Streamlit UI.** Historically caused by key-name mismatches between what the server actually returns (`gap_seconds`, `skeleton_name`) and what the app read (`gap`, `skeleton_video`) — already fixed in the current code, but worth knowing about if you extend it.

## Known limitations

- No trained edge AI model runs on the Pixel Watch. A pitch-vs-idle classifier was trained to 91% accuracy (Edge Impulse, Spectral Analysis features + a small Keras model), but integrating it required either an NDK/JNI build of Edge Impulse's C++ inferencing library or hand-matching its DSP pipeline in Kotlin — both unfinished when time ran out. The watch ships with a simpler peak-acceleration "Effort Trend" heuristic instead.
- The laptop's local LLM runs on CPU via Ollama, not the Hexagon NPU. A GenieX-based swap (which would route through QNN to the NPU) was built and its CLI-level bugs fixed, but the app-level swap was never completed.
- The sequencing metric always returns True/False by design — no "not enough confident data" outcome.
- The onset-timing chart component exists in the analytics app but isn't wired into any page yet.
- The live capture loop was validated in a development sandbox, not real-world deployment conditions.

## Future improvements

- Finish on-device watch-side pitch classification.
- Complete the GenieX swap so the coaching LLM runs on the X Elite's Hexagon NPU.
- Wire in the existing onset-timing chart component.
- Tighten watch/EVK time synchronization (e.g. NTP-style offset correction).
- Add an honest "not enough confident data" sequencing outcome.
- Field-test the live capture loop outside the development sandbox.

## Docs referenced

Qualcomm's QAIRT SDK / AI Runtime SDK docs, the pose model's own model card, Wear OS Health Services/Health Connect docs, Edge Impulse's deployment docs, and GenieX's docs + Qualcomm AI Hub.
