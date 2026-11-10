# DragonPitch

**Your hips don't lie, and now your watch won't either.** A real-time pitching coach that runs pose estimation on a Qualcomm Dragonwing IQ-8275, shows verdicts on a Pixel Watch 4 and writes coaching feedback on a Snapdragon X Elite laptop, all without the cloud.

## Devices & Components

| Item | Qty | Role |
| --- | --- | --- |
| Qualcomm Dragonwing IQ-8275 EVK | 1 | Camera capture, pose estimation on the Hexagon NPU, sequencing score, Flask server |
| USB webcam | 1 | Watches the pitcher |
| Google Pixel Watch 4 (Snapdragon W5 Gen 2) | 1 | Start/stop control, live verdict, effort tracking |
| Snapdragon X Elite laptop | 1 | Coaching dashboard and local LLM |

## Software & Tools

- [Ubuntu for Qualcomm IoT](https://ubuntu.com/download/qualcomm-iot), the operating system on the IQ-8275 EVK
- [Qualcomm Launcher](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_Launcher), to flash Ubuntu onto the EVK
- [Qualcomm AI Runtime (QAIRT)](https://www.qualcomm.com/developer/software/qualcomm-ai-runtime-sdk-qairt), which provides the QNN delegate that runs the model on the Hexagon NPU
- [HRNetPose from Qualcomm AI Hub](https://aihub.qualcomm.com/models/hrnet_pose), the pose estimation model
- [TensorFlow Lite](https://www.tensorflow.org/lite), [OpenCV](https://opencv.org/) and [Flask](https://flask.palletsprojects.com/), for the EVK's vision pipeline and server
- [FFmpeg](https://ffmpeg.org/), to encode the skeleton replays
- [Android Studio](https://developer.android.com/studio) and [Wear OS](https://developer.android.com/wear) with Jetpack Compose, for the watch app
- [Streamlit](https://streamlit.io/), for the coaching dashboard
- [Ollama](https://ollama.com/) with [Gemma 3](https://ollama.com/library/gemma3), for local AI coaching

## Project overview

Every great pitch starts in the hips. When the hips rotate before the shoulders, power flows up the body into the arm. When the shoulders fire first, the arm does all the work. The difference lasts a fraction of a second, far too fast to see from the dugout.

DragonPitch measures it on every throw. It's built in three layers, each running on a different tier of Qualcomm silicon:

- **Vision, on the Dragonwing IQ-8275 EVK.** A USB webcam feeds HRNetPose running on the Hexagon NPU. The EVK detects each pitch, times when the hips and shoulders start rotating, and scores the sequencing.
- **Wearable, on the Pixel Watch 4 (Snapdragon W5 Gen 2).** The pitcher starts and stops sessions from the wrist, sees each verdict, and the watch's accelerometer tracks arm effort.
- **Coaching, on a Snapdragon X Elite laptop.** A Streamlit dashboard shows skeleton replays and per-pitch results, and Gemma 3 running locally writes coaching notes from the data.

The EVK is the hub. It runs a small Flask server, and the watch and laptop talk to it over local WiFi. Nothing is sent to the cloud.

## What you'll build

DragonPitch brings all three layers together, but each one is useful on its own. Build the full system, or start with just the part you need:

- **The full system:** EVK, watch and laptop working together, with live verdicts on the wrist and AI coaching after every session. This is how DragonPitch is designed to run.
- **Pose estimation only:** just the EVK and a webcam. Run the EVK server, start a session from the keyboard, and see each pitch's verdict and skeleton overlay on screen.
- **Coaching analysis only:** just a laptop with Ollama. The dashboard comes with sample sessions for five players, so you can try Gemma's team feedback without any other hardware, then connect an EVK later for live data.

## Key features

1. Real-time pose estimation with HRNetPose, accelerated on the Hexagon NPU.
2. Hardware-accelerated inference through Qualcomm's QNN TensorFlow Lite delegate (QAIRT).
3. Three-layer system across Qualcomm silicon:
   - Pose estimation on the Dragonwing IQ-8275
   - Session control and effort tracking on the Snapdragon W5 Gen 2
   - AI coaching on the Snapdragon X Elite
4. 17-keypoint body pose detection.
5. Automatic pitch detection from hip rotation speed.
6. Hip-to-shoulder sequencing score with an adjustable timing threshold.
7. Skeleton replay clip for every pitch.
8. Live sequencing verdicts on the watch.
9. Effort Trend (Low, Moderate or High) from the watch's accelerometer.
10. Coaching feedback from Gemma 3, running locally through Ollama.
11. HTTP API on the EVK for the watch, the dashboard and other apps.
12. Fully local, with no cloud required.

## Developer workflow

1. **Flash the EVK.** Use [Qualcomm Launcher](https://dragonwingdocs.qualcomm.com/Ubuntu/devices/iq8275-evk/update-software/flash-using-qualcomm-launcher) to flash Ubuntu onto the IQ-8275, then connect the USB webcam.

2. **Install the NPU runtime.** Stock Ubuntu doesn't include the TensorFlow Lite QNN delegate, so install QAIRT and the project's dependencies:

   ```bash
   sudo apt install qairt-libs qairt-headers qairt-tools ffmpeg
   git clone https://github.com/tanishqatp/DragonPitch.git
   cd DragonPitch && pip install -r requirements.txt
   ```

3. **Get the model working on the CPU.** Download the quantized (w8a8) TFLite version of [HRNetPose](https://aihub.qualcomm.com/models/hrnet_pose) and save it as `evk/hrnet_pose.tflite`. Before involving the NPU, confirm it takes a 256×192 image, returns 64×48 heatmaps for 17 joints, and puts each joint on the right body part in a test photo.

4. **Move it to the NPU.** Load Qualcomm's QNN delegate with the HTP (Hexagon NPU) backend and pass it to the interpreter. Then run the same photo through both and compare. On our test photo, every joint peak matched or landed within two heatmap pixels.

   ```python
   delegate = tf.lite.experimental.load_delegate(
       "libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
   interp = tf.lite.Interpreter(model_path, experimental_delegates=[delegate])
   ```

5. **Run the EVK server.** It opens the camera, detects pitches and serves results over HTTP on port 5000. For pose estimation only, stop here and use the `s` and `x` keys in the video window to start and stop sessions.

   ```bash
   cd evk && python server.py
   ```

6. **Build the watch app.** Create a Wear OS project in Android Studio, add `MainActivity.kt` and `SessionViewModel.kt` from `watch/`, set `evkBaseUrl` to the EVK's IP, allow the INTERNET permission and cleartext HTTP, and install it on the watch.

7. **Start the coaching dashboard.** On the laptop, set `EVK_URL` in `analytics/streamlit_app.py`, then run:

   ```bash
   ollama pull gemma3
   streamlit run analytics/streamlit_app.py
   ```

8. **Throw.** Keep the pitcher's full body in frame from windup to follow-through, tap Start on the watch, and pitch. Verdicts appear on the wrist, and replays and coaching appear in the dashboard.

## Repo structure

```
DragonPitch/
├── evk/
│   └── server.py            Pose pipeline and Flask server (runs on the EVK)
├── watch/
│   ├── MainActivity.kt      Wear OS session UI
│   └── SessionViewModel.kt  Session control and effort tracking
├── analytics/
│   └── streamlit_app.py     Coaching dashboard (runs on the laptop)
└── requirements.txt
```
