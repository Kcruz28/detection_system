# Real-Time Drone Detection & Tracking System
### Portfolio project for CV/ML defense-tech roles (e.g., Allen Control Systems)

## Why this project
This project is designed to map directly onto a job description like ACS's Computer Vision & Machine Learning role, which calls out:
- Real-time drone **detection, tracking, and classification**
- Models that run in **resource-constrained environments**
- Integration with **embedded systems and hardware**
- Familiarity with **sensors** (cameras, LIDAR, RADAR)
- Proficiency in **Python, C++, PyTorch/TensorFlow**

Each step below is written to hit one of those keywords with a real, demonstrable artifact — not just a line on a resume.

---

## Build Steps

### 1. Get a drone dataset
- Pull a public labeled dataset from **Roboflow Universe** (search "drone detection" or "UAV detection") — many already have bounding boxes.
- Alternative: scrape drone footage from YouTube and label it yourself with **CVAT** or **LabelImg**.
- Target: a few thousand images minimum for a reasonable fine-tune.

### 2. Train a real-time detector
- Fine-tune **YOLOv8** (or YOLO-NAS) in PyTorch on your dataset.
- Track mAP, precision/recall, and inference speed (FPS) on a validation set.
- This directly answers "real-time drone detection" in the JD.

### 3. Add multi-object tracking
- Layer **ByteTrack** or **DeepSORT** on top of the detector so objects keep a consistent ID across frames.
- Demonstrates understanding of motion/state estimation (Kalman filters) — core robotics/CV knowledge.

### 4. Add a classification layer
- Extend the model to distinguish drone types, or drones vs. birds vs. planes, as a secondary head or separate classifier.
- Hits "classification" explicitly and shows depth beyond a tutorial-level detector.

### 5. Optimize for embedded deployment
- Quantize the model (INT8/FP16) and deploy on a **Jetson Nano/Orin Nano** or **Raspberry Pi + Coral TPU**.
- Benchmark FPS and latency on-device vs. on a full GPU.
- This is the highest-leverage step — most applicants never get past a notebook, and "resource-constrained environments" is called out directly in the posting.

### 6. (Optional) Build a tracking gimbal demo
- Mount a webcam on a 2-axis pan-tilt servo rig (Arduino or Raspberry Pi controlled) so the camera visually follows a detected target in real time.
- Purely observational — no weapon or firing mechanism involved — but makes for a compelling demo video.

### 7. Document and ship it
- Push clean, commented code to **GitHub** with a README covering architecture, benchmarks, and a demo GIF/video.
- Write a short **LinkedIn post or blog walkthrough** — recruiters and hiring managers at small defense-tech startups often actually read these.

---

## Practical Notes

- **Timeline:** Steps 1–4 are a solid weekend-to-week project if you're comfortable with PyTorch. Step 5 (embedded deployment) usually takes longest due to quantization/driver setup — budget extra time.
- **Dataset sourcing:** Roboflow Universe is the fastest starting point for pre-labeled drone data.
- **Resume framing:** Lead with numbers, not just tech names.
  - Weak: *"Built a drone detection model using YOLOv8."*
  - Strong: *"Trained a YOLOv8-based drone detector (87% mAP) with multi-object tracking, deployed at 24 FPS on a Jetson Nano."*

## Stretch Goals
- Sensor fusion: simulate combining camera + simulated RADAR data with a Kalman filter for more robust tracking.
- Edge-to-cloud pipeline: stream detection events to a simple dashboard.
- Adversarial robustness: test detector performance under occlusion, glare, or small/distant targets.

---

## Resources
- Roboflow Universe: https://universe.roboflow.com
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com
- ByteTrack: https://github.com/ifzhang/ByteTrack
- NVIDIA Jetson getting started: https://developer.nvidia.com/embedded/learn/getting-started-jetson
