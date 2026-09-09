# Lesson: Multi-Object Tracking with ByteTrack and DeepSORT

This is a teaching document, not a spec to copy-paste from. It explains how
tracking-by-detection works, how ByteTrack and DeepSORT each solve it
differently, and how either would wire into this repo's YOLO26 detector —
but it deliberately stops short of implementing the tracker. That part is
yours: see `src/tracking/base_tracker.py` (the interface) and
`src/tracking/README.md` (the checklist).

---

## 1. What "multi-object tracking" actually means

A detector like YOLO26 answers one question per frame: *"what objects are
in this image, and where?"* It has zero memory. Run it on frame 1 and frame
2 of a video and you get two independent sets of boxes — nothing in the
detector's output says "the box at (120, 340) in frame 2 is the same drone
as the box at (115, 335) in frame 1." Every frame, every object gets
detected fresh, with no identity carried forward.

For a lot of CV tasks that's fine. It is not fine if you want to answer
questions like:
- "How long has this drone been in view?"
- "What direction and speed is it moving?"
- "Is this the same drone that just flew behind that tower, or a new one?"
- Anything involving counting unique objects, trajectory analysis, or
  predicting where something will be next (which is exactly what you'd
  need for a tracking gimbal, per the project's stretch goal).

**Multi-object tracking (MOT)** solves this by assigning a persistent ID to
each object and carrying it across frames. The dominant modern approach —
and what both ByteTrack and DeepSORT do — is **tracking-by-detection**:

1. Run the detector on each frame independently (you already have this —
   YOLO26).
2. Predict where each *existing* track should be this frame, based on its
   recent motion.
3. Match this frame's new detections against the predicted tracks.
4. Update matched tracks with their new detection; spin up new tracks for
   unmatched detections; age out tracks that got no match.

Everything below is the machinery behind steps 2-4.

---

## 2. The building blocks

### 2.1 IoU-based association — the matching problem

**IoU** (Intersection over Union) measures overlap between two boxes:
`area(A ∩ B) / area(A ∪ B)`, from 0 (no overlap) to 1 (identical boxes). If
a track's predicted box and a new detection's box overlap heavily, they're
probably the same object — objects don't teleport between frames at 30 fps.

The problem is that you don't have one track and one detection — you have
`N` tracks and `M` detections, and you need to decide the best pairing
between the two entire sets simultaneously, not just greedily pick the
best match for each detection one at a time (greedy can lock in an easy
match early and force a worse match later that a global solution would
have avoided). This global "who pairs with whom" problem is the
**assignment problem**.

### 2.2 The Hungarian algorithm — the intuition

You build a cost matrix: rows = existing tracks, columns = new detections,
cell `(i, j)` = cost of matching track `i` to detection `j` (a common
choice: `1 - IoU`, so perfect overlap costs 0). The Hungarian algorithm
(a.k.a. Kuhn-Munkres) finds the assignment that **minimizes total cost
across the whole matrix** in polynomial time — you don't need to know the
proof, just that:

- It's solving "match rows to columns, one-to-one, to minimize the sum of
  chosen cells" — exactly the track↔detection pairing problem.
- It's exact and fast enough for the small N/M you'll have per frame (tens
  of objects, not thousands).
- In Python you almost never implement it by hand:
  `scipy.optimize.linear_sum_assignment(cost_matrix)` or the `lap` package
  (`lap.lapjv`, faster, what ByteTrack's reference implementation uses)
  both give you the optimal assignment directly.
- You also need a threshold: even the "best available" match might be
  garbage (IoU near 0), so any pairing above a cost threshold gets
  rejected rather than forced.

### 2.3 Kalman filters — predicting motion

Between frames, an object moves. Rather than assuming it stays exactly
where it was, both ByteTrack and DeepSORT keep a **Kalman filter** per
track that models position + velocity, and does two things every frame:

1. **Predict**: "based on this track's velocity, where should it be *this*
   frame?" — this gives you the predicted box to run IoU against, instead
   of matching against last frame's stale position.
2. **Correct**: once a detection is matched to the track, blend the
   filter's prediction with the actual measured box — the filter doesn't
   fully trust either the prediction or the noisy detection alone, it
   weighs them (via the filter's uncertainty estimate) to get a smoothed
   position/velocity estimate.

You don't need the covariance math to use one effectively — the intuition
"predict where it should be, then nudge that estimate toward what you
actually observed" is 90% of what you need to reason about tracker
behavior (e.g. why a track can survive a couple of missed detections: the
filter keeps predicting motion even with no measurement to correct with).
Both `filterpy` (a standalone Kalman filter library) and the tracker
libraries below implement this for you — you will not write Kalman filter
math from scratch unless you want to.

### 2.4 Track lifecycle states

A track isn't just "exists" or "doesn't." Implementations vary but the
common states are:

- **Tentative**: just created from an unmatched detection. Not trusted yet
  — could be a false-positive detection. Needs to match again for a few
  consecutive frames before being promoted.
- **Confirmed**: matched enough times in a row; this is a "real" track you
  report/draw.
- **Lost / coasting**: confirmed track that got no match this frame (maybe
  a missed detection, brief occlusion). Kept alive, predicted forward via
  the Kalman filter alone, for some `max_age` frames.
- **Deleted**: exceeded `max_age` with no match — removed for good. Its
  next detection (if the object reappears) will start a brand-new track ID
  unless the tracker has re-identification (DeepSORT's appearance model
  can sometimes recover the same ID; ByteTrack, being motion-only, usually
  can't).

This state machine exists because raw per-frame matching is noisy —
without "tentative," every spurious detection becomes a track; without
"lost," every single missed detection (motion blur, brief occlusion, low
confidence) kills a perfectly good track and gives the same object a new
ID next frame.

---

## 3. ByteTrack

### What makes it different

Vanilla SORT (ByteTrack's predecessor) throws away low-confidence
detections before matching — anything below a confidence threshold is
treated as background noise and discarded. ByteTrack's key insight
("BYTE" is the name of the association strategy) is that **low-confidence
detections are often real objects that are just partially occluded or
blurry, not noise** — throwing them away loses exactly the frames where
you'd want tracking to help you survive occlusion by prediction.

Instead, ByteTrack does **two rounds of matching per frame**:
1. Match high-confidence detections to tracks first (standard IoU +
   Hungarian).
2. Take the *remaining* unmatched tracks (this is the trick — not the
   remaining detections) and try to match them against the low-confidence
   detections. A track that was confidently tracked last frame and now
   only has a weak nearby detection is still probably the same object
   partially visible — match it. Low-confidence detections that don't
   match anything are discarded as noise, but only after this second
   pass.

This gets a meaningful accuracy boost for free, with no extra model, no
appearance embedding, no extra dependency beyond a detector.

### When to prefer it

- No appearance model → fast, low compute, no GPU needed for the tracker
  itself.
- Works best when objects don't fully leave frame or get fully occluded
  for long — it's motion-only, so once a track is deleted it can't
  recognize "oh, that's the same drone from 40 frames ago."
- Good fit for **open-sky drone footage**: small, fast, usually
  unoccluded targets against sky/background, exactly the profile SORT-
  family trackers do well on.

### Setup — be honest about install friction

The canonical repo, https://github.com/ifzhang/ByteTrack, is a full
research codebase (custom fork of an older YOLOX repo, pinned old
dependency versions, not published as a clean pip package). Vendoring it
directly into this repo would mean fighting dependency pins for a
narrow slice of functionality — not worth it here.

Two better-maintained, pip-installable paths that implement the same
BYTE association algorithm:

- **`supervision`** (Roboflow's CV toolkit) ships `sv.ByteTrack`, a clean
  reimplementation with a small, stable API. This is the easiest path and
  plays nicely with Ultralytics output.
  ```bash
  pip install supervision
  ```
- **`ultralytics` built-in tracking** — Ultralytics ships ByteTrack (and
  BoT-SORT) config-driven tracking via `model.track(...)` instead of
  `model(...)`, no extra install beyond `ultralytics` itself (already a
  dependency of this repo). This is the *lowest-friction* option since
  it's already installed, but it bundles detection+tracking into one call,
  which is less useful for learning the association logic yourself.

For this lesson/exercise, **`supervision.ByteTrack` is the recommended
path** — it's a real install, it exposes the update loop explicitly (you
call `.update_with_detections()` yourself per frame), and it's the one
that will actually teach you the tracking-by-detection loop rather than
hiding it behind `.track()`.

### Walkthrough (pseudocode-level, intentionally not a working wrapper)

```python
# Illustrative only — this is the shape of the loop, not a finished
# implementation. You'll write the real version in
# src/tracking/bytetrack_tracker.py.

# tracker = sv.ByteTrack()  # holds all track state across frames
#
# for frame in video:
#     results = model(frame)[0]                 # YOLO26 raw detections
#     detections = sv.Detections.from_ultralytics(results)
#
#     tracked = tracker.update_with_detections(detections)
#     # tracked.tracker_id[i] is now a stable int ID per box,
#     # persisted across frames for the same object.
#
#     for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
#         draw_box_with_id(frame, box, track_id)
```

Internally, `update_with_detections` is doing exactly what section 2
described: Kalman-predict existing tracks forward, split this frame's
detections into high/low confidence, run two rounds of IoU + Hungarian
matching, then update track states (tentative→confirmed, confirmed→lost,
lost→deleted).

---

## 4. DeepSORT

### What it adds on top of SORT

DeepSORT keeps everything SORT does (Kalman motion prediction, IoU
association) and adds a **CNN re-identification (re-ID) embedding**: a
small network produces an appearance feature vector (e.g. 128-d) for each
detected object crop. The matching cost becomes a combination of motion
distance *and* appearance similarity (cosine distance between embeddings),
not IoU alone.

Why this matters: IoU-only matching fails when an object is *fully*
occluded for several frames — there's no box to compute IoU against, so
the track gets deleted, and re-appearance starts a new ID. DeepSORT keeps
a gallery of recent appearance embeddings per (now-lost) track, so when a
new detection shows up later, it can compare "does this look like the
drone that disappeared 15 frames ago?" even with no motion continuity, and
recover the same ID.

### When to prefer it over ByteTrack

- Occlusion-heavy scenes: a drone passing behind a pole, tree, or another
  drone; a target that briefly leaves frame and re-enters.
- Scenes with visually distinct objects where appearance is actually
  discriminative (less useful if every object is a small, low-detail dot
  against sky, where there's not much for the embedder to latch onto —
  worth noting as a real limitation for small/distant drone footage
  specifically).
- You're willing to pay the cost: an extra forward pass (the embedder
  CNN) per detected box, per frame — real compute, no longer "basically
  free" the way ByteTrack is.

### Setup

The original DeepSORT repo (`nwojke/deep_sort`) is also research code with
a TensorFlow 1.x embedder — not something to depend on today. The
maintained, pip-installable option:

```bash
pip install deep-sort-realtime
```

`deep-sort-realtime` bundles a lightweight embedder (default: a small
MobileNet-style re-ID net, downloaded on first use) and exposes a
straightforward `DeepSort` class. This is the one referenced in
`src/tracking/README.md`'s checklist.

### Walkthrough (pseudocode-level)

```python
# Illustrative only — see src/tracking/README.md step 3 for what you'll
# actually build in src/tracking/deepsort_tracker.py.

# from deep_sort_realtime.deepsort_tracker import DeepSort
# tracker = DeepSort(max_age=30)  # frames a lost track survives
#
# for frame in video:
#     results = model(frame)[0]
#     # deep-sort-realtime wants ([x, y, w, h], confidence, class_id) tuples
#     raw_dets = [
#         ([x1, y1, x2 - x1, y2 - y1], conf, cls)
#         for (x1, y1, x2, y2), conf, cls in yolo_boxes_conf_cls(results)
#     ]
#
#     tracks = tracker.update_tracks(raw_dets, frame=frame)  # frame needed
#                                                             # for re-ID crops
#     for t in tracks:
#         if not t.is_confirmed():
#             continue
#         draw_box_with_id(frame, t.to_ltrb(), t.track_id)
```

Note `frame=frame` is required here (unlike ByteTrack) — the embedder
needs actual pixels to crop and encode each box.

---

## 5. ByteTrack vs. DeepSORT — side by side

| | ByteTrack | DeepSORT |
|---|---|---|
| Motion model | Kalman filter | Kalman filter |
| Appearance model | None | CNN re-ID embedding (extra forward pass/box) |
| Association cost | IoU only (two-pass: high+low conf) | Motion + appearance (cosine distance) |
| Speed | Fast — negligible overhead over detection | Slower — embedder cost scales with object count |
| Occlusion recovery | Poor — deleted tracks get new IDs on reappearance | Better — appearance gallery can re-match after full occlusion |
| Dependencies | `supervision` (or `ultralytics.track`) | `deep-sort-realtime` (+ downloaded embedder weights) |
| Best for | Small/fast objects, rarely occluded, real-time-sensitive | Occlusion-heavy or crossing-path scenes, visually distinct objects |

**For this project specifically:** drone detection footage is usually a
single small, fast-moving object against open sky — exactly ByteTrack's
sweet spot, and cheap enough to actually hit real-time FPS numbers you can
report. **Start with ByteTrack.** Reach for DeepSORT if your demo footage
has occlusion or multiple drones crossing paths — where losing an ID to a
tree or a crossing target would actually look bad in a demo video — since
that's the scenario its extra appearance cost is buying you something
ByteTrack structurally can't do.

---

## 6. Wiring a tracker to this repo's YOLO26 detector

Regardless of which tracker you pick, the shape of the integration is the
same, and matches `src/tracking/base_tracker.py`'s interface:

```
frame (np.ndarray, BGR)
   │
   ▼
model(frame)  # src/utils/inference.py already wraps this for single images
   │  results[0].boxes.xyxy   -> (N, 4) float tensor
   │  results[0].boxes.conf   -> (N,)   float tensor
   │  results[0].boxes.cls    -> (N,)   float tensor (class ids)
   ▼
convert to List[Detection]  # src/tracking/base_tracker.py's Detection dataclass
   │
   ▼
YourTrackerWrapper.update(detections, frame) -> List[Track]
   │  each Track has a stable track_id across frames
   ▼
draw boxes + track_id labels on frame
```

Concretely, going from raw YOLO26 output to `Detection` objects looks like:

```python
results = model(frame)[0]
detections = [
    Detection(
        bbox=tuple(box.tolist()),
        confidence=float(conf),
        class_id=int(cls),
    )
    for box, conf, cls in zip(
        results.boxes.xyxy, results.boxes.conf, results.boxes.cls
    )
]
```

Because YOLO26 is end-to-end/NMS-free, this is already the final clean
per-frame detection set — no extra NMS or confidence filtering step needed
before handing it to a tracker (older YOLO versions with NMS post-
processing would give you the same shape here, so nothing about the
wiring is YOLO26-specific — it just means you don't need to worry about
duplicate/overlapping boxes confusing the tracker's association step).

**Where this plugs into the actual repo:**

- `src/tracking/base_tracker.py` — the interface both wrappers implement
  (already scaffolded, don't add logic here).
- `src/tracking/bytetrack_tracker.py`, `src/tracking/deepsort_tracker.py`
  — you write these (see `src/tracking/README.md`).
- `frontend/app.py`'s video-handling path is where a tracker would
  eventually get called per-frame instead of just drawing raw detections
  — see step 4 in `src/tracking/README.md`'s checklist. **Not implemented
  yet, and out of scope until you've built and tested a wrapper
  independently.**

---

## 7. Exercise scaffold

This lesson intentionally does not include a working tracker. What exists
right now:

- `src/tracking/base_tracker.py` — abstract `Tracker` interface, plus
  `Detection` and `Track` dataclasses. No tracking logic.
- `src/tracking/README.md` — a step-by-step checklist (IoU/Hungarian
  warm-up → `ByteTrackWrapper` → `DeepSortWrapper` → optional demo wiring),
  plus a suggested verification test (consistent track ID across a drone
  crossing the frame; expected ID-switch behavior on full occlusion for
  each tracker).

Go implement it there.

---

## 8. Further reading

- ByteTrack paper: *ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box* — https://arxiv.org/abs/2110.06864
- ByteTrack reference repo: https://github.com/ifzhang/ByteTrack
- DeepSORT paper: *Simple Online and Realtime Tracking with a Deep
  Association Metric* — https://arxiv.org/abs/1703.07402
- DeepSORT reference repo: https://github.com/nwojke/deep_sort
- SORT paper (DeepSORT/ByteTrack's common ancestor): *Simple Online and
  Realtime Tracking* — https://arxiv.org/abs/1602.00763
- `supervision` tracking docs:
  https://supervision.roboflow.com/latest/trackers/
- `deep-sort-realtime` (maintained pip package):
  https://github.com/levan92/deep_sort_realtime
- Ultralytics built-in tracking (`model.track()`):
  https://docs.ultralytics.com/modes/track/
- Kalman filter intuition (no derivation required, but good if curious):
  https://www.kalmanfilter.net/kalman1d.html
- Hungarian algorithm overview:
  https://en.wikipedia.org/wiki/Hungarian_algorithm
