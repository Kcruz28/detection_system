# src/tracking/ — implementation checklist

This module is a **learning exercise**, not a finished feature. The full
lesson (concepts, how ByteTrack and DeepSORT actually work, setup options,
wiring into this repo) lives in
[`docs/lessons/tracking_bytetrack_deepsort.md`](../../docs/lessons/tracking_bytetrack_deepsort.md).
Read that first — this file is just the TODO checklist.

`base_tracker.py` defines the interface (`Tracker`, `Track`, `Detection`).
Nothing below it is implemented on purpose. You implement it.

## Suggested order of attack

### 1. Warm up: IoU + Hungarian matching by hand
- [ ] Write a small `iou(box_a, box_b)` function.
- [ ] Write `build_cost_matrix(dets, tracks)` using `1 - iou`.
- [ ] Use `scipy.optimize.linear_sum_assignment` (or `lap.lapjv`) to solve
      the assignment and confirm you understand what the returned index
      pairs mean.
- [ ] Add an IoU threshold — a "valid" match below some IoU should still be
      rejected (treated as no match), not forced.

This has nothing tracker-specific in it yet, but it's the core of both
ByteTrack's association step and SORT's / DeepSORT's motion-cost step, so
get comfortable with it before pulling in a library.

### 2. `ByteTrackWrapper` (recommended first tracker)
- [ ] Decide install path (see lesson doc — `supervision.ByteTrack` is the
      easiest to get running against YOLO26 output).
- [ ] Write `src/tracking/bytetrack_tracker.py` with a `ByteTrackWrapper(Tracker)`
      that:
  - [ ] Converts YOLO26 `Results` (boxes/conf/cls) into whatever array
        shape the underlying tracker expects.
  - [ ] Calls the underlying tracker's per-frame update in `update()`.
  - [ ] Maps its output back into `Track` objects (this repo's dataclass),
        preserving `track_id`.
  - [ ] Implements `reset()`.
- [ ] Verify: run it over a short drone clip and print `track_id` per frame
      for one drone — confirm the same ID persists across the whole clip
      (see "Suggested test" below).

### 3. `DeepSortWrapper` (second tracker, do this once ByteTrack works)
- [ ] `pip install deep-sort-realtime`.
- [ ] Write `src/tracking/deepsort_tracker.py` with a `DeepSortWrapper(Tracker)`
      that:
  - [ ] Converts detections to `([x, y, w, h], confidence, class_id)` tuples
        (that library's expected input — check its README, formats change).
  - [ ] Passes the raw `frame` through so the embedder can crop + embed
        each detection.
  - [ ] Maps `Track` objects from the library's track objects back to this
        repo's `Track` dataclass.
- [ ] Compare behavior against ByteTrack on a clip with occlusion (one
      drone passing behind another object, or briefly leaving frame) — this
      is the case DeepSORT should win.

### 4. Wire into the demo (optional, once at least one wrapper works)
- [ ] In `frontend/app.py`'s video path, instantiate your tracker once per
      video, call `.update(detections, frame)` per frame instead of just
      drawing raw detections, and draw `track.track_id` on each box.
- [ ] Add a sidebar toggle: "Detection only" vs "Detection + Tracking".

Do not modify `scripts/`, `src/detection/`, or `frontend/app.py` logic as
part of this exercise unless you've reached step 4 deliberately — the
detector and demo are considered done; tracking is additive.

## Suggested test (do this for whichever wrapper you build)

1. Grab or record a ~10-20s clip of a drone crossing the frame
   (Roboflow Universe test-split video, or your own footage).
2. Run detection + your tracker frame-by-frame.
3. Log `(frame_idx, track_id, bbox)` for every track.
4. Assert/verify by eye:
   - The drone gets exactly one `track_id` for the whole clip it's visible.
   - No ID switch mid-clip (the same physical drone suddenly reported under
     a new ID).
   - If it briefly leaves and re-enters frame: with ByteTrack, expect a
     **new** ID (motion-only trackers can't re-identify after a full
     disappearance); with DeepSORT, a well-tuned `max_age` may re-assign
     the **same** ID. This difference is the whole point of comparing them
     — write down what you observe.

## Files you'll likely add here

```
src/tracking/
├── __init__.py          # existing, empty
├── base_tracker.py       # existing, interface only — don't add logic here
├── README.md              # this file
├── iou.py                 # your IoU + cost matrix helpers (step 1)
├── bytetrack_tracker.py   # your ByteTrackWrapper (step 2)
└── deepsort_tracker.py    # your DeepSortWrapper (step 3)
```
