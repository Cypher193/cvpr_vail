# Deviations from the pre-registered gates

Each entry: what changed, when, whether the data had been seen, and why. Thresholds in `gates.yaml` are not edited.

## D1 — G1a real-footage check scored on speeds only (2026-10-17, after seeing data)

- **Rule as registered:** symmetric MRA ≥ 0.80 against ground truth on the main renders and on QuantiPhy val.
- **Result:**
  - Renders: 0.947, passed.
  - QuantiPhy val (2D clips, 7 questions): 0.696, failed.
    - The 4 speed questions score 0.94–1.00 (mean ≈ 0.98).
    - The 3 "acceleration at time t" questions score 0.00–0.63.
- **Change:** the QuantiPhy part of G1a is scored on the speed questions only, and passes at ≈ 0.98.
- **Why:**
  - The local-acceleration questions ask for very small accelerations (0.34 m/s², 11 cm/s²) at a single instant. At that scale, 1–2 px of tracking noise dominates.
  - The pilot never reads accelerations that way. Gravity comes from a fit of the bouncing-ball law over the whole track, which recovers g at 0.99 on the renders.
  - The QuantiPhy acceleration questions therefore test a quantity the pilot does not use.
- **Approved by:** the user, after seeing the data, on 2026-10-17.
- **Also changed before scoring:**
  - The QuantiPhy clips first chosen (video type S3) move toward and away from the camera, and their labels are 3D speeds. They were replaced with 2D-motion clips (simulation_0163, internet_0027).
  - The tracker's metre scale now comes from min(mask width, height) instead of mask area.
  - Both changes fix measurement errors and do not alter any threshold.
