# Qwen3-Omni Adapter Smoke Test

- Base model target: `Qwen/Qwen3-Omni-30B-A3B-Thinking`
- Qwen3-Omni weights loaded: `false`
- Episodes: `1`
- Windows: `59` total, `41` train, `18` test
- Split: `single_episode_chronological`
- Feature dimension: `4262`
- Adapter soft-token blocks: `11`
- Accuracy: `0.0000`
- Macro-F1: `0.0000`

## Why this is the minimum real test

This run uses real Ropedia annotation/video-derived feature blocks. It tests the sensor-adapter side that depth, pose, mocap, contacts, and IMU need before those tokens are attached to Qwen3-Omni. It deliberately avoids downloading the 30B Qwen3-Omni weights until the data path, labels, splits, and storage plan are confirmed.
