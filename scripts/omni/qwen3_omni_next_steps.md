# Qwen3-Omni Exploration Notes

This directory separates the concrete Qwen3-Omni plan into two layers.

1. Native Qwen3-Omni inputs: RGB/fisheye video, embedded audio, and text prompts.
2. Xperience-10M sensor adapter inputs: depth, pose/SLAM, mocap, contacts, and IMU.

`qwen3_omni_adapter_smoke.py` validates the second layer first using real
episode windows and real labels. It does not fabricate Qwen outputs and does not
claim Qwen3-Omni was fine-tuned. Once multiple episodes are available and the
storage budget is clear, the next step is to attach the saved adapter tokens to
Qwen3-Omni through LoRA or a cross-attention memory bridge.

Suggested progression:

1. Run the adapter smoke test on one public sample episode.
2. Add a small manifest of additional episodes, capped by storage.
3. Hold out whole episodes for evaluation.
4. Load Qwen3-Omni processor/model on H20 only after data flow is verified.
5. Train sensor adapters first, then LoRA selected Qwen3-Omni layers.
