# Cambodian Sign Language (CBDSL) Recognition using ResNet50
 
My thesis on **Cambodian Sign Language (CBDSL)** recognition with ResNet50. The pilot phase validates the pipeline on ASL before transitioning to CBDSL.
 
## Project Status
- **Current:** Pilot testing on ASL to validate the pipeline
- **Next:** Adapt and retrain for CBDSL gestures
- **Framework:** TensorFlow 2.x (model subclassing)

## Experimental Methodology
Two approaches are compared:
1. **From scratch** — random weight initialization, learns sign features from zero
2. **Transfer learning** — fine-tuned ImageNet weights, better suited for limited CBDSL data

## Architecture (ResNet50 Stem)
Input images are resized to 224×224 before entering the residual blocks:
- **Conv1:** 7×7 filters, stride 2 → 112×112×64
- **MaxPool:** 3×3, stride 2 → 56×56×64
- **Residual stages:** 3–4–6–3 bottleneck blocks → final 2048-dim feature vector

## Setup
```bash
pip install tensorflow numpy matplotlib
```
