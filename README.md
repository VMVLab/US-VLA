<div align="center">

# US-VLA: An Ultrasound Vision-Language-Action Model for Embodied Abdominal Scanning

**Accepted to ACM MM 2026**

</div>

---

## Overview

**US-VLA** is one of the first **vision–language–action (VLA)** frameworks tailored for **automated abdominal ultrasound scanning**. It explicitly encodes clinical semantic goals and generates sequential probe-manipulation actions under **real-time ultrasound feedback**.

Unlike conventional reinforcement-learning or imitation-based ultrasound automation that relies on hand-crafted reward functions or low-level motion supervision, US-VLA augments a pre-trained vision–language model with a dedicated **ultrasound image encoder** and an **ultrasound-aware expert fusion module**. This injects task-relevant ultrasound semantics into the action-generation pathway, enabling closed-loop and standardized acquisition of clinically meaningful standard planes with improved stability and generalization across organs, scanning targets, and diverse clinical conditions.

To support this task, we further construct **US-VLA-Data**, a real-world dataset covering liver and kidney examinations with five clinically defined standard planes, comprising **320 expert scanning trajectories** and approximately **80,000 synchronized timesteps**.

---

## Framework

<div align="center">
  <img src="pipeline.png" alt="US-VLA Framework" width="100%"/>
</div>

The framework consists of three main components:

- **(1) Vision–Language Encoding.** RGB images from the wrist-mounted and side-view cameras are encoded by a **SigLIP** visual encoder, and clinical task instructions describing the target standard plane are tokenized and embedded by a language encoder. Ultrasound images are encoded separately by a universal **US foundation model (USFM)** to avoid the domain mismatch between natural and ultrasound images. A **PaliGemma** backbone aligns the visual and language streams into fused vision–language representations.

- **(2) Ultrasound-Aware Expert Fusion.** A cross-modal attention module injects real-time ultrasound feedback into the decision process: the vision–language features serve as **queries** while the ultrasound features serve as **keys/values**, followed by residual connections and a feed-forward expert block that produces ultrasound-modulated action features.

- **(3) Action Expert and Policy Head.** Conditioned on the fused representations and the robot state, an action expert and policy head map the features to **continuous, sequential probe-control commands** under closed-loop ultrasound guidance.
