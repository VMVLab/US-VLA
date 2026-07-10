<div align="center">

# ForceU-VLA: A Force-Aware Vision–Language–Action Model for Embodied Ultrasound Scanning

<p>
  <a href="#"><img src="https://img.shields.io/badge/Conference-ACM%20MM%202026-1f6feb?style=flat-square" alt="ACM MM 2026"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/Project-Page-2ea44f?style=flat-square" alt="Project Page"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"></a>
</p>

**Official repository for the ACM MM 2026 paper<br/>
_"ForceU-VLA: A Force-Aware Vision–Language–Action Model for Embodied Ultrasound Scanning"_**

<!-- TODO: fill in the author list -->
_Author One<sup>1</sup>, Author Two<sup>1</sup>, Author Three<sup>2</sup>, ..._

<!-- TODO: fill in the affiliations -->
<sup>1</sup> Affiliation One &nbsp;&nbsp; <sup>2</sup> Affiliation Two

</div>

---

## 📖 Overview

**ForceU-VLA** is a **force-aware Vision–Language–Action (VLA)** model for **embodied ultrasound scanning**.
Given multi-view visual observations, a natural-language task instruction (e.g., *"Acquire the abdominal aorta sagittal (long-axis) view"*), real-time B-mode ultrasound images, and the robot's proprioceptive **contact-force state**, ForceU-VLA predicts a temporally consistent action sequence that drives a robotic arm to perform autonomous, closed-loop ultrasound acquisition.

> 📌 **Abstract.** _TODO: paste the final abstract of your paper here._

---

## 🧩 Framework

<div align="center">
  <img src="pipeline.png" alt="ForceU-VLA Pipeline" width="100%"/>
</div>

The framework consists of three tightly coupled stages:

- **(a) Vision–Language Encoding.** Multi-view RGB observations (wrist and side cameras) are encoded with **SigLIP** vision encoders, the language task instruction is embedded with a **Tokenizer**, and B-mode ultrasound frames are encoded with a domain-specific **Ultrasound Foundation Model (USFM)**.

- **(b) Ultrasound-Aware Expert Fusion.** A **PaliGemma** backbone fuses the vision and language tokens into *V–L fused features*. An ultrasound-aware cross-attention expert then injects sonographic evidence — **queries** are derived from the V–L features while **keys/values** come from the USFM tokens — yielding **V–L–US fused features** via multi-head attention, residual connections, and a gated feed-forward block.

- **(c) Action Expert and Policy Head.** Conditioned on the fused features together with the robot **state** (proprioception and contact force), a flow-matching **Action Expert** denoises sampled noise into action features. The **Action Head Project Layer** decodes these into an executable **action sequence** ($t_0, t_1, \dots$) for closed-loop scanning.

---

## ✨ Highlights

- 🩺 **Embodied ultrasound scanning** — a full VLA policy that autonomously acquires standard ultrasound views on a robotic platform.
- 🤖 **Force-aware control** — contact force is a first-class input, enabling safe and stable probe–skin interaction.
- 🔀 **Ultrasound-aware fusion** — a dedicated cross-attention expert grounds the vision–language representation in real-time sonographic feedback.
- 🎯 **Instruction-following** — language-conditioned acquisition of clinically meaningful target views.

---

## 🚀 Getting Started

> 🛠️ **Code, models, and dataset are coming soon.** Please ⭐ **star** and **watch** this repository to stay updated.

- [ ] Release the paper / arXiv preprint
- [ ] Release inference code and pretrained checkpoints
- [ ] Release training code
- [ ] Release the ultrasound scanning dataset

---

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{forceuvla2026,
  title     = {ForceU-VLA: A Force-Aware Vision--Language--Action Model for Embodied Ultrasound Scanning},
  author    = {TODO: Author One and Author Two and others},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

<!-- TODO: acknowledge funding, labs, and open-source projects your work builds upon. -->
_TODO: add acknowledgements here._

---

## 📧 Contact

For questions, please open an issue or contact <!-- TODO: your email --> `your-email@example.com`.
