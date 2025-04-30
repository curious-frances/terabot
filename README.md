# 🐾 TeraBot: Real-Time Edge Deployment of Vision-Language Reinforcement Policies on a Quadruped Robot

This repository supports our project, **TeraBot**, which investigates the deployment of vision-language reinforcement learning policies on a low-cost, embedded quadruped robotic platform for assistive navigation tasks.

---

## 🧭 Motivation

Visually impaired individuals face daily challenges in navigating unfamiliar environments, often relying on guide dogs or human assistance. However, traditional guide dogs are costly, limited in availability, and cannot interpret complex instructions or adapt to diverse, dynamic environments.

TeraBot seeks to address these limitations by offering an autonomous, vision-based robotic guide that combines real-time perception, decision-making, and locomotion — all executed onboard using a Raspberry Pi 5. The ultimate goal is to provide a scalable, cost-effective alternative to traditional assistive technologies for navigation in semi-structured environments like campuses or hospitals.

---

## 🎯 Objective

Our system integrates:

- A **deep reinforcement learning policy** (based on RL2AC) for quadruped locomotion.
- A **lightweight object detection model** (YOLOv11) for visual perception.
- **Onboard real-time inference**, made possible through model quantization and NCNN conversion, running entirely on edge hardware.
- Multimodal integration of **vision and action** to enable autonomous navigation and obstacle avoidance.

This allows the robot to respond intelligently to visual cues (e.g., stop signs, hand gestures) and execute context-aware navigation without the need for cloud computation or external sensors.

---

## 🧱 Project Structure
