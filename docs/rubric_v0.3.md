# Biomechanical Injury-Risk Rubric v0.3

## 1. Overview and Scope

This document specifies **Biomechanical Risk Rubric v0.3** for the Badminton Posture Detection & Injury-Risk Assessment System.

### Fundamental Principles
1. **Movement Classification $\neq$ Injury Risk**: Movement classification determines *what* physical action took place (e.g., a lunge or a jump landing). Injury risk quantifies *how* that action was executed biomechanically based on joint angles, alignment, and impact mechanics.
2. **Active vs. Deferred Movement Categories**:
   * **Active Categories for Observation**:
     * `LUNGE`
     * `SMASH_LANDING`
     * `CHANGE_OF_DIRECTION`
   * **Deferred Categories**:
     * `SCISSOR_JUMP` (Risk scoring deferred; joint risk remains unassessed)
     * `FOOTWORK_SPLIT_STEP` (Risk scoring deferred; preparatory bounce step)
   * The deferred categories are valid movement classification targets, but their injury-risk criteria are not scientifically calibrated in Rubric v0.3. They must **never** be automatically assigned LOW, MODERATE, or HIGH risk.
3. **Unvalidated Cutoffs & H1 Cluster Hypothesis**:
   * Under Rubric v0.3, the H1 cluster-scoring hypothesis is an **unvalidated research hypothesis**.
   * Numerical indicator cutoffs (degrees of flexion, lean angles, asymmetry thresholds) are **unspecified** in Rubric v0.3 pending empirical biomechanical trials.
   * Automated classification into discrete `LOW`, `MODERATE`, or `HIGH` risk categories is **strictly disabled**.
   * Active movements are evaluated as **`PROVISIONAL`** observations reporting measured joint angles without clinical diagnostic claims.
4. **Academic Engineering Prototype Disclaimer**:
   This rubric provides research-oriented biomechanical screening indicators based on published sports medicine literature. It is **NOT** a certified medical diagnostic tool or clinical injury predictor.

---

## 2. Active Movement Biomechanical Indicators

Under Rubric v0.3, the following anatomical indicators are extracted and reported as factual observations. Numerical thresholds are intentionally unspecified until clinical trials validate specific cutoff boundaries.

### 2.1 Lunge (`LUNGE`)
Focus: Patellofemoral joint stress, anterior cruciate ligament (ACL) shear loading, and lumbar spine posture during the lunging plant phase.
* **Lead Knee Flexion Angle ($\theta_{\text{knee}}$)**: Measured at peak plant depth.
* **Lead vs. Trail Knee Asymmetry**: Measured differential between legs.
* **Torso Forward Lean ($\theta_{\text{trunk}}$)**: Measured inclination relative to vertical plane.
* **Status**: `PROVISIONAL (Unvalidated Cutoffs)`.

### 2.2 Overhead Smash Landing (`SMASH_LANDING`)
Focus: Ground reaction force absorption, bilateral impact distribution, and dynamic knee cushioning following an overhead jump.
* **Touchdown Knee Deceleration Flexion**: Knee angle at landing contact.
* **Bilateral Knee Landing Asymmetry ($\|\theta_L - \theta_R\|$))**: Difference in impact absorption between left and right limbs.
* **Lateral Trunk Tilt**: Core deviation from vertical alignment upon impact.
* **Status**: `PROVISIONAL (Unvalidated Cutoffs)`.

### 2.3 Change of Direction (`CHANGE_OF_DIRECTION`)
Focus: Deceleration mechanics, knee braking angles, and lateral trunk stability during aggressive court changes.
* **Braking Plant Knee Angle**: Knee flexion angle during maximum deceleration.
* **Lateral Trunk Deviation**: Lateral tilt of torso outside the base of support.
* **Status**: `PROVISIONAL (Unvalidated Cutoffs)`.

---

## 3. Deferred Movement Categories

### 3.1 Scissor Jump (`SCISSOR_JUMP`)
* **Movement Definition**: Aerial leg swap during overhead rear-court shot to rapidly regain forward momentum.
* **Risk Scoring Status**: **DEFERRED** in Rubric v0.3.
* **Reporting Guideline**: Classified as `SCISSOR_JUMP` for movement category; risk assessment must be displayed as `UNASSESSED (Rubric v0.3 Deferred)` with raw kinematic observations only.

### 3.2 Footwork Split-Step (`FOOTWORK_SPLIT_STEP`)
* **Movement Definition**: Preparatory pre-hop or split-step to lower center of mass and pre-stretch leg musculature before reacting to shuttlecock.
* **Risk Scoring Status**: **DEFERRED** in Rubric v0.3.
* **Reporting Guideline**: Classified as `FOOTWORK_SPLIT_STEP` for movement category; risk assessment must be displayed as `UNASSESSED (Rubric v0.3 Deferred)`.

---

## 4. Assessment Handling Rules

1. **Uncertain or Low Quality Posture**: If key joint landmarks exhibit visibility $< 0.45$, the risk assessment status must be marked as `UNCERTAIN (Low Landmark Quality)`.
2. **Missing Measurements Gate**: If required kinematic measurements are missing, the assessment is marked `UNCERTAIN (Missing Kinematic Measurements)`. Missing data must **NEVER** default to `LOW` risk.
3. **No Fabricated Cutoffs**: Do not invent numerical thresholds for unvalidated clusters.
4. **Camera Viewpoint Sensitivity**: Lateral 90° view is required for planar sagittal joint kinematics. Non-perpendicular angles suffer perspective foreshortening.
