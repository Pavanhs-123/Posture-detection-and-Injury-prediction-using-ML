# Biomechanical Injury-Risk Rubric v0.3

## 1. Overview and Scope

This document specifies **Biomechanical Risk Rubric v0.3** for the Badminton Posture Detection & Injury-Risk Assessment System, including the deterministic numerical scoring extension.

### Fundamental Principles
1. **Movement Classification $
eq$ Injury Risk**: Movement classification determines *what* physical action took place (e.g., a lunge or a jump landing). Injury risk quantifies *how* that action was executed biomechanically based on joint angles, alignment, and impact mechanics.
2. **Active vs. Deferred Movement Categories**:
   * **Active Categories for Observation and Scoring**:
     * `LUNGE`
     * `SMASH_LANDING`
     * `CHANGE_OF_DIRECTION`
   * **Deferred Categories**:
     * `SCISSOR_JUMP` (Risk scoring deferred; joint risk remains unassessed)
     * `FOOTWORK_SPLIT_STEP` (Risk scoring deferred; preparatory bounce step)
   * The deferred categories are valid movement classification targets, but their injury-risk criteria are not scientifically calibrated in Rubric v0.3. They must **never** be automatically assigned a numerical score or risk category.
3. **Provisional Engineering Benchmark Status**:
   * Under Rubric v0.3, numerical scores and category cutoffs are **provisional engineering screening benchmarks** derived from published sports medicine literature (e.g., Koga et al., Hewett et al., Boden et al.).
   * They represent kinematic deviations from reference posture envelopes. They are **NOT** clinically validated injury outcome predictions or medical diagnoses.
   * Active movements are evaluated as **`PROVISIONAL`** observations reporting both factual joint angles and a deterministic 0–100 screening score.
4. **Academic Engineering Prototype Disclaimer**:
   This rubric provides research-oriented biomechanical screening indicators based on published sports medicine literature. It is **NOT** a certified medical diagnostic tool or clinical injury predictor. A score of 70 denotes elevated postural deviation, **NOT** a 70% probability of suffering an injury.

---

## 2. Active Movement Biomechanical Indicators

Under Rubric v0.3, the following anatomical indicators are extracted and reported as factual observations across the three active movement categories:

### 2.1 Lunge (`LUNGE`)
Focus: Patellofemoral joint stress, anterior cruciate ligament (ACL) shear loading, and lumbar spine posture during the lunging plant phase.
* **Lead Knee Flexion Angle ($	heta_{	ext{knee}}$)**: Measured at peak plant depth.
* **Lead vs. Trail Knee Asymmetry**: Measured differential between legs.
* **Torso Forward Lean ($	heta_{	ext{trunk}}$)**: Measured inclination relative to vertical plane.
* **Status**: `PROVISIONAL (Unvalidated Cutoffs)`.

### 2.2 Overhead Smash Landing (`SMASH_LANDING`)
Focus: Ground reaction force absorption, bilateral impact distribution, and dynamic knee cushioning following an overhead jump.
* **Touchdown Knee Deceleration Flexion**: Knee angle at landing contact.
* **Bilateral Knee Landing Asymmetry ($\|	heta_L - 	heta_R\|$))**: Difference in impact absorption between left and right limbs.
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
* **Reporting Guideline**: Classified as `SCISSOR_JUMP` for movement category; risk assessment must be displayed as `UNASSESSED (Rubric v0.3 Deferred)` with raw kinematic observations only. Numerical score is `None`.

### 3.2 Footwork Split-Step (`FOOTWORK_SPLIT_STEP`)
* **Movement Definition**: Preparatory pre-hop or split-step to lower center of mass and pre-stretch leg musculature before reacting to shuttlecock.
* **Risk Scoring Status**: **DEFERRED** in Rubric v0.3.
* **Reporting Guideline**: Classified as `FOOTWORK_SPLIT_STEP` for movement category; risk assessment must be displayed as `UNASSESSED (Rubric v0.3 Deferred)`. Numerical score is `None`.

---

## 4. Assessment Handling Rules & Quality Gates

1. **Uncertain or Low Quality Posture**: If key lower-limb and trunk landmarks exhibit tracking confidence $< 0.45$, the risk assessment status must be marked as `UNASSESSED (Low Landmark Quality)`.
2. **Missing Measurements Gate**: If required kinematic measurements (`left_knee_min`, `right_knee_min`, `trunk_angle_max`) are missing or non-finite, the assessment is marked `UNASSESSED (Incomplete Kinematics)`. Missing data must **NEVER** default to `LOW` risk or substitute a score of 0.
3. **No Clinical Over-Claims**: Do not present the numerical score as a clinical outcome forecast.
4. **Camera Viewpoint Sensitivity**: Lateral 90° view is required for planar sagittal joint kinematics. Non-perpendicular angles suffer perspective foreshortening.

### 4.1 Temporal Conflict Resolution (Segmentation Policy)

When detected intervals overlap, the current segmentation policy gives `LUNGE` and `SMASH_LANDING` precedence over broader footwork events. The footwork interval is trimmed around a target event only when the remaining segment satisfies its minimum duration; otherwise, the overlap is marked `UNCERTAIN`. Competing target events are split only when they form a sequential boundary overlap and both resulting intervals satisfy their minimum durations. Other unresolved target conflicts remain `UNCERTAIN`. Exact duplicate target events with competing labels also remain `UNCERTAIN`; a target event takes precedence over a non-target duplicate while retaining its evidence. In this resolver, `CHANGE_OF_DIRECTION` follows the non-target branch.

This precedence is an engineering choice for temporal segmentation and label conflict handling. It is **not** a proven clinical hierarchy and does not establish that one movement is more injurious or biomechanically important than another.

---

## 5. Numerical Biomechanical Screening Score (0–100)

To provide an interpretable, standardized posture quality index across assessable movements, Rubric v0.3 incorporates a deterministic, bounded 0–100 screening score.

### 5.1 Mathematical Formulation
The total score $S \in [0.0, 100.0]$ is the sum of three independent, non-overlapping kinematic indicator contributions:

$$S = \min\left(100.0, \max\left(0.0, P_{	ext{knee}} + P_{	ext{trunk}} + P_{	ext{asym}}
ight)
ight)$$

Where:
* $P_{	ext{knee}} \in [0.0, 40.0]$: Knee Deceleration Flexion Contribution (Max 40 pts)
* $P_{	ext{trunk}} \in [0.0, 35.0]$: Torso Lean Deviation Contribution (Max 35 pts)
* $P_{	ext{asym}} \in [0.0, 25.0]$: Bilateral Knee Asymmetry Contribution (Max 25 pts)

$$\sum 	ext{Max Points} = 40.0 + 35.0 + 25.0 = 100.0	ext{ pts}$$

### 5.2 Indicator Weight Normalization & Directions of Concern

#### 1. Knee Deceleration Flexion ($P_{	ext{knee}}$, Max 40.0 pts)
* **Measurement**: Minimum side-specific knee angle $	heta_{	ext{knee}} = \min(	heta_{	ext{left}}, 	heta_{	ext{right}})$.
* **Physiological Context**: Ground reaction forces in badminton deceleration exceed 2.5–3.0× body weight. Optimal knee cushioning occurs in $[95.0^\circ, 115.0^\circ]$.
  * **Stiff Knee Concern ($	heta_{	ext{knee}} > 115.0^\circ$)**: Extended knee landing transfers ground shock directly to the ACL and passive knee structures.
  * **Acute Hyperflexion Concern ($	heta_{	ext{knee}} < 95.0^\circ$)**: Excessive knee collapse creates extreme patellofemoral compressive stresses.
* **Piecewise Formula**:
  $$P_{	ext{knee}} = \begin{cases}
  0.0, & 95.0^\circ \le 	heta_{	ext{knee}} \le 115.0^\circ \\
  \min\left(40.0, (	heta_{	ext{knee}} - 115.0) 	imes rac{40.0}{45.0}
ight), & 	heta_{	ext{knee}} > 115.0^\circ \\
  \min\left(40.0, (95.0 - 	heta_{	ext{knee}}) 	imes rac{40.0}{35.0}
ight), & 	heta_{	ext{knee}} < 95.0^\circ
  \end{cases}$$

#### 2. Torso Lean Deviation ($P_{	ext{trunk}}$, Max 35.0 pts)
* **Measurement**: Maximum image-plane torso angle relative to vertical $	heta_{	ext{trunk}}$.
* **Physiological Context**: Upright to moderate athletic posture ($\le 15.0^\circ$) keeps center of mass within base of support. Torso tilt $>15.0^\circ$ creates lumbar shear and induces dynamic knee valgus collapse.
* **Linear Formula**:
  $$P_{	ext{trunk}} = \begin{cases}
  0.0, & 	heta_{	ext{trunk}} \le 15.0^\circ \\
  \min\left(35.0, (	heta_{	ext{trunk}} - 15.0) 	imes rac{35.0}{30.0}
ight), & 	heta_{	ext{trunk}} > 15.0^\circ
  \end{cases}$$
  *(Reaches maximum 35.0 points at $	heta_{	ext{trunk}} \ge 45.0^\circ$)*

#### 3. Bilateral Knee Asymmetry ($P_{	ext{asym}}$, Max 25.0 pts)
* **Measurement**: Absolute difference between left and right event minima $\|	heta_{	ext{left}} - 	heta_{	ext{right}}\|$.
* **Physiological Context**: Symmetric bilateral cushioning ($\le 10.0^\circ$) distributes impact evenly. Large asymmetry ($>10.0^\circ$) indicates unilateral impact overloading or unstable trail leg drag.
* **Linear Formula**:
  $$P_{	ext{asym}} = \begin{cases}
  0.0, & \|	heta_L - 	heta_R\| \le 10.0^\circ \\
  \min\left(25.0, (\|	heta_L - 	heta_R\| - 10.0) 	imes rac{25.0}{35.0}
ight), & \|	heta_L - 	heta_R\| > 10.0^\circ
  \end{cases}$$
  *(Reaches maximum 25.0 points at $\|	heta_L - 	heta_R\| \ge 45.0^\circ$)*

### 5.3 Provisional Engineering Screening Bands

The total score $S$ maps to three provisional screening categories:

| Score Band | Category Label | Badge Color | Biomechanical Interpretation |
| :---: | :---: | :---: | :--- |
| **0.0 – 29.9** | **`LOW`** | 🟢 `#28a745` | Joint kinematics and alignment remain within recommended athletic cushioning envelopes. Minimal mechanical concern. |
| **30.0 – 59.9** | **`MODERATE`** | 🟡 `#ffc107` | Noticeable postural deviation (e.g. moderate knee stiffness, pronounced trunk lean, or unbalanced asymmetry). Cushioning compromised. |
| **60.0 – 100.0** | **`HIGH`** | 🔴 `#dc3545` | Substantial mechanical concern (stiff landing $>140^\circ$, severe trunk deviation $>35^\circ$, or extreme asymmetry $>35^\circ$). High joint stress. |
| **N/A** | **`UNASSESSED`** | ⚪ `#6c757d` | Movement is uncertain, deferred, or visual evidence is insufficient ($Q < 0.45$). Zero risk is **never** substituted. |

---

## 6. Worked Examples

### Example 1: Overhead Smash Landing (`test_smash_movement1.mp4`)
* **Kinematics**: $	heta_{	ext{min knee}} = 102.1^\circ$, $	heta_{	ext{trunk max}} = 14.7^\circ$, $\|	heta_L - 	heta_R\| = 27.7^\circ$.
* **Calculations**:
  * $P_{	ext{knee}} = 0.0	ext{ pts}$ ($102.1^\circ \in [95^\circ, 115^\circ]$ optimal cushioning).
  * $P_{	ext{trunk}} = 0.0	ext{ pts}$ ($14.7^\circ \le 15.0^\circ$ upright alignment).
  * $P_{	ext{asym}} = (27.7 - 10.0) 	imes (25.0 / 35.0) = 17.7 	imes 0.714 = 12.6	ext{ pts}$.
* **Total Score**: $0.0 + 0.0 + 12.6 = \mathbf{12.6 / 100}$.
* **Screening Result**: **`LOW RISK`** (`PROVISIONAL`, $12.6 / 100$).

### Example 2: Forward Net Lunge (`cand_006_LUNGE`)
* **Kinematics**: $\theta_{\text{min knee}} = 102.5^\circ$, $\theta_{\text{trunk max}} = 36.4^\circ$, $\|\theta_L - \theta_R\| = 8.3^\circ$.
* **Calculations**:
  * $P_{\text{knee}} = 0.0\text{ pts}$ ($102.5^\circ \in [95^\circ, 115^\circ]$ deep plant).
  * $P_{\text{trunk}} = (36.4 - 15.0) \times (35.0 / 30.0) = 21.4 \times 1.1667 = 24.97 \approx 25.0\text{ pts}$.
  * $P_{\text{asym}} = 0.0\text{ pts}$ ($8.3^\circ \le 10.0^\circ$ balanced).
* **Total Score**: $0.0 + 25.0 + 0.0 = \mathbf{25.0 / 100}$.
* **Screening Result**: **`LOW RISK`** (`PROVISIONAL`, $25.0 / 100$).

### Example 3: Ambiguous Candidate (`cand_014_UNCERTAIN`)
* **Movement Category**: `UNCERTAIN`.
* **Total Score**: `None` (Displayed as `N/A`).
* **Screening Result**: **`UNASSESSED`** (Reason: Movement classification is uncertain; zero risk is never assumed).\n