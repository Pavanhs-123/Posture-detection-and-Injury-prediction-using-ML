# Academic Viva Voce Defense Guide: Badminton Posture AI

**Project Title:** Badminton Posture Detection and Injury Risk Assessment Using Machine Learning  
**Candidate:** Pavan H S  
**Degree:** B.Tech Capstone Project (Computer Science & Engineering)  
**Core Reference Artifacts:**  
- Model: `movement_classifier_reviewed_161_rf.pkl` (Random Forest, 161 verified events)  
- Dataset: `dataset_v20261025_reviewed.csv` (161 rows, 6 player groups)  
- Results: 3-Fold Player-Grouped Cross-Validation (Macro-F1: 0.5760, Accuracy: 0.6211)  
- Risk Framework: Biomechanical Risk Rubric v0.3 (Provisional Observational Rubric)  
- Test Suite: 110 passed, 26 subtests passed across 12 test modules

---

## Section 1: Machine Learning & Modeling

### Q1: Why did you choose Random Forest instead of a Deep Learning sequence model like LSTM, GRU, or Spatio-Temporal Graph Convolutional Networks (ST-GCN)?
**Answer:**  
"We selected Random Forest based on three deliberate engineering and scientific principles:
1. **Dataset Scale & Overfitting:** Our human-verified training dataset contains 161 sequence events across six player groups. Deep sequence architectures like LSTMs or ST-GCNs require tens of thousands of parameters, making them highly susceptible to memorizing background, court textures, or athlete clothing when trained on small sample cohorts ($N=161$).
2. **Domain-Specific Feature Engineering:** By projecting 33 raw frame coordinates into 10 physiologically validated sequence statistics (minimum knee flexion, maximum torso lean, bilateral asymmetry, stance ratio), we compressed the input space into informative, physically grounded features that decision tree ensembles handle exceptionally well.
3. **Interpretability & Robustness:** In biomechanics, coaches and sports doctors require transparent decision criteria. Random Forest provides clear feature importance and is mathematically invariant to monotonic feature scaling, whereas deep recurrent models function as opaque black boxes."

### Q2: Why is your reported accuracy 62.1% and macro-F1 0.576? Many published action-recognition papers claim 90% or higher. Isn't your accuracy low?
**Answer:**  
"That is an essential question that highlights the distinction between real generalization and data leakage:
* **The Data Leakage Fallacy:** Many published papers achieve $>90\%$ accuracy by performing *random train/test splits* on video frames. In sports video, consecutive frames or multiple clips of the same player share identical clothing, court lighting, body mass, and camera perspective. The model simply memorizes the athlete rather than learning biomechanical movement dynamics.
* **Our Group-Stratified Validation:** We strictly enforced **Player-Grouped Cross-Validation (`GroupKFold`)**. In our 3-fold split, every test fold evaluated athletes who were *completely absent* from the training folds. 
* Achieving **62.1% accuracy and 0.576 macro-F1 on unseen human subjects** across three explosive movement classes represents an honest, realistic baseline for cross-subject transfer. On the same folds, an untrained kinematic heuristic scored only 32.9% accuracy and 0.354 macro-F1, proving that the model learned substantial cross-subject movement representations."

### Q3: Why does `SMASH_LANDING` achieve an F1 of 0.819, while `LUNGE` (0.500) and `CHANGE_OF_DIRECTION` (0.409) perform substantially worse?
**Answer:**  
"This performance differential reflects the underlying physical signatures in monocular lateral video:
1. **Smash Landing (F1 = 0.819):** Jump smashes exhibit a distinct multimodal kinematic signature: vertical hip displacement ($\ge 0.04$), high elevation, and an overhead arm extension, followed by bilateral landing impact. The model easily separates this from floor-level footwork.
2. **Lunge (F1 = 0.500, Recall = 0.455) & Change of Direction (F1 = 0.409):** Both actions occur at court level and involve significant knee flexion and lateral deceleration. In lateral 2D video, when an athlete performs an aggressive lateral cut, the braking plant leg often achieves joint angles ($100^\circ	ext{–}120^\circ$) very similar to a short forecourt lunge. Furthermore, during recovery, the transition footwork can blur the boundary. 
This is why our confusion matrix shows 20 lunges misclassified as COD and 12 CODs misclassified as lunges. This represents a genuine biomechanical limitation of monocular video that we transparently document."

### Q4: How did you address class imbalance in the training data?
**Answer:**  
"In our 161-event reviewed dataset, `SMASH_LANDING` accounts for 72 events (44.7%), `CHANGE_OF_DIRECTION` for 45 events (28.0%), and `LUNGE` for 44 events (27.3%). 
To address this:
1. We utilized class-weighted sample penalties (`class_weight='balanced'`) during Random Forest training to ensure minority classes are not penalized by tree split criteria.
2. We evaluated our model primarily using **Macro-Averaged F1-Score** (which treats all three classes equally regardless of sample size) rather than raw accuracy, which would be skewed toward the majority smash class.
3. In candidate manifest review, we actively prioritized reviewing underrepresented lunge and cut proposals from `player_01`, `player_02`, and `player_04` to prevent further data skew."

---

## Section 2: Computer Vision & Pose Tracking

### Q5: How does Google MediaPipe Pose extract 3D coordinates from a standard 2D video, and what are its physical limitations?
**Answer:**  
"MediaPipe Pose operates via a two-stage convolutional pipeline:
1. A lightweight detector network identifies the person bounding box ROI.
2. A landmark regression model predicts 33 3D skeletal landmarks.
* **The 3D Coordinate Mechanism:** The $x$ and $y$ coordinates represent normalized image coordinates $[0.0, 1.0]$ scaled by frame dimensions. The $z$ coordinate represents estimated *relative depth* (distance from the subject's hip center toward or away from the camera), predicted using anatomical perspective priors learned from large-scale synthetic and MoCap training corpora.
* **Limitations:** MediaPipe's $z$ depth is relative, not metric (i.e. not in true millimeters). Additionally, fast badminton racket swings exceed 300 km/h, which creates visual motion blur at standard 30–60 FPS shutter speeds, occasionally degrading wrist and racket-elbow tracking. This is why our model relies primarily on robust lower-body and trunk landmarks rather than rapid wrist tracking."

### Q6: What happens when an athlete lunges and their trailing arm or leg is occluded by their torso?
**Answer:**  
"Anatomical self-occlusion is an inherent challenge in monocular lateral video. We resolved this through three specific mechanisms:
1. **Pose Quality Decoupling:** Historically, when a trailing arm was occluded, overall pose confidence dropped to $\sim 0.12$, causing the pipeline to reject the entire event. We implemented `lower_body_quality`, which isolates hips, knees, and ankles. In our test lunge video, this showed lower-body tracking was rock-solid at $0.851$ despite the occluded elbow.
2. **Missing Feature Gate:** If an active classifier required an occluded feature (e.g. `right_elbow_min`), rather than fabricating a default value like $180^\circ$ or $0.0$, the pipeline's Quality Gate explicitly flags `Classifier unavailable: Required feature 'right_elbow_min' is missing or invalid` and preserves `UNCERTAIN`.
3. **Bilateral Asymmetry Tracking:** For risk evaluation, the rubric reports the minimum flexion observed on *either* visible knee rather than assuming a fixed left/right orientation."

### Q7: How does your keyframe selection algorithm (`_select_representative_frame`) work, and why did the original midpoint selection fail?
**Answer:**  
"In video event processing, selecting an auditable keyframe is critical for coaches to inspect posture.
* **Failure of Midpoint:** Originally, the pipeline extracted $(t_{	ext{start}} + t_{	ext{end}}) / 2$. However, athletic movements are non-symmetric: in a lunge, an athlete drops into maximum knee flexion within the first 35% of the window and spends the remaining 65% recovering. The midpoint captured the athlete standing up.
* **Our Solution:** `_select_representative_frame` filters candidate frames whose `lower_body_quality \ge 0.45` and computes a rolling median of knee flexion over a $\pm 0.12$-second window (at least 3 frames). It isolates the exact frame where knee flexion depth is maximal and temporally supported, rejecting single-frame noise spikes. On `test_lunge_movement1.mp4`, it isolated frame 25 at $1.00	ext{s}$ as the true flexion peak."

---

## Section 3: Biomechanical Injury-Risk Assessment

### Q8: Why did you separate Movement Classification from Injury-Risk Assessment?
**Answer:**  
"Because they represent two completely different scientific domains:
* **Movement Classification** answers: *What physical movement occurred?* (e.g. Was it a Lunge or a Smash Landing?). This is a pattern recognition problem that machine learning is uniquely suited to solve from kinematic signals.
* **Injury-Risk Assessment** answers: *How safely was that movement executed?* (e.g. Did the knee land stiffly? Was the torso leaning dangerously outside the base of support?). 
Conflating the two—such as treating 'HIGH RISK' as a movement class—corrupts the feature space. An athlete can perform an elite lunge with poor knee depth (high joint stress), or an elite lunge with perfect mechanics. Decoupling them allows the ML model to focus on movement taxonomy, while the biomechanical module evaluates execution safety."

### Q9: Why didn't you train a supervised machine learning model to predict LOW, MODERATE, and HIGH injury risk?
**Answer:**  
"This was a critical ethical and scientific design decision. 
In computer vision, true injury outcomes (e.g., whether an athlete suffered an ACL tear or meniscus damage following a match) require longitudinal clinical trials and medical records that are not available in public or match video datasets. 
If we had trained a classifier to predict 'HIGH RISK' based on heuristic labels that we invented ourselves, we would be training a model to mimic our own arbitrary assumptions, creating a false and dangerous illusion of clinical validity. 
Instead, we implemented **Rubric v0.3 as a deterministic, transparent observational rubric** grounded in published sports biomechanics literature, reporting factual joint angles without making automated clinical claims."

### Q10: What is Rubric v0.3, and why does it report "PROVISIONAL" observations?
**Answer:**  
"Rubric v0.3 is our rule-based biomechanical framework specifying observable posture indicators for our three active classes:
* For **Lunges**: Lead knee plant depth, knee asymmetry, and forward torso lean.
* For **Smash Landings**: Touchdown knee deceleration angle, bilateral impact asymmetry, and lateral trunk tilt.
* For **Changes of Direction**: Deceleration plant angle and lateral torso lean.
Under Rubric v0.3, numerical thresholds (e.g., exactly $110^\circ$ vs $115^\circ$) are intentionally classified as *unspecified research hypotheses* pending empirical sports science trials. Therefore, all assessable events report measured angles under the status **`PROVISIONAL (Unvalidated Cutoffs)`**, accompanied by an explicit engineering disclaimer."

### Q11: What triggers an `UNASSESSED` risk status?
**Answer:**  
"An event receives `UNASSESSED` under four deterministic conditions:
1. If the movement is classified as `UNCERTAIN` or belongs to a deferred category.
2. If landmark tracking quality is below our strict validation threshold of $0.45$.
3. If any required joint measurement is missing or non-finite.
4. If candidate detection windows lack continuous pose support.
This ensures the system never fabricates risk assessments on corrupted or ambiguous data."

---

## Section 4: System Engineering, Architecture & Ethics

### Q12: Why were `SCISSOR_JUMP` and `FOOTWORK_SPLIT_STEP` retired from active movement classes?
**Answer:**  
"Both classes were retired for clinical and engineering rigor:
1. **Scissor Jump:** In monocular lateral 2D video, an athlete performing a scissor kick displays leg-crossing dynamics that overlap heavily with rear-court jump smashes. Furthermore, sports biomechanics literature lacks standardized injury-risk boundaries specifically calibrated for scissor kicks.
2. **Split Step:** A split step is a low-impact preparatory plyometric bounce performed before court traversal. It carries negligible joint shear or ACL hazard compared to deep lunges or smash landings.
To preserve model precision and clinical focus, both classes were retired, and strict code gates were added to block retired classes from entering active datasets."

### Q13: What is the impact of your small athlete cohort ($N=6$ athletes) and `player_05` representing ~48% of the data?
**Answer:**  
"This represents a primary documented limitation of our project:
* **The Imbalance:** Our 161 verified events derive from 6 tournament match recordings. `player_05` participated in the longest match and contributed 78 events ($48.4\%$).
* **Mitigation:** In standard random splitting, this would cause severe overfitting to `player_05`'s individual posture habits. By enforcing **Player-Grouped Cross-Validation**, fold 3 evaluated `player_05` completely held out from training, forcing the model to learn features that generalize beyond individual athlete morphology.
* **Academic Takeaway:** Expanding the cohort to 20+ diverse players is the foremost priority for future work to balance subject representation."

### Q14: How does your system ensure reproducible testing?
**Answer:**  
"The codebase incorporates an automated pytest test suite comprising **110 unit and regression tests with 26 subtests**, executing in 5.15 seconds across 12 test modules. 
The suite validates:
- Coordinate-to-angle math and aspect ratio scaling (`test_pipeline.py`).
- Deterministic risk rubric gating and missing-feature handling (`test_risk_evaluator.py`).
- Player-grouped split isolation with zero subject leakage (`test_train_splitting.py`).
- Quality-gated keyframe selection and peak relocalization (`test_keyframe_selection.py`).
- Candidate review manifest backup integrity (`test_candidate_review.py`).
- Graceful error handling for missing, empty, or corrupted video files (`test_video_pipeline.py`)."

### Q15: How does the system handle an upload of an unsupported file format or a corrupted video stream?
**Answer:**  
"The system uses multi-layered input validation:
1. Streamlit's file uploader enforces allowed video MIME types (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Our backend validator (`validate_video`) inspects the video container using OpenCV: checking file existence, non-zero file size, readable frame headers, valid width/height ($>0$), and positive frame rate ($>0$).
3. If an empty or corrupt `.mp4` is uploaded, OpenCV flags `moov atom not found` or decode failure, and the UI displays a clean error banner: `❌ Invalid Video: Failed to open video file. File may be corrupted or codec unsupported.` The server does not crash."

### Q16: Why did you build the dashboard in Streamlit instead of React/Django or Flask?
**Answer:**  
"Streamlit was chosen because it allows rapid, reactive data science visualization with native Python integration:
1. Direct integration with OpenCV, NumPy, MediaPipe, and Scikit-Learn without serialization overhead.
2. Built-in interactive components (progress bars, video players, probability bar charts, metric cards).
3. Session state isolation for upload keys, preventing prior video results from lingering when a new video is uploaded.
4. Seamless deployment and headless automated testing support via `streamlit.testing.v1.AppTest`."

### Q17: What are the top three technical priorities if you had an additional six months to extend this project?
**Answer:**  
"If continuing this research into graduate work, the top three priorities would be:
1. **Multi-Camera Synchronized Capture:** Deploy a synchronized two-camera setup (90° lateral court-side + 0° rear baseline broadcast view) to eliminate self-occlusion during contralateral lunges and enable true 3D joint angle calculation.
2. **Cohort Expansion:** Partner with collegiate badminton athletic teams to record 50+ diverse male and female athletes across varying skill levels, resolving the current 6-player cohort limitation.
3. **Clinical Collaboration & Markerless Validation:** Conduct a controlled validation trial comparing our MediaPipe-extracted angles against a laboratory Vicon 3D motion capture system, establishing empirical clinical cutoff thresholds for Rubric v0.3."

---
