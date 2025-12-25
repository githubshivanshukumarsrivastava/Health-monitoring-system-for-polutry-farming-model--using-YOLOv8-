This report is a record of the Exploratory Data Analysis (EDA), data prepares, clustering, and preliminary work with the aim of the model-based detection that was conducted concerning mortality monitoring in layer-house and open-cage poultry settings to authenticate the realization of analysis results and provide actionable visual details (Objective 1). The project parameters of Roboflow were used to obtain and optimize annotation, calculate dataset health metrics, create a variety of data visualizations (class balance, bounding-box statistics, sample annotation overlays), apply density and DBSCAN clustering to identify mortality hotspots and run a first-pass object-detection inference run with YOLOv8 (ultralytics) to simulate the automated detection of dead-birds using this interpretation. Main conclusions: the dataset consists of slightly more than 698 images (Roboflow export) of which local parsing identified 696 labelled images (1 unlabelled), and 1,139 bounding-box annotations categorized as either healthy chicken (791) or death chicken (348) - it is evident that there is no class balance. There are no really small bounding boxes found. Dead birds demonstrate spatial clusters and heatmaps in a number of images. Initial inference based on a lightweight YOLOv8 model was also made and the results are tentative and suggest that some images are being under-detected (more training or augmentation may be required). The report also ends with a recommendation of workable solutions of how to improve model performance, develop a real-time prototype, and convert insights into farm operations..


Introduction 
Quick mortality to be detected in the present-day poultry operations is important to control the spread of diseases with quick action to cut down on the labour expenses and guardian of animal rights. Objective 1 involves Justifying the analytic outputs and creating clear visualisations that can support on-farm decisions: (a) prepare and clean dataset on layer-house and open-cage operations; (b) image-recognition used to identify dead birds (a) and (b); (c) a restricted range of clustering used to enumerate mortality patterns (d); and (c) visualisation used to make quick-decision-supportive (map, heatmap, cluster overlay, etc.).

The present report is narrowly confined to Objective 1 - the EDA, label validation, clustering, and initial detection inference- and, is a report on the process, results, constraints, and subsequent recommendations to make the analytics as well as a subsequent prototype better.

Project scope & data used
Dataset: Roboflow Universe project new_chichken_dataset exported in YOLOv8 format. The dataset metadata from Roboflow indicates approximately 698 images. Local parsing via the notebook found:
•	total_images_with_labels: 696
•	images_without_labels: 1 (one unlabeled image flagged for review)
•	total_annotations: 1,139
•	per_class_counts: {'healthy_chicken': 791, 'death_chicken': 348}
•	tiny_boxes_count: 0 (no extremely tiny bounding boxes)
Files produced during EDA:
•	annotations_full.csv — full table of parsed annotations (image, bbox coords, class, normalized area, etc.)
•	dataset_summary.txt — text summary
•	flagged_tiny_boxes.csv — empty (no tiny boxes)
•	death_chicken_counts.csv — results of detection inference (per-image detected death counts)
•	Prediction images saved under runs/detect/predict/ (YOLOv8 default folder when save=True)
Tools & libraries: Python, pandas, NumPy, PIL/OpenCV, matplotlib, scikit-learn (DBSCAN), ultralytics YOLOv8 (for initial inference), Roboflow SDK (for download).
3. Forms - step-by-step 
In the diagram below I ontologically depict the major tasks of Objective 1 against the notebook cells and code blocks used in the analysis so that you can replicate and add that code as appendices.

3.1. Data ingestion and processing.

Purpose: verify the directory hierarchy, assert data.yaml, read images and YOLO .txt label file, transform normalised YOLO boxes into absolute positions.

Notebook mapping: Cell 2 (import, dataset detection), Cell 3 (helper functions), Cell 4 (parsing labels 8) annotations_table.csv).
Checks performed:
The folder arrangement is that of confirmation: train/ valid/ test/ (or train only test not there).
Read data.yaml to identify the class names and nc.
Make a file, image,img path,class,class name,xc,yc,w,h,xmin,ymin,xmax,ymax,imgw,imagemag,area,areanorm, aspec to make annotations-full.csv.
3.2. Dataset health checks 
Purpose: find data problems that will cause modeling to be poor (data label that is missing, small boxes too small, duplicates, out of bounds boxes).
Key outcomes:
tiny_boxes_count = 0 — acceptable.
images images not labeled = 1 - recheck and annotate the image, or delete it.
Duplicates scanned by hash of MD5 file- this time no large duplication problems were reported (at least not).
Cropping of images during parsing - few out of range points would be indicated.

3.3. Exploratory analysis of data (Eda)

Purpose: learn how to interpret the schedules of classes, densities of annotation, sizes of bound-boxes, the number of annotations per image and goes on to visualize annotation sample.

Notebook mapping Cell 5 (basic metrics), Cell 6 (visualization: class distribution), bbox area), Cell 7 (sample annotated images).

Visualizations created in the report:

Bar chart: classification (healthy chicken vs death chicken)

Histogram: remarks on each image.

Log scale histogram: distribution - normalized bounding-box area.

Aspect ratio distribution, area vs aspect scatter.

An array of sample images containing a bounding box of quick visual QA of health samples (healthy examples), and death examples.

3.4. Denstities and spatial clustering analysis.

Purpose: identify location groups of dead birds within an individual image and combine hotspots across images (assists in setting priorities on the camera perspective or fixed surveillance areas).

Notebook mapping: Cell 9 (DBSCAN (per image) heatmap), Cell 10 (heatmap from (per image) overlaying all cells), and an improved version of cell 10 (heatmap from (per image) enables the aggregation of centers across images).

Methods details:

Compute centers of bounding boxes, see above computing centres of images: cx = (xmin+xmax)/2, cy =(ymin+ymax)/2.

Run DBSCAN on centers (EPS is eps_ratio diag (image) with eps_ratio adjusted, min samples=2 to start with).

Cluster points Visualize clusters in the image preparing centers and naming cluster centroid.

Plot 2D hotspots in histograms of center densities/image and across images (log scale color map: on a visual scale).

3.5. Proof of concept detection Model-based inference (proof-of-concept detection)

Purpose: to show how death_chicken can be automatically detected with a lightweight YOLOv8 model and make actual predicted bounding boxes with which downstream cluster/alter actions can be performed.

Notebook mapping: the latter training/inference cells will be added later; the latter detecting and counting code will be written in the "predict + count" cell.

Implementation choices:
Stepped off with used ultralytics YOLOv8 which is yolov8n.pt.
Training: short run training smoke test (epochs= 2 in initial experiments) user set. In the production ready model, make the model have more epochs and more data augmentation.

Inference: results = model.predict(source=valid/images, save=True, conf=0.3) and predicted images are saved in runs/detect/predict/.

Counting: iterate through results and match each prediction with result.path to find the source image; number of death Singapore eggs detections by each image to death chicken counts.csv.

4. Findings (quantity, graphics as well as exposition)
4.1. statistics The statistics of the dataset (of parsing).
The image count being reported at Roboflow export was 698.
Local parsing produced:

696 labeled images

1 unlabeled image (flagged).

Annotations 1,139 interacting annotations all total.

Class separation: 791 healthy chicken, 348 death chicken.

There are no holes or boxes in the film that need to be removed.

Interpretation: The class imbalance is dominant: a significant number of instances of healthy and dead is 2.3-fold. This is critical since the object detectors do not offer normal special algorithms are likely to perform lower on the minority (death) category.

4.2. Quality/ distribution of annotation.

There were annotations on the individual image: number of chickens per image was zero to many, a few of the pictures with many looked like small clusters (hotspots to be identified).

BBox size: no very small bounding boxes, majority of the boxes are of a useable size. The area distribution (log-scale) of Bboxes showed the sizes were widely distributed - implies that the model needs to be resilient to scale changes.
Sample visual quality checks: the annotated images were inspected manually, revealing that the rectangular coverage of annotated chicken was reasonable; a few of the images are obstructed and have different intensities of light.

4.3. Clustering and hot-spoting

DBSCAN (per-image) determined clusters within images where many dead chickens were identified near each other (that could be localised mortality events).

Aggregated heatmap of all the death chicken centers gave valid hot spots in some areas on the image that mortalities happen frequently at those areas across frames practical implication apply camera or staff attention there instead.

4.4. There is the first YOLOv8 inference (proof-of-concept).

Initial inference on the validation images was done by a lightweight YOLOv8 model.

There were numerous cases of no detections in certain images and successful detections of healthy chicken in other images. This shows first model confidence is ambivalent -wise because the first generation model is often reasonable when the training time is short or the general model being used has not received enough fine-tuning on the datasets being a highly specialized one.

Generated death chicken counts.csv indicates per image counts as detected (useful to construct alerts).

Interpretation: The end-to-end (images to detections to cluster/heatmap) inference run already shows that the model can work end-to-end, although does not have a proper training/validation loop in place as needed to achieve operational quality, and requires the model to be analyzed by proper metrics (mAP, per-class precision/recall).

Visualization
5. Graphs to be added to the report (and the way to recreate them)

To include in the final report penetration strikes, slides, and figures, the following are included (they are all generated by a cell in a notebook - see mapping):
Summary table of datasets (Cell 11) - contains overall number of images, total amount of annotations, number of classes.
Normalized bbox area histogram (log scale
Sample annotated images grid display sample image of death chicken and healthy chicken.
Image stacked bar chart (healthy vs death) - is there more visualization to be had (after separate enhancement cell you ran).
Time-sequence plot of counts by image - in case the names of the files contain any sequence information (enhanced cell).
Per-image DBscan cluster overlay (Cell 9) - centers clustered have complete cluster IDs.

Spatial density heatmap (aggregate) Cell Heatmaps (Cell 10) - per-image and aggregate spatial density heatmaps.

The example of the inference output of YOLOv8 — the screenshots of the runs/detect/predict/ with the predictions displayed.

Limitations & data caveats
6. Limitations & data caveats

Disparity in the classes: There are lower death cases than healthy ones (348 vs 791). This is more than likely to bias the detection models to the majority class. Mitigations: death chicken should be specifically augmented, class weighted, oversampled, or more death images gathered.
Images: There were 698 images reported by Roboflow, local parsing had 696 images with labels and 1 image without a label. Minor errors may occur due to the encodings of the file names or the files have disappeared or been corrupt; ensure that all the original images in the original files are found.
Model training & evaluation incompleteness: The first YOLOv8 run was smoke test. Strong assessment (mAP50/95, per-class precision/recall, confusion matrix) must be obtained prior to deployment.
Metadata by time and place: In most of the images, the metadata has in the form of sheds/cameras are unreliable and the time and place are not known. To do actual on-farm monitoring and trend analysis make metadata available at the capture time (camera ID, shed ID, timestamp, GPS optional).

7. Recommendations (actionable, ordered)
Data-quality & EDA improvements
1.	Correct the unlabeled image and re-run parsing. Confirm total image count.
2.	Add metadata to future captures: camera/shed ID and timestamps — essential for Objective 2 (trend & pilot testing).
3.	Balance the dataset: gather additional death_chicken images from different sheds, lighting, and camera angles or apply careful augmentation (brightness variation, cropping, rotation, mosaic) focused on the minority class.
Modeling & evaluation
4.	Train a YOLOv8 model with a full training schedule:
o	Use yolov8n or yolov8s to start, then scale to larger models if needed.
o	Train for 50–200 epochs with appropriate augmentations.
o	Use validation set and compute per-class mAP, precision, recall.
o	Use cross-validation if dataset is small.
o	Example evaluation command: model.val(data="path/to/data.yaml") (ultralytics) to compute metrics.
5.	Monitor recall for death_chicken specifically — in mortality detection, recall (minimizing false negatives) is often more important than overall precision.
6.	Post-processing: tune confidence thresholds and non-maximum suppression (NMS) and add minimum-box-area filters to reduce false positives.
Clustering & dashboard
7.	Automate DBSCAN on detection outputs: apply DBSCAN to predicted centers in real-time and produce live heatmaps/cluster alerts for cameras with rising cluster counts.
8.	Real-time dashboard: build a simple Streamlit or Dash app to show live camera frames, detections, cluster overlays, and a per-camera alerting panel (e.g., “if more than X deaths in Y minutes → alert”).
Pilot & business analysis
9.	Pilot test in 1–2 sheds: instrument 2 cameras per shed for 1–2 months, collect ground-truth event logs, compare detection alerts to manual findings, and compute operational savings estimates (reduction in manual inspection time, time-to-removal of dead birds, disease containment metrics).
10.	Cost/ROI estimate: after pilot, quantify labour savings and reduced mortality propagation to support scaling decisions.

9. Conclusion
Objective 1 — dataset preparation, EDA, clustering, and initial detection inference — has been successfully executed to a stage where meaningful insights can be extracted. The analysis revealed dataset health (annotations, class imbalance), spatial patterns of mortality (clusters and heatmaps), and demonstrated the feasibility of automated detection using YOLOv8 as an enabling technology. Key immediate priorities are to (a) address class imbalance and finalize model training with robust evaluation, (b) ensure metadata capture for temporal & spatial analysis, and (c) proceed to prototype development and pilot testing (Objective 2). Implementing the recommendations will move the system from an analytical proof-of-concept to an actionable on-farm monitoring solution.

