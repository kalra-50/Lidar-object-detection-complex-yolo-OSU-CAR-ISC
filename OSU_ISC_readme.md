# Lidar Object Detection, Tracking, and Path Prediction with Complex YOLO

This project focuses on detecting, tracking, and predicting paths for road users in a lidar point cloud dataset. The methodology involves creating pseudo-2D training data from 3D lidar point clouds, training a Complex-YOLOv4 model, and implementing tracking and prediction algorithms.

## Clone the Repository
The code can be cloned from the repository:  
[https://github.com/kalra-50/Lidar-object-detection-complex-yolo-OSU-CAR-ISC.git](https://github.com/kalra-50/Lidar-object-detection-complex-yolo-OSU-CAR-ISC.git)

---

## Scripts Overview

### 1. **`createTrainingData`**
This script generates 2D training data using 3D bounding boxes and lidar point cloud data. It also produces pseudo-2D images and superimposes bounding boxes on them for verification.

#### Inputs:
- **Point Cloud Data (PCD):** 3D bounding boxes stored in a timetable format (time column can be filled with `NaT`).
- **Grid Parameters:** Defined by `helper.getGridParameters()`, which specifies:
  - Range for the point cloud
  - Pseudo-image resolution
  - Grid refinement resolution

#### Processing:
- The function calls `helper.preprocess()` to convert the lidar point cloud into a Bird’s Eye View (BEV) image.
  - **BEV Image Channels:**
    - R: Density Map
    - G: Height Map
    - B: Intensity Map
- Outputs:
  - BEV pseudo-images
  - Labels saved as 2D bounding boxes in `Cuboids/BEVGroundTruthLabels.mat`

---

### 2. **`complexYOLOv4TransferLearn.m`**
This script trains the Complex-YOLOv4 model using preprocessed BEV data.

#### Pretrained Model Options:
- `complex-yolov4-pandaset`
- `tiny-complex-yolov4-pandaset`
- User-provided pretrained model

#### Training:
- BEV images and bounding box labels are loaded.
- Training is performed separately for VRU (Vulnerable Road Users) and car classes.
  - Modify labels for class-specific training:
    ```matlab
    % 'processedLabels' is the name of the table
    processedLabels.Pedestrian = cell(height(processedLabels), 1);

    % Display the updated table to verify
    disp(processedLabels);
    ```

---

### 3. **`detection_tracking_prediction_dump.m`**
This script performs detection, tracking, and path prediction for road users.

#### Workflow:
1. **Detection:** 
   - Calls separate trained networks for VRU and car classes.
   - Functions:
     - `detectVRUBoundingBoxes` (VRU)
     - `detectCarBoundingBoxes` (Car)
   - Outputs 3D bounding boxes.

2. **Tracking:**
   - Bounding boxes are passed to the tracker. Different trackers were evaluated; the **JDPA tracker** was selected as the default.
   - Tracking is performed separately for each class (VRU and Car).
   - Custom visualization functions are available for debugging.

3. **Path Prediction:**
   - Outputs from the Kalman filter are used to predict trajectories for the next 5 seconds for each road user in the intersection Region of Interest (ROI).

---

## Notes
- The tracking and prediction processes are computationally intensive. Visualization of tracks can slow down processing but is useful for debugging.
- Users can experiment with custom trackers or other prediction algorithms.

---
