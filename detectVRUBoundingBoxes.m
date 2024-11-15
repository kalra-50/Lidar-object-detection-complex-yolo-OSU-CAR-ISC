function [bboxCuboid, scores, labels] = detectVRUBoundingBoxes(ptCloud)
    % detectCarBoundingBoxes: Takes input as the merged PCD file and gives bounding boxes for car detections.
    % ptCloud: Point cloud object (input point cloud)
    model_big = true;
    %% Load the trained network
    persistent net;
    if isempty(net)
        %% Load the trained network once
        if model_big
        loadedData = load('/home/isc/ISC/matlab_models/Lidar-object-detection-using-complex-yolov4/models/trainedModel_epoch_20_VRU.mat');  
        else
        loadedData = load('/home/isc/ISC/matlab_models/Lidar-object-detection-using-complex-yolov4/models/trainedModel_epoch_90_tiny_VRU.mat');% change this after OSC loading is done.
        end
        net = loadedData.net;
    end

    %% Translate the original point cloud (to have positive Y values)
    ptCld = ptCloud;  % Create a copy to preserve the original point cloud.
    rotationAngles = [0 0 0];
    translation = [0 50 0];
    tform = rigidtform3d(rotationAngles, translation);
    ptCloudOut = pctransform(ptCld, tform);

    %% Get the configuration parameters
    gridParams = helper.getGridParameters;

    %% Set class names (only one class: Pedestrain)
    classNames = categorical("Pedestrain");

    %% Get bird's-eye-view RGB map from the point cloud
    [img, ptCldOut] = helper.preprocess(ptCloudOut, gridParams);

    %% Define anchor boxes for car detection

    if model_big
    anchors.anchorBoxes = [    4    4;
                               5    4;
                               5    7;
                               7    7;
                               5    10;
                               10    7;
                               7    11;
                               13    8;
                               10    12];
    anchors.anchorBoxMasks = {[1,2,3]; [4,5,6]; [7,8,9]};
    else 
        anchors.anchorBoxes = [    12    9;
                               8    11;
                               5    11;
                               7    6;
                               5    6;
                               4    4];
         anchors.anchorBoxMasks  = {[1,2,3];[4,5,6]};

    end 

    %% Perform detection on the bird's-eye-view image
    executionEnvironment = 'auto';
    [bboxes, scores, labels] = detectComplexYOLOv4(net, img, anchors, classNames, executionEnvironment);

    %% Handle case where no detection is found
    if isempty(bboxes)
        bboxCuboid = [];
        scores = [];
        labels = [];
        return;
    end

    %% Transfer bounding boxes to the point cloud
    bboxCuboid = transferbboxToPointCloud_VRU(bboxes, gridParams, ptCldOut);

function bboxCuboid = transferbboxToPointCloud_VRU(bboxes, gridParams, ptCldOut)
    % Transfer labels from bird's-eye-view images to the point cloud.
    
    % Calculate the height of the ground plane.
    groundPtsIdx = segmentGroundSMRF(ptCldOut, 3, 'MaxWindowRadius', 5, 'ElevationThreshold', 0.4, 'ElevationScale', 0.25);
    loc = ptCldOut.Location;
    groundHeight = mean(loc(groundPtsIdx, 3));
    
    % Assume height of objects to be a constant based on input data.
    objectHeight = 1.55;
    
    % Transfer labels back to the point cloud.
    bboxCuboid = zeros(size(bboxes, 1), 9);
    bboxCuboid(:, 1) = (bboxes(:, 2) - 1 - gridParams.bevHeight / 2) * gridParams.gridH;
    bboxCuboid(:, 2) = (bboxes(:, 1) - 1) * gridParams.gridW;
    bboxCuboid(:, 4) = bboxes(:, 4) * gridParams.gridH;
    bboxCuboid(:, 5) = bboxes(:, 3) * gridParams.gridW;
    bboxCuboid(:, 9) = -bboxes(:, 5);
    bboxCuboid(:,2)  = bboxCuboid(:,2) - 50;    % translating the y position back to 
    bboxCuboid(:, 6) = objectHeight;
    bboxCuboid(:, 3) = groundHeight + (objectHeight / 2);
end
end