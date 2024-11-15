%% This scripts converts the 3-d bounding box information into 2-d bounding box 
% Each lidar frame is converted from 3-d to 2-d Bird's eye view psuedo
% image
%% Create the Bird's Eye View Image from the point cloud data

outputFolder = '/home/isc/ISC/matlab_models/Lidar-object-detection-using-complex-yolov4/Output_Folder';
load("gt_labels_yolo.mat")
boxLabels = gt_labels(:,2:end);
lidarFolder = '/media/isc/Extreme Pro/merged_pcd_files/';
boxLabels = renamevars(boxLabels,["Passenger","VRU"],["Car","Pedestrain"]);
%%

% Get the configuration parameters.
gridParams = helper.getGridParameters();
% Get classnames of Pandaset dataset.
classNames = boxLabels.Properties.VariableNames;
numFiles = size(boxLabels,1);
processedLabels = cell(size(boxLabels));

%%

for i = 1:numFiles
    % Read the LiDAR data and transform it.
    lidarPath = fullfile(lidarFolder, sprintf('%06d.pcd', i));
    ptCld = pcread(lidarPath);

    rotationAngles = [0, 0, 0];
    translation = [0, 50, 0];
    tform = rigidtform3d(rotationAngles, translation);
    ptCloud = pctransform(ptCld, tform);

    % Get ground truth for the current file.
    groundTruth = boxLabels(i, :);

    % Process the point cloud into BEV image.
    [processedData, ~] = helper.preprocess(ptCloud, gridParams);
    % Loop through each class
    for ii = 1:numel(classNames)
        labels = groundTruth(1, classNames{ii}).Variables;
        % Ensure all labels are accessed if they are stored in a cell array.
        if iscell(labels)
            labels = vertcat(labels{:});
        end
        display(labels)
        if ~isempty(labels)
            % Get the label indices that are in the selected RoI.
            labelsIndices = labels(:, 1) - labels(:, 4) > gridParams.xMin ...
                          & labels(:, 1) + labels(:, 4) < gridParams.xMax ...
                          & labels(:, 2) - labels(:, 5) > gridParams.yMin ...
                          & labels(:, 2) + labels(:, 5) < gridParams.yMax ...
                          & labels(:, 4) > 0 ...
                          & labels(:, 5) > 0 ...
                          & labels(:, 6) > 0;
            labels = labels(labelsIndices, :);

            if ~isempty(labels)
                % Prepare the labels for BEV representation.
                labelsBEV = labels(:, [2, 1, 5, 4, 9]);
                labelsBEV(:, 5) = -labelsBEV(:, 5); % Adjust yaw

                labelsBEV(:, 1) = int32(floor(labelsBEV(:, 1) / gridParams.gridW)) + 1;
                labelsBEV(:, 2) = int32(floor(labelsBEV(:, 2) / gridParams.gridH) + gridParams.bevHeight / 2) + 1;

                labelsBEV(:, 3) = int32(floor(labelsBEV(:, 3) / gridParams.gridW)) + 1;
                labelsBEV(:, 4) = int32(floor(labelsBEV(:, 4) / gridParams.gridH)) + 1;

                % Accumulate labels for each frame and class.
                if isempty(processedLabels{i, ii})
                    processedLabels{i, ii} = labelsBEV;
                else
                    processedLabels{i, ii} = [processedLabels{i, ii}; labelsBEV];
                end
            end
        end
    end

    % Save the BEV image for this frame.
    writePath = fullfile(outputFolder, 'BEVImages');
    if ~isfolder(writePath)
        mkdir(writePath);
    end
    imgSavePath = fullfile(writePath, sprintf('%06d.jpg', i));
    imwrite(processedData, imgSavePath);
end

% Convert processedLabels to a table and assign class names.
processedLabels = cell2table(processedLabels);
numClasses = size(processedLabels, 2);
for j = 1:numClasses
    processedLabels.Properties.VariableNames{j} = classNames{j};
end

% Save the processed labels.
labelsSavePath = fullfile(outputFolder, 'Cuboids/BEVGroundTruthLabels.mat');
save(labelsSavePath, 'processedLabels');

%%
% creating a truck column which is empty
processedLabels.Truck = cell(height(processedLabels), 1);  

% Display the updated table to verify
disp(processedLabels);

%%
labelsSavePath = fullfile(outputFolder,'Cuboids/BEVGroundTruthLabels.mat');
save(labelsSavePath,'processedLabels');

%% verify the labels by plotting some of them back on images
% Define the output folder for labeled images
labeledImagesPath = fullfile(outputFolder, 'LabeledBEVImages');
if ~isfolder(labeledImagesPath)
    mkdir(labeledImagesPath);
end
% Define colors for different classes
classColors = {'red', 'yellow', 'blue'};  % Red for Pedestrian, Yellow for Car, Blue for Truck
lineWidth = 6;  % Thicker line width for better visibility
% Helper function to calculate rotated corners
rotateBox = @(w, h, yaw) [
    cos(yaw), -sin(yaw); 
    sin(yaw), cos(yaw)
] * [ -w/2,  w/2,  w/2, -w/2;
      -h/2, -h/2,  h/2,  h/2 ];

% Loop through all images
for i = 1:numFiles
    % Load the BEV image
    imgPath = fullfile(outputFolder, 'BEVImages', sprintf('%06d.jpg', i));
    bevImage = imread(imgPath);

    % Convert grayscale to RGB if needed
    if size(bevImage, 3) == 1
        bevImage = repmat(bevImage, 1, 1, 3);
    end

    % Iterate over the classes (Car, Pedestrian, Truck)
    for ii = 1:numClasses
        labelsCell = processedLabels{i, ii};  % Extract labels for the current image and class

        if ~isempty(labelsCell) && ~isempty(labelsCell{1})
            labels = labelsCell{1};  % Extract matrix from the cell

            % Plot each bounding box for the class
            for k = 1:size(labels, 1)
                % Extract label information
                x = labels(k, 1);  % Center X
                y = labels(k, 2);  % Center Y
                w = labels(k, 3);  % Width
                h = labels(k, 4);  % Height
                yaw = deg2rad(labels(k, 5));  % Convert yaw to radians

                % Get the rotated corner points
                corners = rotateBox(w, h, yaw) + [x; y];

                % Reshape corners into a polygon-compatible format
                polygonPoints = reshape(corners, 1, []);

                % Draw the rotated polygon on the image
                bevImage = insertShape(bevImage, 'Polygon', polygonPoints, ...
                    'LineWidth', lineWidth, 'Color', classColors{ii});
            end
        end
    end

    % Save the labeled image
    labeledImagePath = fullfile(labeledImagesPath, sprintf('%06d_labeled.jpg', i));
    imwrite(bevImage, labeledImagePath);

    fprintf('Saved labeled image for Frame: %d\n', i);  % Log progress
end
fprintf('All labeled images have been saved.\n');




