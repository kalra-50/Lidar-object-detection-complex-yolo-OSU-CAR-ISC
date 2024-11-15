% final script 
%Main script to process LiDAR folder and visualize detections
folderPath = "/media/isc/Extreme Pro/Submission_Data/Test Data 1/";
historicalYawDataPath = '/home/isc/Downloads/Pass_Vehicle_xyz_zrot.csv';
historicalYawData = readmatrix(historicalYawDataPath);

% Process point cloud data and detect objects
runNumber = 351;
% [all_car_detections, all_vru_detections] = processPcdFolder(folderPath, runNumber, historicalYawData);

% Load tracking data once
trackingFolderPath = fullfile(folderPath, ['Run_', num2str(runNumber)], 'tracking');
carData = load(fullfile(trackingFolderPath, 'car_detections.mat'), 'all_car_tracks');
vruData = load(fullfile(trackingFolderPath, 'vru_detections.mat'), 'all_vru_tracks');

% Pass tracking data to functions
all_car_tracks = carData.all_car_tracks;
all_vru_tracks = vruData.all_vru_tracks;

%%
% % Visualize the detections
visualizeDetections(folderPath, runNumber, all_car_tracks, all_vru_tracks);
% %%
% % Predict the trajectory
% predictTrajectory(folderPath, runNumber, all_car_tracks, all_vru_tracks);
% 
% % Generate CSV from the loaded data
% generateCSVFromMat(folderPath, runNumber, all_car_tracks, all_vru_tracks);


%% loop 
% Base folder containing all the runs
baseFolderPath = char('/media/isc/Extreme Pro/Submission_Data/Test Data 1/');

% Load historical yaw data once
historicalYawDataPath = char('/home/isc/Downloads/Pass_Vehicle_xyz_zrot.csv');
% Debugging: Print the value and type of historicalYawDataPath before use
fprintf('Loading historicalYawData from path: %s (Type: %s)\n', historicalYawDataPath, class(historicalYawDataPath));

if ~isfile(historicalYawDataPath)
    error('The file at historicalYawDataPath does not exist: %s', historicalYawDataPath);
end

historicalYawData = readmatrix(historicalYawDataPath); % Load the data once at the start

% Debugging: Confirm the successful load and print size of loaded data
fprintf('Successfully loaded historicalYawData. Size: %d rows and %d columns\n', size(historicalYawData, 1), size(historicalYawData, 2));

% List all folders in the base directory that match the pattern "Run_XX"
runFolders = dir(fullfile(baseFolderPath, 'Run_*'));
runFolders = runFolders([runFolders.isdir]); % Ensure we only get directories

% Extract the run numbers from folder names
runNumbers = nan(1, length(runFolders)); % Initialize an array for run numbers

for i = 1:length(runFolders)
    runNumberStr = regexp(runFolders(i).name, '\d+', 'match');
    if ~isempty(runNumberStr)
        runNumbers(i) = str2double(runNumberStr{1});
    end
end

% Remove entries where no valid run number was found
runNumbers = runNumbers(~isnan(runNumbers));

% Process each run sequentially using a for loop
for k = 1:length(runNumbers)
    try
        % Extract run number for current iteration
        runNumber = runNumbers(k);
        
        % Display which run number is being processed
        fprintf('Processing Run_%d...\n', runNumber);

        % Construct the run folder path and tracking folder path
        runFolderPath = char(fullfile(baseFolderPath, ['Run_', num2str(runNumber)]));
        trackingFolderPath = char(fullfile(runFolderPath, 'tracking'));

        % Debugging: Print paths to verify they are correctly formed
        fprintf('runFolderPath: %s\n', runFolderPath);
        fprintf('trackingFolderPath: %s\n', trackingFolderPath);
        [all_car_detections, all_vru_detections] = processPcdFolder(baseFolderPath, runNumber, historicalYawData);
        % Create the tracking folder if it does not exist
        if ~isfolder(trackingFolderPath)
            mkdir(trackingFolderPath);
            fprintf('Created tracking folder: %s\n', trackingFolderPath);
        end

        % Load tracking data once
        fprintf('Loading tracking data...\n');
        carData = load(fullfile(trackingFolderPath, 'car_detections.mat'), 'all_car_tracks');
        vruData = load(fullfile(trackingFolderPath, 'vru_detections.mat'), 'all_vru_tracks');

        % Extract loaded data
        all_car_tracks = carData.all_car_tracks;
        all_vru_tracks = vruData.all_vru_tracks;

        % Visualize the detections
        % visualizeDetections(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks);

        % Predict the trajectory
        predictTrajectory(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks);

        % Generate CSV from the loaded data
        generateCSVFromMat(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks);

        % Print completion message
        fprintf('Completed processing Run_%d\n', runNumber);

        % Clear variables to free up memory
        clear all_car_tracks all_vru_tracks carData vruData;

    catch ME
        % Log error to console
        fprintf('Error processing Run_%d: %s\n', runNumber, ME.message);
        fprintf('Error occurred in function: %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
end

% below are the functions
%% process pcd folder and generate outputs. 


function [all_car_detections, all_vru_detections] = processPcdFolder(baseFolderPath, runNumber, historicalYawData)
    % Construct full path to merged point cloud data folder
    folderPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], 'merged_point_cloud_data');
    
    carTracker = trackerJPDA();  % Initialize tracker for cars
    vruTracker = trackerJPDA();  % Initialize tracker for VRUs
    % Get list of all PCD files
    pcdFiles = dir(fullfile(folderPath, '*.pcd'));
    numFiles = numel(pcdFiles);

    % Initialize storage for detections and tracks
    all_car_detections = cell(numFiles, 1);
    all_car_tracks = cell(numFiles, 1);  % Store car tracks for each frame
    all_vru_detections = cell(numFiles, 1);
    all_vru_tracks = cell(numFiles, 1);  % Store VRU tracks for each frame

    % Define time parameters
    time = 0;
    dt = 0.1;

    % Create output tracking directory if it doesn't exist
    outputDir = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], 'tracking');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Process each frame
    for i = 1:numFiles
        % Read the point cloud
        pcdFilePath = fullfile(folderPath, pcdFiles(i).name);
        ptCloud = pcread(pcdFilePath);

        % Detect bounding boxes in the point cloud for vehicles
        [carBboxCuboid, carScores, carLabels] = detectCarBoundingBoxes(ptCloud);
        
        % Detect bounding boxes in the point cloud for VRUs
        [vruBboxCuboid, vruScores, vruLabels] = detectVRUBoundingBoxes(ptCloud);

        % Store detections for each frame
        all_car_detections{i} = struct('Frame', i, 'BoundingBoxes', carBboxCuboid, ...
                                       'Scores', carScores, 'Labels', carLabels);
        all_vru_detections{i} = struct('Frame', i, 'BoundingBoxes', vruBboxCuboid, ...
                                       'Scores', vruScores, 'Labels', vruLabels);

        % Create detection objects for vehicles
        numCarDetections = size(carBboxCuboid, 1);
        carDetections = cell(numCarDetections, 1);
        for j = 1:numCarDetections
            detected_center = carBboxCuboid(j, 1:3); 
            attributes = {carBboxCuboid(j, :)};
            carDetections{j} = objectDetection(time, detected_center', ...
                                               "ObjectAttributes", attributes);
        end

        % Create detection objects for VRUs
        numVruDetections = size(vruBboxCuboid, 1);
        vruDetections = cell(numVruDetections, 1);
        for j = 1:numVruDetections
            detected_center = vruBboxCuboid(j, 1:3); 
            attributes = {vruBboxCuboid(j, :)};
            vruDetections{j} = objectDetection(time, detected_center', ...
                                               "ObjectAttributes", attributes);
        end

        % Update trackers with detections if detections exist
        % Car Tracking
        if ~isempty(carDetections)
            current_car_tracks = carTracker(carDetections, time);

            % Update yaw angles using historical data
            for t = 1:numel(current_car_tracks)
                track = current_car_tracks(t);
                if ~isempty(track.ObjectAttributes) && ~isempty(track.ObjectAttributes{1})
                    tracker_box = track.ObjectAttributes{1}{1}; % Extract tracker box data

                    % Ensure the tracker box has enough elements to update yaw
                    if isnumeric(tracker_box) && numel(tracker_box) == 9
                        % Find nearest yaw using historical data
                        current_position = tracker_box(1:2); % Extract [x, y] from the tracker box
                        nearestYaw = findNearestYaw(current_position, historicalYawData);

                        % Update yaw angle in the tracker box
                        tracker_box(9) = nearestYaw;

                        % Update track attributes with the new yaw
                        track.ObjectAttributes{1}{1} = tracker_box;
                        current_car_tracks(t) = track;
                    else
                        fprintf('Warning: Tracker box does not have enough elements for yaw update in Frame %d, Track %d.\n', i, t);
                    end
                end
            end

            % Save the updated tracker output
            all_car_tracks{i} = current_car_tracks;
        else
            all_car_tracks{i} = [];
        end

        % VRU Tracking
        if ~isempty(vruDetections)
            current_vru_tracks = vruTracker(vruDetections, time);
            all_vru_tracks{i} = current_vru_tracks;  % Save the entire tracker output
        else
            all_vru_tracks{i} = [];
        end

        % Increment time
        time = time + dt;
    end

    % Save detections and tracks to MAT file in the tracking directory
    save(fullfile(outputDir, 'car_detections.mat'), 'all_car_detections', 'all_car_tracks');
    save(fullfile(outputDir, 'vru_detections.mat'), 'all_vru_detections', 'all_vru_tracks');
end

%% 
function visualizeDetections(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks)
    % Construct path to merged point cloud data
    folderPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], 'merged_point_cloud_data');

    % Get list of all PCD files
    pcdFiles = dir(fullfile(folderPath, '*.pcd'));
    numFiles = numel(pcdFiles);

    % Create pcplayer for visualization with fixed limits as provided
    xlimits = [-29.0409 52.1945];
    ylimits = [-49.9114 54.5738];
    zlimits = [-6 1];
    player = pcplayer(xlimits, ylimits, zlimits);
    xlabel(player.Axes, 'X (m)');
    ylabel(player.Axes, 'Y (m)');
    zlabel(player.Axes, 'Z (m)');

    % Loop through all frames for visualization
    for i = 1:numFiles
        % Read the point cloud without reducing its density
        pcdFilePath = fullfile(folderPath, pcdFiles(i).name);
        ptCloud = pcread(pcdFilePath);

        % Aggregate all tracks for the current frame
        fprintf('==== Frame %d - Visualizing All Tracks ====\n', i);

        % Initialize matrices to store bounding boxes and labels
        all_boxes = [];
        all_colors = []; % Initialize an empty array for colors

        % ----- Process Car Tracks -----
        car_tracks = all_car_tracks{i};
        for j = 1:numel(car_tracks)
            track = car_tracks(j);
            % Access ObjectAttributes
            if iscell(track.ObjectAttributes) && ~isempty(track.ObjectAttributes{1})
                tracker_attributes = track.ObjectAttributes{1};
                if iscell(tracker_attributes) && ~isempty(tracker_attributes{1})
                    tracker_box = tracker_attributes{1}; % Extract actual tracker box data
                    if isnumeric(tracker_box) && numel(tracker_box) == 9
                        all_boxes = [all_boxes; tracker_box];
                        all_colors = [all_colors; 1, 0, 0]; % Add color (red) for car
                    end
                end
            end
        end

        % ----- Process VRU Tracks -----
        vru_tracks = all_vru_tracks{i};
        for j = 1:numel(vru_tracks)
            track = vru_tracks(j);
            if iscell(track.ObjectAttributes) && ~isempty(track.ObjectAttributes{1})
                tracker_attributes = track.ObjectAttributes{1};
                if iscell(tracker_attributes) && ~isempty(tracker_attributes{1})
                    tracker_box = tracker_attributes{1}; % Extract actual tracker box data
                    if isnumeric(tracker_box) && numel(tracker_box) == 9
                        all_boxes = [all_boxes; tracker_box];
                        all_colors = [all_colors; 0, 0, 1]; % Add color (blue) for VRU
                    end
                end
            end
        end

        % Visualize all bounding boxes
        if ~isempty(all_boxes)
            view(player, ptCloud); % Display the point cloud for context
            showShape('cuboid', all_boxes, ...
                      'Parent', player.Axes, 'Color', all_colors, ...
                      'Opacity', 0.7, 'LineWidth', 2.0);
            fprintf('All bounding boxes visualized for Frame %d.\n', i);
        else
            fprintf('No bounding boxes found for Frame %d.\n', i);
        end

        drawnow;
        pause(0.1); % Adjust pause for smoother visualization (if needed)
    end
end
%%
% predict traj 
function predictTrajectory(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks)
    % Define parameters for prediction
    finalFrame = size(all_car_tracks, 1);
    predictSteps = 50;
    dt = 0.1;
    trackingFolderPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], 'tracking');
    % Load timestamps CSV for the last 50 frames
    timestampCsvPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], ['Run_', num2str(runNumber), '_Prediction_Timstamps.csv']);
    timestampData = readmatrix(timestampCsvPath);

    % Code for predicting trajectory using the data from the loaded car and VRU tracks...
% Initialize empty arrays to store final predicted data
    timestamps = [];
    path_ID = [];
    subclass = {};
    x_center = [];
    y_center = [];
    z_center = [];
    x_length = [];
    y_length = [];
    z_length = [];
    confidence_score = [];

    % Iterate through the last 10 frames and extract the latest occurrence of each track
    last_occurrences = containers.Map('KeyType', 'double', 'ValueType', 'any');
    startFrame = max(1, finalFrame - 9);  % Only iterate through the last 10 frames
    for i = startFrame:finalFrame
        % Process Car Tracks
        car_tracks = all_car_tracks{i};
        num_car_tracks = size(car_tracks, 1);
        for j = 1:num_car_tracks
            track = car_tracks(j);
            track_id = track.TrackID;
            last_occurrences(track_id) = struct('frame', i, 'timestamp', track.UpdateTime, 'state', track.State, 'box', track.ObjectAttributes{1}{1}, 'class', 'Passenger_Vehicle');
        end

        % Process VRU Tracks
        vru_tracks = all_vru_tracks{i};
        num_vru_tracks = size(vru_tracks, 1);
        for j = 1:num_vru_tracks
            track = vru_tracks(j);
            track_id = track.TrackID;
            last_occurrences(track_id) = struct('frame', i, 'timestamp', track.UpdateTime, 'state', track.State, 'box', track.ObjectAttributes{1}{1}, 'class', 'VRU_Adult');
        end
    end

    % Predict trajectories for each unique track ID
    all_track_ids = keys(last_occurrences);
    for idx = 1:numel(all_track_ids)
        track_id = all_track_ids{idx};
        data = last_occurrences(track_id);
        last_frame = data.frame;
        last_state = data.state;
        box_dimensions = data.box(4:6);  % Extract x_length, y_length, z_length
        object_class = data.class;
        
        % Extract position and velocity from state
        x = last_state(1);
        vx = last_state(2);
        y = last_state(3);
        vy = last_state(4);
        z = data.box(3);  % Use z_center from bounding box
        
        % Predict trajectory for the next 50 frames, starting from frame (finalFrame + 1)
        for step = 1:predictSteps
            % Predict next position using constant velocity model
            next_x = x + vx * (step + (finalFrame - last_frame)) * dt;
            next_y = y + vy * (step + (finalFrame - last_frame)) * dt;
            
            % Only record predictions starting from the frame after the last actual frame
            if step > (finalFrame - last_frame)
                % Append data to the arrays for CSV writing
                timestamp = timestampData(step);  % Use the appropriate timestamp from the CSV data
                timestamps = [timestamps; timestamp];
                path_ID = [path_ID; idx];  % Use unique path ID for each track
                subclass = [subclass; object_class];
                x_center = [x_center; next_x];
                y_center = [y_center; next_y];
                z_center = [z_center; z];
                x_length = [x_length; box_dimensions(1)];
                y_length = [y_length; box_dimensions(2)];
                z_length = [z_length; box_dimensions(3)];
                confidence_score = [confidence_score; 1];  % Assuming fixed confidence score
            end
        end
    end

    % Save predicted trajectories to CSV file with headers
    T = table(timestamps, path_ID, subclass, x_center, y_center, z_center, x_length, y_length, z_length, confidence_score);
    outputCsvPath = fullfile(trackingFolderPath, ['Path_Prediction_Submission_Run_', num2str(runNumber), '.csv']);
    writetable(T, outputCsvPath, 'WriteVariableNames', true);
    fprintf('CSV file saved to: %s\n', outputCsvPath);
end

%% 
% generate the classification and the localisation file.

function generateCSVFromMat(baseFolderPath, runNumber, all_car_tracks, all_vru_tracks)
    % Load timestamps from CSV file
    timestampCsvPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], ['Run_', num2str(runNumber), '_Detect_Classify_Localize_Timestamps.csv']);
    timestampData = readmatrix(timestampCsvPath);
    trackingFolderPath = fullfile(baseFolderPath, ['Run_', num2str(runNumber)], 'tracking');

     % Initialize arrays for CSV data
    timestamps = [];
    subclass = {};
    x_center = [];
    y_center = [];
    z_center = [];
    x_length = [];
    y_length = [];
    z_length = [];
    z_rotation = [];

    % Iterate through all frames and extract information
    numFrames = numel(all_car_tracks);
    for i = 1:numFrames
        % Get timestamp for the current frame
        if i <= length(timestampData)
            currentTimestamp = timestampData(i);
        else
            currentTimestamp = NaN;  % Assign NaN if there are no matching timestamps
        end

        % Process Car Tracks
        car_tracks = all_car_tracks{i};
        num_car_tracks = size(car_tracks, 1);
        for j = 1:num_car_tracks
            track = car_tracks(j);
            if iscell(track.ObjectAttributes) && ~isempty(track.ObjectAttributes{1})
                tracker_box = track.ObjectAttributes{1}{1};
                if isnumeric(tracker_box) && numel(tracker_box) == 9
                    % Append data for car track
                    timestamps = [timestamps; currentTimestamp];
                    subclass = [subclass; "Passenger_Vehicle"];
                    x_center = [x_center; tracker_box(1)];
                    y_center = [y_center; tracker_box(2)];
                    z_center = [z_center; tracker_box(3)];
                    x_length = [x_length; tracker_box(4)];
                    y_length = [y_length; tracker_box(5)];
                    z_length = [z_length; tracker_box(6)];
                    z_rotation = [z_rotation; tracker_box(9)];
                end
            end
        end

        % Process VRU Tracks
        vru_tracks = all_vru_tracks{i};
        num_vru_tracks = size(vru_tracks, 1);
        for j = 1:num_vru_tracks
            track = vru_tracks(j);
            if iscell(track.ObjectAttributes) && ~isempty(track.ObjectAttributes{1})
                tracker_box = track.ObjectAttributes{1}{1};
                if isnumeric(tracker_box) && numel(tracker_box) == 9
                    % Append data for VRU track
                    timestamps = [timestamps; currentTimestamp];
                    subclass = [subclass; "VRU_Adult"];
                    x_center = [x_center; tracker_box(1)];
                    y_center = [y_center; tracker_box(2)];
                    z_center = [z_center; tracker_box(3)];
                    x_length = [x_length; tracker_box(4)];
                    y_length = [y_length; tracker_box(5)];
                    z_length = [z_length; tracker_box(6)];
                    z_rotation = [z_rotation; tracker_box(9)];
                end
            end
        end
    end

    % Create table for the CSV file
    T = table(timestamps, subclass, x_center, y_center, z_center, x_length, y_length, z_length, z_rotation);

    % Save the table to a CSV file
    outputCsvPath = fullfile(trackingFolderPath, ['Detection_Classification_Localization_Submission_Run_', num2str(runNumber), '.csv']);
    writetable(T, outputCsvPath, 'WriteVariableNames', true);

    fprintf('CSV file saved to: %s\n', outputCsvPath);
end


function nearestYaw = findNearestYaw(current_position, historicalYawData)
    % Find the nearest yaw value from historical data based on the current position
    %
    % Inputs:
    %   - current_position: 1x2 vector representing [x, y] position.
    %   - historicalYawData: Matrix with columns [x, y, yaw].
    %
    % Output:
    %   - nearestYaw: The yaw value from historical data corresponding to the nearest [x, y] point.

    % Calculate distances to all historical points
    distances = sqrt((historicalYawData(:, 1) - current_position(1)).^2 + ...
                     (historicalYawData(:, 2) - current_position(2)).^2);

    % Find the nearest point and extract yaw
    [~, idx] = min(distances);
    nearestYaw = historicalYawData(idx, 3); 
end





