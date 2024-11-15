function gridParams = getGridParameters()
% The getGridParameters function returns the grid parameters that controls
% the range of point cloud.
%
% Copyright 2021 The MathWorks, Inc.

    xMin = -30.0;     % Minimum value along X-axis.
    xMax = 62.0;      % Maximum value along X-axis.
    yMin = 0;         % Minimum value along Y-axsis.
    yMax = 105.0;      % Maximum value along Y-axis.l
    zMin = -7.0;      % Minimum value along Z-axis. % Original ROI network X [-25 25], Y: [0 50]
    zMax = 1.0;      % Maximum value along Z-axis.  %Original ROI ISC : X: [-30, 62], Y: [-50, 55],
    f_x  = (xMax - xMin)/50;
    f_y = (yMax - yMin)/50;

    pcRange = [xMin xMax yMin yMax zMin zMax];

    % Define the dimensions for the pseudo-image.
    bevHeight = round(f_x*608);
    bevWidth =  round(f_y*608);

    % Find grid resolution.
    gridW = (pcRange(4) - pcRange(3))/bevWidth;
    gridH = (pcRange(2) - pcRange(1))/bevHeight;
   
    gridParams.xMin = xMin;
    gridParams.xMax = xMax;
    gridParams.yMin = yMin;
    gridParams.yMax = yMax;
    gridParams.zMin = zMin;
    gridParams.zMax = zMax;
    
    gridParams.bevHeight = bevHeight;
    gridParams.bevWidth = bevWidth;
    
    gridParams.gridH = gridH;
    gridParams.gridW = gridW;    
end