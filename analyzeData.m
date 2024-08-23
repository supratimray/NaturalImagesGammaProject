function [correlationsFull,correlationsSelected,numSelectedImages,predictionString] = analyzeData(subjectName,expDate,protocolName,imageFolderName,imageIndices,powerOption,selectOptions,radiusMatrixDeg,folderSourceString,predictionType)

if ~exist('powerOption','var');         powerOption=3;                  end
if ~exist('radiusMatrixDeg','var');     radiusMatrixDeg=[];             end
if ~exist('predictionType','var');      predictionType='None';          end

rCutoff = 0.3;
patchSizeDeg = 2;
if isempty(radiusMatrixDeg)
    radiusMatrixDeg = 0.3:0.3:patchSizeDeg;
else
    patchSizeDeg = max(patchSizeDeg,max(radiusMatrixDeg));
end

plottingDetails.displayPlotsFlag=0;
    
if ~exist('folderSourceString','var');  folderSourceString = '';        end
if isempty(folderSourceString)
    folderSourceString = fileparts(pwd);
end

rawImageFolder = fullfile(folderSourceString,'data','images',imageFolderName);

% 1. Get actual gamma power
[powerST,powerBL,electrodeList] = getMeanEnergy(subjectName,expDate,protocolName);
if powerOption==1
    powerST = squeeze(powerST(:,:,imageIndices)); % Only take stimulus power
elseif powerOption==2
    powerST = squeeze(powerST(:,:,imageIndices)) ./ squeeze(powerBL(:,:,imageIndices)); % Ratio between ST and BL
elseif powerOption==3
    [powerST2,powerBL2,electrodeList] = getMeanEnergy(subjectName,expDate,protocolName,'',{[80 150]}); % Take power between 80 to 150 Hz
    powerST = squeeze(powerST(:,:,imageIndices)) ./ squeeze(powerBL(:,:,imageIndices)); % Ratio between ST and BL
    powerST2 = squeeze(powerST2(:,:,imageIndices)) ./ squeeze(powerBL2(:,:,imageIndices)); % Ratio between ST and BL in high gamma
    powerST = powerST - powerST2;
end

% 2. Get stimulus parameters for image patches
disp('Getting stim params...');
numElectrodes = size(powerST,1);
numImages = size(powerST,2);
allStimParams = cell(numElectrodes,numImages);
allPatchData = cell(numElectrodes,numImages);

for i=1:numImages
    % Load image
    imageFileName = fullfile(rawImageFolder,['Image' num2str(imageIndices(i)) '.png']);
    [patchData,imageAxesDeg] = getImagePatches(imageFileName,electrodeList,subjectName,folderSourceString,patchSizeDeg,plottingDetails);
    allPatchData(:,i) = patchData;
    
    % Get Stim Parameters
    for j=1:numElectrodes
        stimParams = getSingleImageParameters(rgb2hsv(patchData{j}),imageAxesDeg,[0 0],radiusMatrixDeg,selectOptions,0);
        allStimParams{j,i} = stimParams;
    end
end

% 3. Get prediction values from image patches
predictionVals = zeros(numElectrodes,numImages);
for i=1:numElectrodes
    stimParams = allStimParams(i,:);
    patchData = allPatchData(i,:);
    predictionVals(i,:) = getPredictionValues(stimParams,patchData,imageAxesDeg,predictionType);
end

% 4. Get predicted gamma power and correlations
correlationsFull = zeros(8,numElectrodes);
correlationsSelected = zeros(8,numElectrodes);
numSelectedImages = zeros(1,numElectrodes);
for i=1:numElectrodes
    actualPower = powerST(i,:);
    stimParams = allStimParams(i,:);
    [correlationsFull(:,i),correlationsSelected(:,i),predictionString,~,selectedImageIndices] = getAllCorrelations(subjectName,stimParams,actualPower,rCutoff,predictionVals(i,:));
    numSelectedImages(i) = length(selectedImageIndices);
end
end