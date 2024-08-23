% This function takes the hue patch predictions (allStimParams) and actual
% power values (allPower) for different images. It computes the predicted
% power when one or more of the patch properties are used for prediction.

% The predictions are for the following conditions:
% 1. Only H. Remaining to their max values 
% 2. Only S.
% 3. Only V.
% 4. HS
% 5. HSV
% 6. HSVR (full model)

% The predicted power values for the full model are also returned.

% Correlation is computed for two cases:
% 1. All images (correlationValsFull)
% 2. selected images for which predition is deemed non-trivial (r>rCutoff)
% (correlationValsSelected)

function [correlationValsFull,correlationValsSelected,predictionString,predictedPower,selectedImageIndices,predictedPower2,mCorr1,mCorr2] = getAllCorrelations(subjectName,allStimParams,allPower,rCutoff,predictionVals,getMeanPredictionsAcrossFoldsFlag)

if ~exist('rCutoff','var');         rCutoff = 0.3;                      end
if ~exist('predictionVals','var');  predictionVals = [];                end
if ~exist('getMeanPredictionsAcrossFoldsFlag','var'); getMeanPredictionsAcrossFoldsFlag = 1;    end

rMax = 10; % Large radius for which the gamma vs radius function saturates

numStimuli = length(allStimParams);
selectedImageIndices = [];
for i=1:numStimuli
    if (allStimParams{i}.radiusDeg > rCutoff)
        selectedImageIndices = cat(2,selectedImageIndices,i);
    end
end

% Only use Hue
tmpStimParams = allStimParams;
for i=1:numStimuli
    tmp = tmpStimParams{i};
    tmp.sat = 1;
    tmp.contrastPC = 100;
    tmp.spatialFreqPhaseDeg = 90;
    tmp.radiusDeg = rMax;
    tmpStimParams{i} = tmp;
end
predictionString{1} = 'H';
[correlationValsFull(1),correlationValsSelected(1)] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices);

% Only use Sat
tmpStimParams = allStimParams;
for i=1:numStimuli
    tmp = tmpStimParams{i};
    tmp.hueDeg = 0; % Choose red
    tmp.contrastPC = 100;
    tmp.spatialFreqPhaseDeg = 90;
    tmp.radiusDeg = rMax;
    tmpStimParams{i} = tmp;
end
predictionString{2} = 'S';
[correlationValsFull(2),correlationValsSelected(2)] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices);

% Only use Val - leave both contrastPC and spatialFreqPhaseDeg unchanged
tmpStimParams = allStimParams;
for i=1:numStimuli
    tmp = tmpStimParams{i};
    tmp.hueDeg = 0; % Choose red
    tmp.sat = 1;
    tmp.radiusDeg = rMax;
    tmpStimParams{i} = tmp;
end
predictionString{3} = 'V';
[correlationValsFull(3),correlationValsSelected(3)] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices);

% Only Hue and Sat
tmpStimParams = allStimParams;
for i=1:numStimuli
    tmp = tmpStimParams{i};
    tmp.contrastPC = 100;
    tmp.spatialFreqPhaseDeg = 90;
    tmp.radiusDeg = rMax;
    tmpStimParams{i} = tmp;
end
predictionString{4} = 'HS';
[correlationValsFull(4),correlationValsSelected(4)] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices);

% Use HSV
tmpStimParams = allStimParams;
for i=1:numStimuli
    tmp = tmpStimParams{i};
    tmp.radiusDeg = rMax;
    tmpStimParams{i} = tmp;
end
predictionString{5} = 'HSV';
[correlationValsFull(5),correlationValsSelected(5)] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices);

% Use HSVR (full model)
predictionString{6} = 'HSVR';
[correlationValsFull(6),correlationValsSelected(6),predictedPower] = getCorrelations(subjectName,allStimParams,allPower,selectedImageIndices); % Full model

% Extend if predictionVals are provided
if ~isempty(predictionVals)
    % Correlation of power with just the prediction values
    predictionString{7} = 'P';
    [correlationValsFull(7),correlationValsSelected(7)] = getCorrelations2(allPower,predictionVals,selectedImageIndices);

    % Correlation after combining the predicted power from HSVR and
    % predictionVals
    predictionString{8} = 'HSVR+P';
    [predictedPower2,mCorrs] = combinePredictability(allPower,predictedPower,predictionVals);
    [correlationValsFull(8),correlationValsSelected(8)] = getCorrelations2(allPower,predictedPower2,selectedImageIndices);
    
    if getMeanPredictionsAcrossFoldsFlag
        correlationValsFull(6) = mCorrs(1);
        correlationValsFull(8) = mCorrs(2);
    end
end
end

function [rFull,rSelected,predictedPower] = getCorrelations(subjectName,tmpStimParams,allPower,selectedImageIndices)
numStimuli = length(tmpStimParams);

predictedPower = zeros(1,numStimuli);
for i=1:numStimuli
    predictedPower(i) = getPredictedGamma(subjectName,tmpStimParams{i});
end
[rFull,rSelected] = getCorrelations2(allPower,predictedPower,selectedImageIndices);
end
function [rFull,rSelected] = getCorrelations2(allPower,predictedPower,selectedImageIndices)

tmp = corrcoef(allPower,predictedPower);
rFull = tmp(1,2);
if length(selectedImageIndices)>2 % Need to have at least 3 data points. Otherwise correlations are trivially 1 or -1.
    tmp = corrcoef(allPower(selectedImageIndices),predictedPower(selectedImageIndices));
    rSelected = tmp(1,2);
else
    rSelected = 0;
end
end

function [Y,mCorrs] = combinePredictability(allPower,predictedPower,predictionVals)

numFolds = 1; % - 1: No cross-validation, length(allPower): Leave one out
numSamples = length(allPower(:));

trainingSet = cell(1,numFolds);
testingSet = cell(1,numFolds);

if numFolds==1
    trainingSet{1} = 1:numSamples;
    testingSet{1} = 1:numSamples;

elseif numFolds==numSamples
    for i=1:numFolds
        trainingSet{i} = setdiff(1:numSamples,i);
        testingSet{i} = i;
    end
else
    cvp = cvpartition(zeros(1,numSamples),'KFold',numFolds);
    for i=1:numFolds
        trainingSet{i} = find(cvp.training(i));
        testingSet{i} = find(cvp.test(i));
    end
end

allPower = allPower(:);
predictedPower = predictedPower(:);
predictionVals = predictionVals(:);

Y = zeros(numSamples,1);
corr1 = zeros(1,numFolds);
corr2 = zeros(1,numFolds);

for i=1:numFolds
    trainPredictedPower = predictedPower(trainingSet{i});
    trainPredictionVals = predictionVals(trainingSet{i});
    trainPower = allPower(trainingSet{i});

    b = regress(trainPower(:),[ones(length(trainPower),1) trainPredictedPower(:) trainPredictionVals(:)]);
    testPredictionVals = b(1)+ b(2)*predictedPower(testingSet{i}) + b(3)*predictionVals(testingSet{i});

    Y(testingSet{i}) = testPredictionVals;

    % Also compute correlations for individual folds
    tmp = corrcoef(allPower(testingSet{i}),predictedPower(testingSet{i}));
    corr1(i) = tmp(1,2);

    tmp = corrcoef(allPower(testingSet{i}),testPredictionVals);
    corr2(i) = tmp(1,2);
end
mCorrs = [mean(corr1) mean(corr2)];
end