% This program generates grating stimuli and plots predicted gamma for the
% different grating hue/grating combinations used in the paper

% Inputs
% subjectName: 'alpaH' or 'kesariH'
% stimParamsToDisplay = 'SFOR','SZOR' or 'CNOR' for Gratings; 'Hue','SZHue', 'Sat' or 'Val' for Color stimuli

function plotGammaTuningCurves(subjectName,stimParamsToDisplay,displayStimuliFlag)

if ~exist('displayStimuliFlag','var');  displayStimuliFlag=1;           end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Make stimParamsList %%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(stimParamsToDisplay(end-1:end),'OR')
    stimType = 'Grating';
else
    stimType = 'HuePatch';
end

% Fixed parameters. Not needed to get predicted gamma but needed to generate the stimuli
gaborStim.azimuthDeg=0;
gaborStim.elevationDeg=0;
gaborStim.sigmaDeg=100000; % The program makeGaborStimulus actually produces Gabors. However, when sigma is extremely large, it is essentially a grating
    
if strcmp(stimType,'Grating')
    var1List = 0:22.5:157.5; % One trace for each orientation, which is the first variable    
else
    if strcmp(stimParamsToDisplay,'Hue')
        var1List = 1; % redundant variable, since only hue trace is shown.
    else
        var1List = 0:60:300; % One trace for each hue, which is the first variable
    end
end
numVar1 = length(var1List);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Grating options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colorNameList = jet(numVar1);
if strcmp(stimParamsToDisplay,'SFOR')
    var2List = [0.5 1 2 4 8]; % Spatial Frequency
    numVar2 = length(var2List);
    var2Name = 'SF, cpd';
    gaborStim.radiusDeg=100; % FS
    gaborStim.contrastPC=100;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    for i=1:numVar1
        for j=1:numVar2
            tmpGaborStim = gaborStim;            
            tmpGaborStim.spatialFreqCPD=var2List(j);
            tmpGaborStim.orientationDeg=var1List(i);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end
    
if strcmp(stimParamsToDisplay,'SZOR') % Size-Ori
    var2List = [0.3 0.6 1.2 2.4 4.8 9.6]; % Sizes
    numVar2 = length(var2List);
    var2Name = 'Radius, deg';
    gaborStim.spatialFreqCPD= 2;
    gaborStim.contrastPC=100;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    for i=1:numVar1
        for j=1:numVar2
            tmpGaborStim = gaborStim;            
            tmpGaborStim.radiusDeg=var2List(j);
            tmpGaborStim.orientationDeg=var1List(i);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end

if strcmp(stimParamsToDisplay,'CNOR') % Con-Ori
    var2List = [0 3.1 6.25 12.5 25 50 100]; % Cons
    numVar2 = length(var2List);
    var2Name = 'Contrast, %';
    gaborStim.spatialFreqCPD= 2; % 
    gaborStim.radiusDeg=100; % FS
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    for i=1:numVar1
        for j=1:numVar2
            tmpGaborStim = gaborStim;            
            tmpGaborStim.contrastPC=var2List(j);
            tmpGaborStim.orientationDeg=var1List(i);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Color options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(stimParamsToDisplay,'Hue')
    var2List = 0:10:350; % Hues
    numVar2 = length(var2List);
    var2Name = 'Hue, deg';
    gaborStim.radiusDeg=100; % FS
    gaborStim.contrastPC=100;
    gaborStim.orientationDeg=0; % Does not matter since SF is zero
    gaborStim.spatialFreqCPD=0; % For color patch
    gaborStim.spatialFreqPhaseDeg=90;
    gaborStim.sat = 1;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    colorNameList = [1 0 0];
    for i=1:numVar1
        for j=1:numVar2
            tmpGaborStim = gaborStim;            
            tmpGaborStim.hueDeg = var2List(j);
                 
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end

if strcmp(stimParamsToDisplay,'SZHue')
    var2List= [0.15 0.3 0.6 1.2 2.4 4.8 9.6]; % Sizes
    numVar2 = length(var2List);
    var2Name = 'Radius, deg';
    gaborStim.contrastPC=100;
    gaborStim.orientationDeg=0; % Does not matter since SF is zero
    gaborStim.spatialFreqCPD=0; % For color patch
    gaborStim.spatialFreqPhaseDeg=90;
    gaborStim.sat = 1;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    colorNameList = zeros(numVar1,3);
    
    for i=1:numVar1
        colorNameList(i,:) = hsv2rgb([var1List(i)/360 1 1]);
        
        for j=1:numVar2
            tmpGaborStim = gaborStim;
            tmpGaborStim.radiusDeg=var2List(j);
            tmpGaborStim.hueDeg = var1List(i);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end

if strcmp(stimParamsToDisplay,'Sat')
    var2List = 0:0.25:1; % Saturation values
    numVar2 = length(var2List);
    var2Name = 'Saturation';
    gaborStim.radiusDeg=100; % FS
    gaborStim.contrastPC=100;
    gaborStim.orientationDeg=0; % Does not matter since SF is zero
    gaborStim.spatialFreqCPD=0; % For color patch
    gaborStim.spatialFreqPhaseDeg=90;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    colorNameList = zeros(numVar1,3);
    
    for i=1:numVar1
        colorNameList(i,:) = hsv2rgb([var1List(i)/360 1 1]);
        
        for j=1:numVar2
            tmpGaborStim = gaborStim;
            tmpGaborStim.hueDeg = var1List(i);
            tmpGaborStim.sat = var2List(j);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end

if strcmp(stimParamsToDisplay,'Val')
    var2List = [0 6.25 12.5 25 50 100]; % Contrast (Value)
    numVar2 = length(var2List);
    var2Name = 'Value, %';
    gaborStim.radiusDeg=100; % FS
    gaborStim.orientationDeg=0; % Does not matter since SF is zero
    gaborStim.spatialFreqCPD=0; % For color patch
    gaborStim.spatialFreqPhaseDeg=90;
    gaborStim.sat = 1;
    
    stimParamsList = cell(numVar1,numVar2);
    predictedGammaList = zeros(numVar1,numVar2);
    colorNameList = zeros(numVar1,3);
    
    for i=1:numVar1
        colorNameList(i,:) = hsv2rgb([var1List(i)/360 1 1]);
        
        for j=1:numVar2
            tmpGaborStim = gaborStim;
            tmpGaborStim.hueDeg = var1List(i);
            tmpGaborStim.contrastPC = var2List(j);
            
            stimParamsList{i,j} = tmpGaborStim;
            predictedGammaList(i,j) = getPredictedGamma(subjectName,stimParamsList{i,j});
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot Gamma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(stimParamsToDisplay,'Hue')  % has only var2List. var1List=1;
    predictedGammaList = circshift(predictedGammaList(:),length(predictedGammaList)/2);  % to keep red hue 0 in middle
    stimParamsList     = circshift(stimParamsList(:), length(stimParamsList)/2); 
    predictedGammaList = predictedGammaList'; stimParamsList = stimParamsList';
end

hGamma = subplot('Position',[0.075 0.075 0.375 0.9]); hold(hGamma,'on');
legendStr = cell(1,numVar1);
for i=1:numVar1
    plot(hGamma,predictedGammaList(i,:),'color',colorNameList(i,:));
    legendStr{i} = num2str(var1List(i));
end
set(hGamma,'XTick',1:length(var2List),'XTickLabel',var2List);
xlabel(var2Name); ylabel('Scaled Gamma response');
legend(hGamma,legendStr,'location','best');

%%%%%%%%%%%%%%%%%%%%%%% Display Stimuli %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if displayStimuliFlag
    hStimuli = getPlotHandles(numVar1,numVar2,[0.5 0.075 0.45 0.9]);
    
    % Each stimulus must be of the size specified by monitorSpecifications.
    [xAxisDeg,yAxisDeg] = getMonitorDetails;
    xAxisDeg = downsample(xAxisDeg,5); 
    yAxisDeg = downsample(yAxisDeg,5);
    colormap gray
    for i=1:numVar1
        for j=1:numVar2
            gaborStim = stimParamsList{i,j};
            gaborPatch = makeGaborStimulus(gaborStim,xAxisDeg,yAxisDeg,0);
            imagesc(xAxisDeg,yAxisDeg,gaborPatch,'Parent',hStimuli(i,j));
            caxis(hStimuli(i,j),[0 1]);
        end
    end
end