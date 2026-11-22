function [r,fClean,eligibleIdx,selectedFiles,selectedFrameNumbers] = loadRaceCarSamples(imageFolder,centerCount,useAllImages)
% Match car100: pair grayscale image vectors with angles in physical frame order.
if ~isfolder(imageFolder)
    error('loadRaceCarSamples:MissingFolder','Image folder not found: %s',imageFolder);
end
imageFiles = dir(fullfile(imageFolder,'*.png'));
totalImages = numel(imageFiles);
if totalImages < 2
    error('loadRaceCarSamples:TooFewImages','At least two PNG images are required.');
end

frameTokens = regexpi({imageFiles.name},'(\d+)\.png$','tokens','once');
if any(cellfun('isempty',frameTokens))
    error('loadRaceCarSamples:MissingFrameNumber','PNG filenames must end in a numeric frame number.');
end
frameNumbers = cellfun(@(token) str2double(token{1}),frameTokens);
[frameNumbers,frameOrder] = sort(frameNumbers(:));
if any(diff(frameNumbers) ~= 1)
    error('loadRaceCarSamples:InvalidFrameOrder','Frame numbers must be unique and consecutive.');
end
imageFiles = imageFiles(frameOrder);
allAngles = (frameNumbers-frameNumbers(1)) * (2*pi/totalImages);

centerCount = min(centerCount,totalImages);
firstCenter = floor((totalImages-centerCount)/2)+1;
centerIdx = (firstCenter:firstCenter+centerCount-1).';
if useAllImages
    selectedIdx = (1:totalImages).';
    eligibleIdx = centerIdx;
else
    selectedIdx = centerIdx;
    eligibleIdx = (1:centerCount).';
end
selectedFiles = imageFiles(selectedIdx);
selectedFrameNumbers = frameNumbers(selectedIdx);
fClean = allAngles(selectedIdx);

firstImage = im2double(imread(fullfile(imageFolder,selectedFiles(1).name)));
if ~ismatrix(firstImage)
    error('loadRaceCarSamples:NotGrayscale','Expected grayscale images.');
end
r = zeros(numel(firstImage),numel(selectedFiles));
r(:,1) = firstImage(:);
for i = 2:numel(selectedFiles)
    img = im2double(imread(fullfile(imageFolder,selectedFiles(i).name)));
    if ~ismatrix(img) || ~isequal(size(img),size(firstImage))
        error('loadRaceCarSamples:ImageSizeMismatch','Image %s differs in size or is not grayscale.',selectedFiles(i).name);
    end
    r(:,i) = img(:);
end
end
