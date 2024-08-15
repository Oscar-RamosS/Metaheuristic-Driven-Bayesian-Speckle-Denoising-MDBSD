clc;
clear all;

oimg = imread('p1_t20.png');
img = imresize(oimg, [512, 512]);
blockSize = 5; % size of the block
windowSize = 30; % size of the search window
gapBwnBlock = 2; % gap between the search block (in order to solve computational burden)
h = 10; % filtering parameter controlling the decay of the exponential function

img = ImgNormalize(img);
processedImg = BayesianNLM(img, blockSize, windowSize, gapBwnBlock, h);

figure
subplot 131
imshow(img)
title('Origin Image')
subplot 132
imshow(processedImg)
title('Despecked Image')
subplot 133
delta = ~logical(img - processedImg);
imshow(double(delta))
title('Subtraction Image')

% imwrite(processedImg, 'despeckledImage.png')
% imwrite(processedImg, 'despeckledImage.png')

restored_image = imresize(processedImg, size(oimg));
figure(2)
imshow(restored_image)