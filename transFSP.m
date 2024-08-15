function B = transFSP(image, params)
    blockSize = 5; % size of the block
    windowSize = 21; % size of the search window
    gapBwnBlock = 2; % gap between the search block (in order to solve computational burden)
    h = params; % filtering parameter controlling the decay of the exponential function
    J = BayesianNLM(image, blockSize, windowSize, gapBwnBlock, h);
    B = imadjust(J, [], [], 1.8);
end
