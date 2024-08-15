    function fitness_value = fitnessSP(image, params)
    J = transFSP(image, params);
    fitness_value = -kurtosis(J(:))/entropy(J(:));
end
