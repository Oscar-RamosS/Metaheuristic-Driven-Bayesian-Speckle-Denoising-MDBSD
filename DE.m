function [bestCost, bestSolution, convergenceCurve] = DE(nPop, maxIter, lb, ub, dim, fitnessFunc)
    % Parámetros del algoritmo DE
    F = 0.5; % Factor de escala
    CR = 0.9; % Tasa de cruce (crossover rate)

    % Inicialización de la población
    pop = repmat(lb, nPop, 1) + rand(nPop, dim) .* repmat((ub - lb), nPop, 1);
    costs = zeros(nPop, 1);

    % Evaluación inicial de la población
    for i = 1:nPop
        costs(i) = fitnessFunc(pop(i, :));
    end

    % Inicialización del mejor costo y solución
    [bestCost, idx] = min(costs);
    bestSolution = pop(idx, :);

    % Registro de la convergencia
    convergenceCurve = zeros(maxIter, 1);
    convergenceCurve(1) = bestCost;

    % Bucle principal de optimización
    for iter = 2:maxIter
        for i = 1:nPop
            % Selección aleatoria de 3 soluciones diferentes
            idxs = randperm(nPop, 3);
            a = pop(idxs(1), :);
            b = pop(idxs(2), :);
            c = pop(idxs(3), :);

            % Mutación
            mutant = a + F .* (b - c);

            % Recombinación
            mask = rand(1, dim) < CR;
            trial = pop(i, :);
            trial(mask) = mutant(mask);

            % Limitar la solución al rango
            trial = max(trial, lb);
            trial = min(trial, ub);

            % Evaluación de la función de aptitud
            costTrial = fitnessFunc(trial);

            % Actualización de la población
            if costTrial < costs(i)
                pop(i, :) = trial;
                costs(i) = costTrial;

                if costTrial < bestCost
                    bestCost = costTrial;
                    bestSolution = trial;
                end
            end
        end

        % Registro de la convergencia
        convergenceCurve(iter) = bestCost;
    end
end
