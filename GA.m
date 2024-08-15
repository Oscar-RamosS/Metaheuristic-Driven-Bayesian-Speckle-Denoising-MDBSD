function [Best_Cost_GA, Best_X_GA, Convergence_curve_GA] = GA(nP, MaxIt, lb, ub, dim, fitness_func)
    % Parámetros del algoritmo genético
    pCrossover = 0.8; % Probabilidad de cruce
    pMutation = 0.1;  % Probabilidad de mutación
    mu = pMutation;    % Tasa de mutación
    
    % Inicialización
    pop = repmat(lb, nP, 1) + rand(nP, dim) .* (repmat(ub, nP, 1) - repmat(lb, nP, 1));
    costs = zeros(nP, 1);
    for i = 1:nP
        costs(i) = fitness_func(pop(i, :));
    end
    
    % Mejor solución encontrada hasta el momento
    [Best_Cost_GA, idx] = min(costs);
    Best_X_GA = pop(idx, :);
    
    % Historial de la evolución del costo
    Convergence_curve_GA = zeros(MaxIt, 1);
    
    % Ciclo de optimización
    for it = 1:MaxIt
        % Selección por torneo
        [~, SortOrder] = sort(costs);
        pop = pop(SortOrder(1:ceil(0.5 * nP)), :);
        
        % Cruce
        nCrossovers = 0.5 * pCrossover * nP;
        for k = 1:nCrossovers
            i1 = randi([1, size(pop, 1)]);
            i2 = randi([1, size(pop, 1)]);
            c1 = pop(i1, :);
            c2 = pop(i2, :);
            alpha = rand(1, dim);
            new_c1 = alpha .* c1 + (1 - alpha) .* c2;
            new_c2 = alpha .* c2 + (1 - alpha) .* c1;
            pop = [pop; new_c1; new_c2];
        end
        
        % Mutación
        nMutations = mu * nP * dim;
        for k = 1:nMutations
            i = randi([1, size(pop, 1)]);
            j = randi([1, size(pop, 2)]);
            mutation_value = lb(j) + rand() * (ub(j) - lb(j));
            pop(i, j) = mutation_value;
        end
        
        % Evaluación de la población
        costs = zeros(size(pop, 1), 1);
        for i = 1:size(pop, 1)
            costs(i) = fitness_func(pop(i, :));
        end
        
        % Actualización de la mejor solución
        [Best_Cost_GA, idx] = min(costs);
        Best_X_GA = pop(idx, :);
        
        % Actualización de la curva de convergencia
        Convergence_curve_GA(it) = Best_Cost_GA;
        
        % Mostrar información en cada iteración si se desea
        disp(['Iteración ', num2str(it), ': Mejor costo = ', num2str(Best_Cost_GA)]);
    end
end
