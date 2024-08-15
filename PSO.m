function [Best_Cost, Best_X, Convergence_curve] = PSO(nP, MaxIt, lb, ub, dim, fitness_func)
    % PSO Parameters
    w = 0.5;        % Inertia weight
    c1 = 2;         % Personal acceleration coefficient
    c2 = 2;         % Social acceleration coefficient

    % Initialization
    X = rand(nP, dim) .* (ub - lb) + lb;  % Particle positions
    V = rand(nP, dim);                    % Particle velocities
    PersonalBest = X;                     % Personal best positions
    PersonalBestCost = zeros(nP, 1);      % Personal best costs
    GlobalBest = zeros(1, dim);           % Global best position
    GlobalBestCost = Inf;                 % Global best cost
    Convergence_curve = zeros(MaxIt, 1);  % Convergence curve

    % Main Loop
    for it = 1:MaxIt
        for i = 1:nP
            % Evaluate fitness
            cost = fitness_func(X(i, :));

            % Update personal best
            if cost < PersonalBestCost(i)
                PersonalBest(i, :) = X(i, :);
                PersonalBestCost(i) = cost;
            end

            % Update global best
            if cost < GlobalBestCost
                GlobalBest = X(i, :);
                GlobalBestCost = cost;
            end
        end

        % Update velocities and positions
        for i = 1:nP
            r1 = rand(1, dim);
            r2 = rand(1, dim);

            V(i, :) = w * V(i, :) + c1 * r1 .* (PersonalBest(i, :) - X(i, :)) + c2 * r2 .* (GlobalBest - X(i, :));
            X(i, :) = X(i, :) + V(i, :);

            % Bound the particles
            X(i, :) = max(X(i, :), lb);
            X(i, :) = min(X(i, :), ub);
        end

        % Update convergence curve
        Convergence_curve(it) = GlobalBestCost;
    end

    % Output results
    Best_Cost = GlobalBestCost;
    Best_X = GlobalBest;
end
