
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Implementation of the Metaheuristic-Driven Bayesian Speckle Denoising (MDBSD), whose main approach is the OCT image speckle denoising.
% 
% 'Enhancing Retinal OCT Scans via Metaheuristic-Driven Bayesian Speckle Denoising' 
% presented in the 2024 IEEE 37th International Symposium on Computer-Based Medical Systems (CBMS) 
% by Oscar Ramos-Soto, Angel Casas-Ordaz, Diego Oliva, Sandra E Balderas-Mata, Saúl Zapotecas-Martínez
% DOI: 10.1109/CBMS61543.2024.00062

%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
addpath("OBNLM")
Io = im2gray(imread('Images\p18_t57.png'));
Img_norm = ImgNormalize(Io);
img = imresize(Img_norm, [200,200]); 

%Reduce image size, Np, and MaxIt for faster computation
%Paper configuration: Population size = 10, Maximum iterations = 15, image
%size 500 by 500
nP =  5; %Particles number
MaxIt = 10; %Maximum iterations
lb = 1; %Lower boundary
ub = 30; %Lower boundary
dim = 1; %Dimensions


%Uncomment the algorithm to test, process the image (Res_PSO, Res_DE,
%Res_GA), and plot of convergence curves. You can process several more than
%one algorithm per run.

[Best_Cost_PSO,Best_Value_PSO,Convergence_curve_PSO] = PSO(nP, MaxIt, lb, ub, dim, @(x)fitnessSP(img, x));
% [Best_Cost_DE,Best_Value_DE,Convergence_curve_DE] = DE(nP, MaxIt, lb, ub, dim, @(x)fitnessSP(img, x));
% [Best_Cost_GA,Best_Value_GA,Convergence_curve_GA] = GA(nP, MaxIt, lb, ub, dim, @(x)fitnessSP(img, x));

Res_PSO = transFSP(img, Best_Value_PSO);
% Res_DE = transFSP(img, Best_Value_DE);
% Res_GA = transFSP(img, Best_Value_GA);

Res_PSO_upsized = imresize(Res_PSO, [400, 400]);
% Res_DE_upsized = imresize(Res_DE, [600, 600]);
% Res_GA_upsized = imresize(Res_GA, [600, 600]);

figure(1)
imshow(Io)
figure(2)
imshow(Res_PSO_upsized)
% imshow(Res_DE_upsized)
% imshow(Res_GA_upsized)

% Plot convergence curves
figure;
plot(1:MaxIt, Convergence_curve_PSO, 'LineWidth', 2, 'DisplayName', 'PSO');
% hold on;
% plot(1:MaxIt, Convergence_curve_DE, 'LineWidth', 2, 'DisplayName', 'DE');
% hold on;
% plot(1:MaxIt, Convergence_curve_GA, 'LineWidth', 2, 'DisplayName', 'GA');
% hold on;
title('Convergence Curves');
xlabel('Iteration');
ylabel('Best Fitness Value');
legend('show');
grid on;
hold off;
toc
% saveas(gcf, 'convergence_curve.png');
