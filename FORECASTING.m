% MONTE CARLO.m: Code to examine all PVAR priors at once
% NOTE: This code replicates the results in Korobilis (2015)
% *************************************************************************
% Written by Dimitris Korobilis, on 15/03/2015
% University of Glasgow
% Dimitris.Korobilis@glasgow.ac.uk
% *************************************************************************

clear all;
clc;
tic;

% Add path of random number generators
addpath('functions');
addpath('data');

% Gibbs-related preliminaries
nsave = 5000;           % Number of draws to save
nburn = 2000;           % Number of draws to discard
ntot = nsave + nburn;   % Number of total draws
iter = 100;             % Print every "iter" iteration


% 1) VAR specification
p = 1;    % Number of lags
N = 10;   % Number of cross sections
G = 3;    % Number of VAR variables for each cross-section
h = 12;   % Forecast horizon

% Load Eurozone data, data are already in spreads from the German data
load spreads.dat;
load ip.dat;
% load debt.dat;
load bid_ask.dat;
spreads = spreads(1:end-1,:);
spreads = spreads(2:end,:) - spreads(1:end-1,:);
bid_ask = bid_ask(2:end,:) - bid_ask(1:end-1,:);
ip = ip(2:end,:);

% Define the final data used in the PVAR application
Yraw = [spreads(:,1:N),ip(:,1:N),bid_ask(:,1:N)];
Yraw = Yraw-repmat(mean(Yraw),size(Yraw,1),1);
Ylag = mlag2(Yraw,p); Xraw = Ylag(p+1:end,:); Yraw = Yraw(p+1:end,:);

t = size(Yraw,1);
M = size(Yraw,2);
t0 = round(0.6*t);

MSFE_BFCS = zeros(N,h,t-h-t0+1);
MSFE_SSSS = zeros(N,h,t-h-t0+1);
MSFE_BMS = zeros(N,h,t-h-t0+1);
MSFE_GLP = zeros(N,h,t-h-t0+1);
MSFE_SSVS = zeros(N,h,t-h-t0+1);
MSFE_CC = zeros(N,h,t-h-t0+1);
MSFE_OLS = zeros(N,h,t-h-t0+1);

PL_BFCS = zeros(N,h,t-h-t0+1);
PL_SSSS = zeros(N,h,t-h-t0+1);
PL_BMS = zeros(N,h,t-h-t0+1);
PL_GLP = zeros(N,h,t-h-t0+1);
PL_SSVS = zeros(N,h,t-h-t0+1);
PL_CC = zeros(N,h,t-h-t0+1);

for nMC = t0:t-h
    nMC
    
    Y = Yraw(1:nMC,:);
    X = Xraw(1:nMC,:);
    Y_f = Yraw(nMC+1:nMC+h,1:N);
    T = nMC;
    
    %======================================================================
    % Estimate PVAR coefficients using various priors
    [alpha_draws_BFCS] = MCMC_PVAR_BFCS(Y,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_SSSS] = MCMC_PVAR_SSSS(Y,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_BMS]  = MCMC_PVAR_BMS(Y,N,G,p,nsave,nburn,ntot,iter);
    
    [alpha_draws_GLP]  = MCMC_PVAR_GLP(Y,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_SSVS] = MCMC_PVAR_SSVS(Y,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_CC]   = MCMC_PVAR_CC(Y,N,G,p,nsave,nburn,ntot,iter);
    [alpha_OLS,sigma_OLS] = OLS_PVAR(Y,N,G,p);
        
    % Forecasting
    [Y_pred_BFCS] = PVARpred(Y,X,alpha_draws_BFCS,sigma_OLS,T,M,G,N,p,h,nsave);
    [Y_pred_SSSS] = PVARpred(Y,X,alpha_draws_SSSS,sigma_OLS,T,M,G,N,p,h,nsave);
    [Y_pred_BMS] = PVARpred(Y,X,alpha_draws_BMS,sigma_OLS,T,M,G,N,p,h,nsave);
    
    [Y_pred_GLP] = PVARpred(Y,X,alpha_draws_GLP,sigma_OLS,T,M,G,N,p,h,nsave);
    [Y_pred_SSVS] = PVARpred(Y,X,alpha_draws_SSVS,sigma_OLS,T,M,G,N,p,h,nsave);
    [Y_pred_CC] = PVARpred(Y,X,alpha_draws_CC,sigma_OLS,T,M,G,N,p,h,nsave);
    
    [Y_pred_OLS] = PVARpred(Y,X,repmat(alpha_OLS,1,nsave)',sigma_OLS,T,M,G,N,p,h,nsave);
    
    % Evaluation of mean forecasts
    MSFE_BFCS(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_BFCS,1)) - Y_f').^2;
    MSFE_SSSS(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_SSSS,1)) - Y_f').^2;
    MSFE_BMS(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_BMS,1)) - Y_f').^2;
    
    MSFE_GLP(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_GLP,1)) - Y_f').^2;
    MSFE_SSVS(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_SSVS,1)) - Y_f').^2;
    MSFE_CC(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_CC,1)) - Y_f').^2;
    
    MSFE_OLS(:,:,nMC-t0+1) = (squeeze(mean(Y_pred_OLS,1)) - Y_f').^2;
    
    % Predictive likelihoods
    for i = 1:h; for j = 1:N; PL_BFCS(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_BFCS(:,j,i)),Y_f(i,j)); end; end
    for i = 1:h; for j = 1:N; PL_SSSS(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_SSSS(:,j,i)),Y_f(i,j)); end; end
    for i = 1:h; for j = 1:N; PL_BMS(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_BMS(:,j,i)),Y_f(i,j)); end; end
    
    for i = 1:h; for j = 1:N; PL_GLP(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_GLP(:,j,i)),Y_f(i,j)); end; end
    for i = 1:h; for j = 1:N; PL_SSVS(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_SSVS(:,j,i)),Y_f(i,j)); end; end
    for i = 1:h; for j = 1:N; PL_CC(j,i,nMC-t0+1) = ksdensity(squeeze(Y_pred_CC(:,j,i)),Y_f(i,j)); end; end
end

save forecasting.mat;