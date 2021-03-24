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
addpath('functions')


% Matrices to save stuff for Monte Carlo
% 1 a) Store posterior mean coefficients for the four PVAR priors
alpha_MCMC_BFCS = [];    % Bayesian Factor, Clustering and Selection 
alpha_MCMC_SSSS = [];    % Stochastic Search Specification Selection
alpha_MCMC_BMS = [];     % Bayesian Mixture Shrinkage
alpha_MCMC_BSS = [];     % Bayesian Semiparametric Shrinkage
% 1 b) Store posterior mean coefficients for competing priors
alpha_MCMC_GLP = [];     % Giannone, Lenza, Primiceri (2015)
alpha_MCMC_SSVS = [];    % George, Sun and Ni (2008)
alpha_MCMC_CC = [];      % Canova and Ciccarelli (2009)
alpha_OLS = [];          % OLS

% 2) Storage matrices for Mean Absolute Deviation (MAD), Mean Squared
% Deviation (MSD), and Trace Statistic (TRACE)
MAD_BFCS = [];  MSD_BFCS = [];  TRACE_BFCS = [];
MAD_SSSS = [];  MSD_SSSS = [];  TRACE_SSSS = [];
MAD_BMS = [];   MSD_BMS = [];   TRACE_BMS = [];
MAD_BSS = [];   MSD_BSS = [];   TRACE_BSS = [];
MAD_GLP = [];   MSD_GLP = [];   TRACE_GLP = [];
MAD_SSVS = [];  MSD_SSVS = [];  TRACE_SSVS = [];
MAD_CC = [];    MSD_CC = [];    TRACE_CC = [];
MAD_OLS = [];   MSD_OLS = [];   TRACE_OLS = [];

% Gibbs-related preliminaries
nsave = 50000;           % Number of draws to save
nburn = 20000;           % Number of draws to discard
ntot = nsave + nburn;   % Number of total draws
iter = 100;            % Print every "iter" iteration

for nMC = 1:100
    nMC
    
    % =============| Generate VAR model
    % 1) VAR specification
    p = 1;    % Number of lags
    N = 3;    % Number of cross sections
    G = 2;    % Number of VAR variables for each cross-section
    
    % 2) Generate VAR matrices
    PHI = zeros(N*G*p);
    PSI = eye(N*G);

    for i = 1:N*G*p
        for j = 1:N*G*p
            PHI(i,j) = 0.3*(0.5*rand^(abs(i-j)));
        end
    end
    if max(abs(eig(PHI)))>0.999
        error('Nonstationary VAR') 
    end
    
    for i = 1:N*G   
        for j = i+1:N*G
            PSI(i,j) = 0.5;
        end
    end
    
    % 3) Generate VAR data from DGP
    [Yraw,PHI] = simpvardgp(100,N*G,p,PHI,PSI);
    Ylag = mlag2(Yraw,p); X = Ylag(p+1:end,:);

    %======================================================================
    % Estimate PVAR coefficients using various priors
    [alpha_draws_BFCS] = MCMC_PVAR_BFCS(Yraw,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_SSSS] = MCMC_PVAR_SSSS(Yraw,N,G,p,nsave,nburn,ntot,iter);
%     [alpha_draws_BSS]  = MCMC_PVAR_BSS(Yraw,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_BMS]  = MCMC_PVAR_BMS(Yraw,N,G,p,nsave,nburn,ntot,iter);
    
    [alpha_draws_GLP]  = MCMC_PVAR_GLP(Yraw,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_SSVS] = MCMC_PVAR_SSVS(Yraw,N,G,p,nsave,nburn,ntot,iter);
    [alpha_draws_CC]   = MCMC_PVAR_CC(Yraw,N,G,p,nsave,nburn,ntot,iter);
    [alpha_OLS,~]        = OLS_PVAR(Yraw,N,G,p);
    
    % obtain posterior means, and reshape to a matrix
    a_BFCS = mean(alpha_draws_BFCS,1)'; a_SSSS = mean(alpha_draws_SSSS,1)';
%     a_BSS = mean(alpha_draws_BSS,1)';   
    a_BMS = mean(alpha_draws_BMS,1)';
    a_GLP = mean(alpha_draws_GLP,1)';   a_SSVS = mean(alpha_draws_SSVS,1)';
    a_CC = mean(alpha_draws_CC,1)';     a_OLS = alpha_OLS;  
    
    A_BFCS = reshape(mean(alpha_draws_BFCS,1)',N*G,N*G*p); A_SSSS = reshape(mean(alpha_draws_SSSS,1)',N*G,N*G*p);
%     A_BSS = reshape(mean(alpha_draws_BSS,1)',N*G,N*G*p);   
    A_BMS = reshape(mean(alpha_draws_BMS,1)',N*G,N*G*p);
    A_GLP = reshape(mean(alpha_draws_GLP,1)',N*G,N*G*p);   A_SSVS = reshape(mean(alpha_draws_SSVS,1)',N*G,N*G*p);
    A_CC = reshape(mean(alpha_draws_CC,1)',N*G,N*G*p);     A_OLS = reshape(alpha_OLS,N*G,N*G*p);   
    
    %======================================================================
    % Compute the accuracy of each prior
    MAD_BFCS = [MAD_BFCS; mean(mean(abs(X*A_BFCS - X*PHI)))];   MSD_BFCS = [MSD_BFCS; mean(mean((X*A_BFCS - X*PHI).^2))];
    MAD_SSSS = [MAD_SSSS; mean(mean(abs(X*A_SSSS - X*PHI)))];   MSD_SSSS = [MSD_SSSS; mean(mean((X*A_SSSS - X*PHI).^2))];
    MAD_BMS = [MAD_BMS; mean(mean(abs(X*A_BMS - X*PHI)))];      MSD_BMS = [MSD_BMS; mean(mean((X*A_BMS - X*PHI).^2))];
%     MAD_BSS = [MAD_BSS; mean(mean(abs(X*A_BSS - X*PHI)))];      MSD_BSS = [MSD_BSS; mean(mean((X*A_BSS - X*PHI).^2))];
    MAD_GLP = [MAD_GLP; mean(mean(abs(X*A_GLP - X*PHI)))];      MSD_GLP = [MSD_GLP; mean(mean((X*A_GLP - X*PHI).^2))];
    MAD_SSVS = [MAD_SSVS; mean(mean(abs(X*A_SSVS - X*PHI)))];   MSD_SSVS = [MSD_SSVS; mean(mean((X*A_SSVS - X*PHI).^2))];
    MAD_CC = [MAD_CC; mean(mean(abs(X*A_CC - X*PHI)))];         MSD_CC = [MSD_CC; mean(mean((X*A_CC - X*PHI).^2))];
    MAD_OLS = [MAD_OLS; mean(mean(abs(X*A_OLS - X*PHI)))];      MSD_OLS = [MSD_OLS; mean(mean((X*A_OLS - X*PHI).^2))];
    
    TRACE_BFCS = [TRACE_BFCS; trace(((PHI(:)'*a_BFCS)/(a_BFCS'*a_BFCS))*a_BFCS'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_SSSS = [TRACE_SSSS; trace(((PHI(:)'*a_SSSS)/(a_SSSS'*a_SSSS))*a_SSSS'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_BMS = [TRACE_BMS; trace(((PHI(:)'*a_BMS)/(a_BMS'*a_BMS))*a_BMS'*PHI(:))/trace(PHI(:)'*PHI(:))];
%     TRACE_BSS = [TRACE_BSS; trace(((PHI(:)'*a_BSS)/(a_BSS'*a_BSS))*a_BSS'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_GLP = [TRACE_GLP; trace(((PHI(:)'*a_GLP)/(a_GLP'*a_GLP))*a_GLP'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_SSVS = [TRACE_SSVS; trace(((PHI(:)'*a_SSVS)/(a_SSVS'*a_SSVS))*a_SSVS'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_CC = [TRACE_CC; trace(((PHI(:)'*a_CC)/(a_CC'*a_CC))*a_CC'*PHI(:))/trace(PHI(:)'*PHI(:))];
    TRACE_OLS = [TRACE_OLS; trace(((PHI(:)'*a_OLS)/(a_OLS'*a_OLS))*a_OLS'*PHI(:))/trace(PHI(:)'*PHI(:))];
end