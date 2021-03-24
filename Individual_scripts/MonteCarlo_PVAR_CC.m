% PRIORS for Panel Vector autoregressions
% *************************************************************************
% CASE 4: Bayesian Mixture Shrinkage (BMS) prior
% *************************************************************************
% Written by Dimitris Korobilis, on 13/07/2011
% CORE, Universite Catholique de Louvain
% dikorobilis@googlemail.com
% *************************************************************************

clear all;
clc;
tic;


% Add path of random number generators
addpath('functions')

alpha_MCMC_MC = [];
sigma_MCMC_MC = [];

alpha_OLS_MC = [];
sigma_OLS_MC = [];

MAD = [];
MADOLS = [];

MSD = [];
MSDOLS = [];

TRACESTAT = [];
TRACESTATOLS = [];


for nMC = 1:1
    nMC

%---------------------------| USER INPUT |---------------------------------
% Gibbs-related preliminaries
nsave = 3000;           % Number of draws to save
nburn = 1000;           % Number of draws to discard
ntot = nsave + nburn;  % Number of total draws
iter = 1000;            % Print every "iter" iteration

% VAR specification
p = 1;    % Number of lags
N = 3;    % Number of cross sections
G = 2;    % Number of VAR variables for each cross-section

%----------------------------| END INPUT |---------------------------------
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

% Load data
[Yraw,PHI] = simpvardgp(100,N*G,p,PHI,PSI);

% Create VAR data matrices
[Traw, NG] = size(Yraw);
if NG ~= N*G; error('wrong specification of N and G'); end  % Check dimensions
Ylag = mlag2(Yraw,p);
n = p*NG*NG;          % total number of regression coefficients
k = p*NG;             % number of coefficients in each equation
X = Ylag(p+1:Traw,:); % VAR data (RHS) matrix on the original specification    
x = kron(eye(NG),X);  % VAR data (RHS) matrix on the SURE model
% Correct time-series dimension of Y due to taking lags
Y = Yraw(p+1:Traw,:); 
T = Traw-p;
y = Y(:);    % This is the final matrix y = vec()Y

[x_t,K] = create_RHS_noint(Yraw,NG,p,Traw);
y_t = Y;
yy = y_t'; yy = yy(:);

% Storage space for posterior samples
alpha_draws = zeros(nsave,n);
sigma_draws = zeros(nsave,NG,NG);

% =================| PRIORS |=================
% OLS quantities
alpha_OLS_vec = inv(x'*x)*(x'*y);
alpha_OLS_vec2 = inv(x_t'*x_t)*(x_t'*yy);
alpha_OLS_mat = inv(X'*X)*(X'*Y);
SSE = (Y - X*alpha_OLS_mat)'*(Y - X*alpha_OLS_mat);
sigma_OLS = SSE./(T-(k-1));

[KSI_ALL,K_state] = create_KSI(N,G,K);

% -----| Initialize other parameters
sigma = sigma_OLS;
sigma_inv = inv(sigma);
sigma_til = sigma;
alpha = alpha_OLS_vec;
W = kron(sigma_inv,eye(T));
index_kron = find(W~=0);
x_til = x*KSI_ALL;
ss2 = 1;%rand;

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
    
    % 1. Update alpha from Normal
    xSx = 0;xSy = 0;
    for i = 1:T
        xtemp = x_t((i-1)*N*G+1:i*N*G,:);
        xtemp2 = xtemp*KSI_ALL;
        xtil = (xtemp2'/((eye(N*G) + ss2*xtemp*xtemp')*sigma));
        xSx = xSx + xtil*xtemp2;
        xSy = xSy + xtil*y_t(i,:)';
    end
    theta_var = inv(0.25*eye(K_state) + xSx);
    theta_mean = theta_var*(xSy);
    theta = theta_mean + chol(theta_var)'*randn(K_state,1); % Draw vector of coefficients   

    alpha = KSI_ALL*theta;
    alpha_mat = reshape(alpha,k,NG);                  % Create matrix of coefficients 
    
    % 4. Update sigma2 from Inverse Wishart
    SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
    sigma_inv = wish(SSE+eye(NG),T);
    sigma = inv(sigma_inv);
    
    
    % Save draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        sigma_draws(irep-nburn,:,:) = sigma;
    end
    
end
clc;
toc;

alpha_MCMC_MC = [alpha_MCMC_MC, mean(alpha_draws,1)'];
sigma_MCMC_MC = [sigma_MCMC_MC, vec(mean(sigma_draws,1))];

alpha_OLS_MC = [alpha_OLS_MC, alpha_OLS_vec];
sigma_OLS_MC = [sigma_OLS_MC, vec(sigma_OLS)];

A = mean(alpha_draws,1)';

MAD = [MAD, mean(mean(abs(X*reshape(A,N*G,N*G) - X*PHI)))];
MADOLS = [MADOLS, mean(mean(abs(X*alpha_OLS_mat - X*PHI)))];

MSD = [MSD, mean(mean((X*reshape(A,N*G,N*G) - X*PHI).^2))];
MSDOLS = [MSDOLS, mean(mean((X*alpha_OLS_mat - X*PHI).^2))];

TRACESTAT = [TRACESTAT, trace(((PHI(:)'*A)/(A'*A))*A'*PHI(:))/trace(PHI(:)'*PHI(:))];
TRACESTATOLS = [TRACESTATOLS, trace(((PHI(:)'*alpha_OLS_vec)/(alpha_OLS_vec'*alpha_OLS_vec))*alpha_OLS_vec'*PHI(:))/trace(PHI(:)'*PHI(:))];
end
