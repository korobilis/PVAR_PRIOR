% PRIORS for Panel Vector autoregressions
% *************************************************************************
% CASE 1: Stochastic Search Specification Selection (S^4) prior
% *************************************************************************
% Written by Dimitris Korobilis, March 2014
% University of Glasgow
% dikorobilis@googlemail.com
% *************************************************************************

clear all; close all; clc;

% Add path of random number generators
addpath('functions')

gamma_MC = [];

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

for nMC = 1:100
    nMC

%---------------------------| USER INPUT |---------------------------------
% Gibbs-related preliminaries
nsave = 3000;           % Number of draws to save
nburn = 1000;           % Number of draws to discard
ntot = nsave + nburn;  % Number of total draws
iter = 1000;            % Print every "iter" iteration

% VAR specification
p = 1;    % Number of lags
N = 6;    % Number of cross sections
G = 4;    % Number of VAR variables for each cross-section

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
T=Traw-p;
y = Y(:);    % This is the final matrix y = vec()Y

[x_t,~] = create_RHS_noint(Yraw,NG,p,Traw);
y_t = Y;
yy = y_t'; yy = yy(:);

% ====| Set priors
% OLS quantities
alpha_OLS_vec = inv(x'*x)*(x'*y);
alpha_OLS_vec2 = inv(x_t'*x_t)*(x_t'*yy);
alpha_OLS_mat = inv(X'*X)*(X'*Y);
SSE = (Y - X*alpha_OLS_mat)'*(Y - X*alpha_OLS_mat);
sigma_OLS = SSE./(T-(k-1));
sigma = sigma_OLS;
sigma_inv = inv(sigma);

alpha = alpha_OLS_vec;
alpha_CS = cell(p,1);

% alpha ~ N(0,DD), where D = diag(tau_i)
tau_0 = .001;
tau_1 = 4;

% gamma_j ~ Bernoulli(1,p_j)
p_j = 0.5*ones(n,1);

% Initialize parameters
gamma = ones(n,1);
h_i = zeros(n,1);

% Create storage matrices for posteriors
alpha_draws = zeros(nsave,n);
sigma_draws = zeros(nsave,NG,NG);
gamma_draws = zeros(nsave,n);

Ftau_0 = (1/tau_0)^2;
Ftau_1 = (1/tau_1)^2;

tic;
% ===============| GIBBS SAMPLER
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
   
    %------------------------------------------------------
    % STEP 1: Update VAR coefficients alpha from Normal
    %------------------------------------------------------
    for kk = 1:n
        if gamma(kk,1) == 0
           h_i(kk,1) = Ftau_0;
        elseif gamma(kk,1) == 1
           h_i(kk,1) = Ftau_1;
        end       
    end        
    D = diag(h_i'); % Create D. Here D=diag(h_i) will also do
%     DRD = D*D;   % Prior covariance matrix for Phi_m
    psi_xx = kron(sigma_inv,(X'*X));
    Delta_alpha = inv(psi_xx + D);
    miu_alpha = Delta_alpha*((psi_xx)*alpha_OLS_vec);    
    alpha = miu_alpha + chol(Delta_alpha)'*randn(n,1);
    alpha_mat = reshape(alpha,k,NG);
    
    %----------------------------------------------------------------------
    % STEP 2: Update restriction indexes of alpha from Bernoulli
    %----------------------------------------------------------------------    
    u_i1 = mvnpdf(alpha,0,tau_0).*p_j;           
    u_i2 = mvnpdf(alpha,0,tau_1).*(1- p_j);
    gst = u_i2./(u_i1 + u_i2);
    gamma = bernoullimrnd(n,gst);       
    
    %------------------------------------------------------------------
    % STEP 3: Update VAR covariance matrix and SI restriction indexes
    %------------------------------------------------------------------
    SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
    sigma_inv = wish(SSE+eye(NG),T);
    sigma = inv(sigma_inv);
  
    
    % ========| Save post-burn-in draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        sigma_draws(irep-nburn,:,:) = sigma;
        gamma_draws(irep-nburn,:) = gamma;
    end
end
clc;
toc;

gamma_MC = [gamma_MC, mean(gamma_draws,1)'];

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

toc;