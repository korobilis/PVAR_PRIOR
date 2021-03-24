% PRIORS for Panel Vector autoregressions
% *************************************************************************
% CASE 3: Bayesian Semiparametric Shrinkage (BSS) prior
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

prob_draws_MC = [];

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

[x_t,~] = create_RHS_noint(Yraw,NG,p,Traw);
y_t = Y;
yy = y_t'; yy = yy(:);

% Storage space for posterior samples
alpha_draws = zeros(nsave,n);
clus_draws = zeros(nsave,n);  
miu_draws = zeros(nsave,n);
tau_draws = zeros(nsave,n);
lambda_draws = zeros(nsave,n);
prob_draws = zeros(nsave,1);
sigma_draws = zeros(nsave,NG,NG);

% =================| PRIORS |=================
% OLS quantities
alpha_OLS_vec = inv(x'*x)*(x'*y);
alpha_OLS_vec2 = inv(x_t'*x_t)*(x_t'*yy);
alpha_OLS_mat = inv(X'*X)*(X'*Y);
SSE = (Y - X*alpha_OLS_mat)'*(Y - X*alpha_OLS_mat);
sigma_OLS = SSE./(T-(k-1));


% ========| PRIOR:
% Choose hyperparameters
mu0 = 0;             %Prior mean for Betas  
P0 = 1/4;            %Prior Precision for Betas
%Prior for lambda ~ Gamma(a1,a2)
a1 = 2; 
a2 = 1;
al0 = 1;
%Prior for pi ~ Beta(c1,c2)
c1 = 100;
c2 = 1;
pi0 = 0.5;           % probability of null cluster membership for gamma
%Prior for phi ~ iGamma(b1,b2)
b1 = 5;
b2 = 1;

% ========| INITIALIZE COEFFICIENTS AND STORAGE SPACE
% initialize unknowns related to clustering 
kG = 10;                        % number of clusters in gamma
Gam = [0;rand(kG-1,1)];                       % vector of unique values of gamma
Tht = Gam(2:kG)'; %#ok<NASGU> % regression parameters in current configuration
Sg = ones(n,1);
gam = Gam(Sg);
Sgsum=(repmat(Sg(:,1),[1 kG])-repmat(1:kG,[n 1])==0);
pg = ones(1,n)*Sgsum;
xt = zeros(T*NG,kG-1); 
for m = 1:kG-1
    xt(:,m) = x*(Sg==m);    
end
Tht = Gam(2:kG)';

sigma = sigma_OLS;
sigma_inv = inv(sigma);
alpha = alpha_OLS_vec;
W = kron(sigma_inv,eye(T));
index_kron = find(W~=0);


%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
    
    % Create the VAR covariance matrix in the SURE model
    % NOTE: The code below is a faster way of calculating the kronecker
    % product at each iternation...
    Wtemp = zeros(T*NG*NG,1);
    for i = 1:NG
        Wtemp((i-1)*NG*T+1:i*NG*T) = repmat(sigma_inv(:,i),T,1);
    end
    W(index_kron) = Wtemp;
%     W = kron(sigma_inv,eye(T));

    % ----STEP 2: Sample parameters conditional on configuration into clusters       
    if kG>1
        Tht0 = mu0*ones(1,kG-1)';
        PTht0 = diag(P0*ones(1,kG-1));
        xt_star = xt'*W;
        VTht = inv(PTht0 + xt_star*xt);
        ETht = VTht*(PTht0*Tht0 + (xt_star*y));  %#ok<MINV>
        Tht = mvnrnd(ETht,VTht,1)';
    end
    Gam = [0 Tht(1:kG-1)']';   
    gam = Gam(Sg);
  
    % ----STEP 3: Sample configuration of coefficients into clusters
    % a) Update weights of each component
    A = zeros(n,kG+1);  % first element is new cluster, 2nd is zero cluster
    A(:,1) = (1-pi0)*(al0./(al0+n-pg(1)+(Sg==1)-1)); 
    A(:,2) = pi0*ones(n,1); 
    for h = 3:kG+1
        A(:,h) = (1-pi0)*(pg(h-1)-(Sg==h-1))./(al0+n-pg(1)+(Sg==1)-1); 
    end
    % b) Update mean and variance for new components and obtain conditional probs
    Vg = zeros(n,1);
    Eg = zeros(n,1);
    xW = x'*W;
    for h = 1:n
        if kG>1
            %ysh = y - x*gam + x(:,h)*gam(h);
            ysh = y - xt*Tht + x(:,h)*gam(h);
        else
            ysh = y + x(:,h)*gam(h);
        end
        
        Vg(h) = 1/(P0 + xW(h,:)*x(:,h)); 
        Eg(h) = Vg(h)*(xW(h,:)*ysh); 
        A(h,1) = log(A(h,1)) + 0.5*(log(Vg(h)) + log(P0) + Eg(h)^2/Vg(h) - P0*mu0^2);  
        for l = 1:kG
            xG = x(:,h)*Gam(l);
            A(h,l+1) = log(A(h,l+1)) + sum(diag(sigma_inv))*( (xG)'*ysh - ((xG)'*(xG))/2 ); 
        end
        A(h,:) = exp(A(h,:) - max(A(h,:)));    % To avoid computational problems
    end
    A = A./repmat(A*ones(kG+1,1),[1 kG+1]);   % normalize each row to add to one
    
    % c) Update configuration for Gamma
    uni = unifrnd(0,1,[n 1]); 
    i1 = uni<A(:,1);
    if sum(i1)>0
        gam(i1) = normrnd(Eg(i1),Vg(i1).^(0.5)); 
    end
    % if one of existing clusters assign appropriate value
    for l = 1:kG
        gam(uni>A(:,1:l)*ones(l,1) & uni<=A(:,1:l+1)*ones(l+1,1)) = Gam(l); 
    end
    % d) Redefine Gam, Sg, kG, pg appropriately
    Gam = [0 unique(gam(gam~=0))']';
    kG = min(numel(Gam),3);%length(Gam);
    for l = 1:kG
        Sg(gam==Gam(l))=l;
    end
    Sgsum=(repmat(Sg(:,1),[1 kG])-repmat(1:kG,[n 1])==0);
    pg = ones(1,n)*Sgsum;
    xt = zeros(T*NG,kG-1);                    % define modified design matrix
    for m = 1:kG-1       
        xt(:,m) = x*(Sg==m);
    end
    Tht = Gam(2:kG);
    
    % ----STEP 4: Sample point mass probability
    pi0 = betarnd(c1 + sum(Sg==1),c2 + sum(Sg~=1));
       
    % ----STEP 5: Sample DP precision
    if pg(1)<n       
        xi = betarnd(al0+1,n-pg(1));
        pxi = (a1 + kG - (pg(1)>0) - 1)/(a1 + kG - (pg(1)>0) - 1 + (n-pg(1))*(a2 - log(xi))); 
        al0 = pxi*gamrnd(a1 + kG - (pg(1)>0),1/(a2-log(xi))) + ... 
            (1-pxi)*gamrnd(a1 + kG - (pg(1)>0) - 1,1/(a2-log(xi)));       
    else
        al0 = gamrnd(a1,1/a2);
    end
    
%     % ----STEP 6: Sample precision in base distribution   
%     if kG>1
%         ga = b1+(kG-1)/2;
%         gb = ((Gam(2:kG)-mu0*ones(1,kG-1)')'*(Gam(2:kG)-mu0*ones(1,kG-1)'))/2+1/b2;
%         P0 = gamrnd(ga,1/gb);
%     end
        
    % 4. Update sigma2 from Inverse Wishart
    alpha = gam;
    alpha_mat = reshape(alpha,k,NG);
    SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);   
    sigma_inv = wish(SSE+eye(NG),T+NG+1);
    sigma = inv(sigma_inv);
    
    % Save draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        clus_draws(irep-nburn,:) = Sg;
        prob_draws(irep-nburn,:) = pi0;
        sigma_draws(irep-nburn,:,:) = sigma;
    end
    
end
clc;
toc;


alpha_MCMC_MC = [alpha_MCMC_MC, mean(alpha_draws,1)'];
sigma_MCMC_MC = [sigma_MCMC_MC, vec(mean(sigma_draws,1))];

alpha_OLS_MC = [alpha_OLS_MC, alpha_OLS_vec];
sigma_OLS_MC = [sigma_OLS_MC, vec(sigma_OLS)];

prob_draws_MC = [prob_draws_MC, mean(prob_draws,1)'];

A = mean(alpha_draws,1)';

MAD = [MAD, mean(mean(abs(X*reshape(A,N*G,N*G) - X*PHI)))];
MADOLS = [MADOLS, mean(mean(abs(X*alpha_OLS_mat - X*PHI)))];

MSD = [MSD, mean(mean((X*reshape(A,N*G,N*G) - X*PHI).^2))];
MSDOLS = [MSDOLS, mean(mean((X*alpha_OLS_mat - X*PHI).^2))];

TRACESTAT = [TRACESTAT, trace(((PHI(:)'*A)/(A'*A))*A'*PHI(:))/trace(PHI(:)'*PHI(:))];
TRACESTATOLS = [TRACESTATOLS, trace(((PHI(:)'*alpha_OLS_vec)/(alpha_OLS_vec'*alpha_OLS_vec))*alpha_OLS_vec'*PHI(:))/trace(PHI(:)'*PHI(:))];

end
