function  [alpha_draws] = MCMC_PVAR_SSVS(Yraw,N,G,p,nsave,nburn,ntot,iter)


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

psi_xx = kron(sigma_inv,(X'*X));
PPA = (psi_xx)*alpha_OLS_vec;
Ftau_0 = (1/tau_0)^2;
Ftau_1 = (1/tau_1)^2;

% ===============| GIBBS SAMPLER
disp('Now running SSVS prior. Please wait...')
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
    DRD = diag(h_i');     
    Delta_alpha = inv(psi_xx + DRD);
    miu_alpha = Delta_alpha*(PPA);    
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