function  [alpha_draws] = MCMC_PVAR_GLP(Yraw,N,G,p,nsave,nburn,ntot,iter)


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

disp('Now running GLP prior. Please wait...')
r = bvarGLP(Yraw,p,'Ndraws',ntot,'Ndrawsdiscard',nburn);

% Create storage matrices for posteriors
alpha_d = r.mcmc.beta(2:end,:,:);
alpha_draws = reshape(alpha_d,NG^2,nsave)';
sigma_draws = permute(r.mcmc.sigma,[3,1,2]);

clc;