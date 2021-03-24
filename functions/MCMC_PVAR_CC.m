function  [alpha_draws] = MCMC_PVAR_CC(Yraw,N,G,p,nsave,nburn,ntot,iter)


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

xSx = 0;
xSy = 0;
for i = 1:T
    xtemp = x_t((i-1)*N*G+1:i*N*G,:);
    xtemp2 = xtemp*KSI_ALL;
    xtil = (xtemp2'/((eye(N*G) + ss2*xtemp*xtemp')*sigma));
    xSx = xSx + xtil*xtemp2;
    xSy = xSy + xtil*y_t(i,:)';
end
theta_var = inv(0.25*eye(K_state) + xSx);
theta_mean = theta_var*(xSy);
CTV = chol(theta_var)';
theta = repmat(theta_mean,1,ntot) + CTV*randn(K_state,ntot); % Draw vector of coefficients 

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
disp('Now running CC prior. Please wait...')
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end

    alpha = KSI_ALL*theta;
%     alpha_mat = reshape(alpha,k,NG);                  % Create matrix of coefficients 
    
%     % 4. Update sigma2 from Inverse Wishart
%     SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
%     sigma_inv = wish(SSE+eye(NG),T);
%     sigma = inv(sigma_inv);
    
    
    % Save draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha(:,irep);
        sigma_draws(irep-nburn,:,:) = sigma;
    end
    
end

clc;