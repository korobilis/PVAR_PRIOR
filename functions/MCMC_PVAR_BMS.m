function  [alpha_draws] = MCMC_PVAR_BMS(Yraw,N,G,p,nsave,nburn,ntot,iter)

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
lambda_draws = zeros(nsave,n);
prob_draws = zeros(nsave,n);
sigma_draws = zeros(nsave,NG,NG);

% =================| PRIORS |=================
% OLS quantities
alpha_OLS_vec = inv(x'*x)*(x'*y);
alpha_OLS_vec2 = inv(x_t'*x_t)*(x_t'*yy);
alpha_OLS_mat = inv(X'*X)*(X'*Y);
SSE = (Y - X*alpha_OLS_mat)'*(Y - X*alpha_OLS_mat);
sigma_OLS = SSE./(T-(k-1));

% ----| Define hyperparameters
% 1. Hyperparameters of Gamma(a0,b0)
a0 = 100; b0 = .1;
% 2. Hyperparameters of Gamma(a1,b1)
a1 = 2; b1 = 1;
% 3. Hyperparameters of N(c,d)
c = 0; d = 4;
% 4. Hyperparameter of DP(theta*D0)
theta = 1;

% -----| Initialize other parameters
clus = randperm(3)';
p_star = max(clus);
miu = 0*ones(n,1);
LAMBDA = 4*eye(n);
iLAMBDA = inv(LAMBDA);
lambda = 1*ones(n,1);
sigma = sigma_OLS;
sigma_inv = inv(sigma);
alpha = alpha_OLS_vec;
W = kron(sigma_inv,eye(T));
x_star = x'*W;
xSy = x_star*y;
xSx = x_star*x;
index_kron = find(W~=0);
U = zeros(n,1);
prob_incl = zeros(n,1);

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
disp('Now running BMS prior. Please wait...')
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
    
    % 1. Update alpha from Normal
    alpha_var = inv(iLAMBDA + xSx);
    alpha_mean = alpha_var*(xSy + iLAMBDA*miu);
    alpha = alpha_mean + chol(alpha_var)'*randn(n,1); % Draw vector of coefficients   
    alpha_mat = reshape(alpha,k,NG);                  % Create matrix of coefficients    
    
    % 3. Update prior location and scale parameters
%     p_star = max(clus); % define total number of bins
    for kk = 1:p_star
        if kk == 1            
            m(kk,1) = sum(clus==kk);  % Number of predictors that fall in the 1st bin
            % 3a. Update the pair (miu_bar[1],tau_bar[1])
            if m(kk,1) ~= 0                
                miu_bar(kk,1) = 0;
                lambda_bar(kk,1) = 1e+10;%Draw_Gamma( m(kk,1)/2 + a0, sum(((alpha(clus==kk) - miu(clus==kk)).^2))/2 + b0 );
            elseif m(kk,1) == 0        
                miu_bar(kk,1) = 0;
                lambda_bar(kk,1) = Draw_Gamma(a0,b0);
            end
            % 3b. Update pi[1]= V_t
            V(kk,1) = Draw_Beta(m(kk,1) + 10, n - sum(m(1:kk,1)) + theta);
            pi(kk,1) = V(kk,1);
        else
            m(kk,1) = sum(clus==kk);
            % 3a. Update the pairs
            % (miu_bar[2],tau_bar[2]),...,(miu_bar[p_star],tau_bar[p_star])
            if m(kk,1) ~= 0 
                V_miu_bar_t = inv(1/d + sum(1./lambda(clus==kk)));
                E_miu_bar_t = V_miu_bar_t*(c/d + sum(alpha(clus==kk)./lambda(clus==kk)));
                miu_bar(kk,1) = Draw_Normal(E_miu_bar_t,V_miu_bar_t);
                lambda_bar(kk,1) = Draw_Gamma( m(kk,1)/2 + a1, sum(((alpha(clus==kk) - miu(clus==kk)).^2))/2  + b1 );
            elseif m(kk,1) == 0        
                miu_bar(kk,1) = Draw_Normal(c,d);
                lambda_bar(kk,1) = Draw_Gamma(a1,b1);
            end       
            % 3b. Update pi_t from prod(1 - pi[h])V_t for h=1,...,kk-1
            V(kk,1) = Draw_Beta(m(kk,1) + 1, n - sum(m(1:kk,1)) + theta);
            pi(kk,1) = prod( (1 - pi(1:kk-1,1)) )*V(kk,1);
        end
    end
    
    % Make sure pi is a properly defined probability density
    pi = pi./sum(pi);
    
%     U = rand(n,1);
%     for j = 1:n        
%         q = pi.*(normpdf(alpha(j,1),miu_bar,1./lambda_bar));
%         q = q./sum(q);
%         clus(j,1) = min(find(cumsum(q)>=U(j,1)));         %#ok<MXFND>
%         prob_incl(j,1) = q(1,1);
%     end

    U = rand(n,1);
    q = repmat(pi',n,1).*normpdf(repmat(alpha,1,p_star),repmat(miu_bar',n,1), repmat(lambda,1,p_star)) + 1e-10;
    q = q./repmat(sum(q,2),1,p_star);
    clus = p_star - sum(repmat(U,1,p_star)<cumsum(q,2),2)+1;
    pp = q(1,:)';
    prob_incl = 1 - pp(clus);
    
    % Update the estimate of the prior mean
    miu = miu_bar(clus);
    iLAMBDA = diag(lambda_bar(clus));
    %lambda = 1./(lambda_bar(clus) + 1e-10);    
    %LAMBDA = diag(lambda);
    
    SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
    sigma_inv = wish(SSE+eye(NG),T);
    sigma = inv(sigma_inv);
    
    % Save draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        clus_draws(irep-nburn,:) = clus;
        miu_draws(irep-nburn,:) = miu;
        lambda_draws(irep-nburn,:) = lambda;
        prob_draws(irep-nburn,:) = prob_incl;
        sigma_draws(irep-nburn,:,:) = sigma;
    end
    
end


clc;