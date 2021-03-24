function  [alpha_draws] = MCMC_PVAR_SSSS(Yraw,N,G,p,nsave,nburn,ntot,iter)

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

% ====| Examine restrictions
% All VAR coefficients are in a single NG x NG x p column vector. With the
% code below I am trying to index the groups of elements of this huge vector 
% which correspond to C-S Heterogeneity, and DI restrictions. 
index_restriction = zeros(G,G,N*N);
index_var=zeros(G,G,N);   
for i_country = 1:N           
    index_temp = (i_country-1)*G+1:i_country*G;
    for i_variable = 1:G
        index_var(i_variable,:,i_country) = index_temp + (i_variable-1)*NG;
    end
end
% Note that for p=1 lags we have a single NGxNG coefficient matrix and an NGxNG covariance matrix, 
% so the index_restriction variable can be used to obtain both DI and CS restrictions, as well as SI restrictions
for i_country = 1:N
    index_restriction(:,:,(i_country-1)*N+1:i_country*N) = index_var + (i_country-1)*N*G*G;
end

% Now take indexes of the position of each restriction.
% Position of C-S heterogeneity restrictions
CS_index = 1:N+1:N*N;
% Position of DI interdependency restrictions
DI_index = 1:N*N; 
DI_index(CS_index) = [];   

n_CS = N*(N-1)/2;    % Number of C-S restrictions
n_DI = length(DI_index);   % Number of DI restrictions

index_CS = index_restriction(:,:,CS_index); % To be used to obtain index of C-S restrictions
index_restr_DI = index_restriction(:,:,DI_index); % Index of DI restrictions

% The "index_CS" variable indexes the matrices A_{1}^{i} of country i.
% We need to test pairs of restrictions of the form A_{1}^{i} = A_{1}^{j}
% in order to test homogeneity of countries i and j.
% a) First create pairs of countries:
pairs_index = combntns(1:N,2);   % Index of pairs
% b) Second obtain index of CS restrictions. For each pair of countries, we
% are testing the equivalence of the GxG matrices A_{1}^{i}.
index_restr_CS = cell(n_CS,1);
index_rest_CS_all = zeros(G*G*n_CS,2);
for ir = 1:n_CS
    temp =  index_CS(:,:,pairs_index(ir,:));
    temp1 = temp(:,:,1); temp2 = temp(:,:,2);
    index_restr_CS{ir,1} =  [temp1(:),temp2(:)];
    index_rest_CS_all((ir-1)*G*G+1:ir*G*G,:) = [temp1(:),temp2(:)];
end

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
tau_0 = .01;
tau_1 = 4;

ksi_0 = .01;
ksi_1 = 4;

% gamma_j ~ Bernoulli(1,p_j)
p_j_DI = 0.5*ones(n_DI,1);
p_j_CS = 0.5*ones(n_CS,1);

% Initialize parameters
gamma_DI = ones(n_DI,1);
gamma_CS = ones(n_CS,1);
GAMMA2 = cell(n_CS,1);
for i = 1:n_CS; GAMMA2{i,1} = speye(n); end
GAMMA = speye(n);
for i = 1:n_CS
    GAMMA = GAMMA*GAMMA2{i,1};
end

% Create storage matrices for posteriors
alpha_draws = zeros(nsave,n);
sigma_draws = zeros(nsave,NG,NG);
gammaDI_draws = zeros(nsave,n_DI);
gammaCS_draws = zeros(nsave,n_CS);

psi_xx = kron(sigma_inv,(X'*X));
PPA = (psi_xx)*alpha_OLS_vec;
Ftau_0 = (1/tau_0)^2;
Ftau_1 = (1/tau_1)^2;
% ===============| GIBBS SAMPLER
disp('Now running SSSS prior. Please wait...')
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
   
    %------------------------------------------------------
    % STEP 1: Update VAR coefficients alpha from Normal
    %------------------------------------------------------
    h_i = (1./(tau_1*tau_1))*ones(n,1);   % h_i is tau_0 if gamma=0 and tau_1 if gamma=1
    for kk = 1:n_DI
        ind_temp = index_restr_DI(:,:,kk)';
        if gamma_DI(kk,1) == 0
           h_i(ind_temp(:)) = Ftau_0;
        elseif gamma_DI(kk,1) == 1
           h_i(ind_temp(:)) = Ftau_1;
        end       
    end        
    DRD = diag(h_i');  
    Delta_alpha = inv(psi_xx + DRD);
    miu_alpha = Delta_alpha*(PPA);    
    alpha = miu_alpha + chol(Delta_alpha)'*randn(n,1);
    alpha = GAMMA*alpha;
    alpha_mat = reshape(alpha,k,NG);
    
    %----------------------------------------------------------------------
    % STEP 2: Update DI and CS restriction indexes of alpha from Bernoulli
    %---------------------------------------------------------------------- 
    % 1) Examine dynamic interdependencies (DI) 
%         p_j_DI = repmat(betarnd(1 + sum(gamma_DI==1),1 + sum(gamma_DI~=1)),n_DI,1);
        for kk = 1:n_DI       
            ind_temp = index_restr_DI(:,:,kk)';
            u_i1 = mvnpdf(alpha(ind_temp(:)),zeros(G*G,1),tau_0*eye(G*G))*p_j_DI(kk);
            u_i2 = mvnpdf(alpha(ind_temp(:)),zeros(G*G,1),tau_1*eye(G*G))*(1- p_j_DI(kk));
            gst = u_i2./(u_i1 + u_i2);
            gamma_DI(kk,1) = bernoullirnd(gst);   
        end
    
    % 2) Examine cross-sectional (CS) heterogeneities
%         p_j_CS = repmat(betarnd(1 + sum(gamma_CS~=1),1 + sum(gamma_CS==1)),n_CS,1);
        % update gamma, one at a time (their prior is independent)
        for kk = 1:n_CS       
            ind_temp = index_restr_CS{kk,1};
            u_i1 = mvnpdf(alpha(ind_temp(:,1)),alpha(ind_temp(:,2)),ksi_0*eye(G*G))*p_j_CS(kk);
            u_i2 = mvnpdf(alpha(ind_temp(:,1)),alpha(ind_temp(:,2)),ksi_1*eye(G*G))*(1- p_j_CS(kk));
            gst = u_i2./(u_i1 + u_i2);
            gamma_CS(kk,1) = bernoullirnd(gst);
            if gamma_CS(kk) == 0
                for d_G = 1:G*G   
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,1)) = 0;
                    GAMMA2{kk,1}(ind_temp(d_G,2),ind_temp(d_G,2)) = 1;                   
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,2)) = 1;
                end
            else
                for d_G = 1:G*G
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,1)) = 1;
                    GAMMA2{kk,1}(ind_temp(d_G,2),ind_temp(d_G,2)) = 1;
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,2)) = 0;
                end
            end
            GAMMA = speye(n);
            for i = 1:n_CS
                GAMMA = GAMMA*GAMMA2{i,1};
            end
        end
    
%     %------------------------------------------------------------------
%     % STEP 3: Update VAR covariance matrix and SI restriction indexes
%     %------------------------------------------------------------------
%     SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
%     sigma_inv = wish(SSE+eye(NG),T);
%     sigma = inv(sigma_inv);
  
    
    % ========| Save post-burn-in draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        sigma_draws(irep-nburn,:,:) = sigma;
        gammaDI_draws(irep-nburn,:) = gamma_DI;        
        gammaCS_draws(irep-nburn,:) = gamma_CS;
    end
end
clc;