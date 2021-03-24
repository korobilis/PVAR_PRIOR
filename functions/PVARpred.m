
function [Y_pred] = PVARpred(Y,X,alpha_draws,sigma_OLS,T,M,G,N,p,h,nsave)

% Forecasting (predictive simulation)
Y_pred = zeros(nsave,N,h);   
X_fore = [Y(T,:) X(T,1:M*(p-1))];
for irep = 1:nsave       
    A = reshape(alpha_draws(irep,:)',N*G,N*G*p);
    % Forecast of T+1 conditional on data at time T
    Y_hat = X_fore*A + randn(1,M)*chol(sigma_OLS);
    Y_pred(irep,:,1) = Y_hat(1:N);                                   
    for i = 1:h-1  % Predict T+2, T+3 until T+h
        if i <= p                       
            X_new_temp = [Y_hat X_fore(:,1:M*(p-i))];
            % This gives the forecast T+i for i=1,..,p
            Y_temp = X_new_temp*A + randn(1,M)*chol(sigma_OLS);
            Y_pred(irep,:,i+1) = Y_temp(1:N);                
            Y_hat = [Y_hat Y_temp];
        else
            X_new_temp = Y_hat(:,1:M*p);
            Y_temp = X_new_temp*A + randn(1,M)*chol(sigma_OLS);
            Y_pred(irep,:,i+1) = Y_temp(1:N);
            Y_hat = [Y_hat Y_temp];
        end
    end
end