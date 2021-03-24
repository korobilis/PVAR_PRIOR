function [y,PHI] = simpvardgp(T,NG,L,PHI,PSI)
%--------------------------------------------------------------------------
%   PURPOSE:
%      Get matrix of Y generated from a VAR model
%--------------------------------------------------------------------------
%   INPUTS:
%     T     - Number of observations (rows of Y)
%     N     - Number of series (columns of Y)
%     L     - Number of lags
%
%   OUTPUT:
%     y     - [T x N] matrix generated from VAR(L) model
% -------------------------------------------------------------------------

randn('seed',sum(100*clock));
rand('seed',sum(100*clock));
%-----------------------PRELIMINARIES--------------------
if nargin == 0
    T = 50;            %Number of time series observations (T)
    N = 3;             %Number of countries (N)
    G = 2;             %Number of macro variables (G)
    NG = N*G;
    L = 1;             %Lag order

%     PHI = [0.3  0;
%            0    0.3];
%        
%     PSI = [1   0.5;
%            0   1 ];

%     PHI = [0.7  0.2    0 ;
%            0    0.7  0 ;
%            0    0.4    0.7];
%        
%     PSI = [1   0.5 0.5;
%            0   1   0.5;
%            0   0   1;];
      
%     PHI = [0.8  0    0.5  0    0    0;
%            0    0.8  0    0.5  0    0;
%            0    0    0.8  0    0    0;
%            0    0    0    0.8  0    0;
%            0.6  0    0    0    0.5  0.3;
%            0    0.6  0    0   -0.4  0.5];

   PHI = diag(0.4*rand(NG,1)+0.3) + diag(rand(NG-2,1) - 0.5,2) + diag(rand(NG-2,1) - 0.5,-2)  + diag(rand(NG-4,1) - 0.5,4) + diag(rand(NG-4,1) - 0.5,-4) ;
    
    
    PSI = [1   0.5 0.5 0.5 0.5 0.5;
           0   1   0   0   0   0;
           0   0   1   0   0   0;
           0   0   0   1   0   0;
           0   0   0   0   1   0;
           0   0   0   0   0   1];
       
       
%     PHI = [0.6 -0.5  0    0    0.9  0    0    0;
%            0.4  0.6  0    0    0    0.9  0    0;
%            0    0    0.7  0    0    0    0    0;
%            0    0    0    0.7  0    0    0    0;
%            0    0    0    0    0.6 -0.5  0    0;
%            0    0    0    0    0.4  0.6  0    0;
%            0    0    0.8  0.5  0    0    0.7  0;
%            0    0    0.5  0.8  0    0    0    0.7];
%         
%     PSI = [1   0.5 0.5 0.5 0.5 0.5 0.5 0.5;
%            0   1   0.5 0.5 0.5 0.5 0.5 0.5;
%            0   0   1   0.5 0.5 0.5 0.5 0.5;
%            0   0   0   1   0.5 0.5 0.5 0.5;
%            0   0   0   0   1   0.5 0.5 0.5;
%            0   0   0   0   0   1   0.5 0.5;
%            0   0   0   0   0   0   1   0.5;
%            0   0   0   0   0   0   0   1];
     
%             PSI = eye(NG); 
end

sigma = inv(PSI*PSI');
%----------------------GENERATE--------------------------
% Set storage in memory for y
% First L rows are created randomly and are used as 
% starting (initial) values 
y =[rand(L,NG) ; zeros(T,NG)];

% Now generate Y from VAR (L,PHI,PSI)
for nn = L+1:T+L    
    u = chol(sigma)'*randn(NG,1);
    y(nn,:) = y(nn-1,:)*PHI' + u';
end
y = y(L+1:end,:);
PHI=PHI';