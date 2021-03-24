function [KSI_ALL,K_state] = create_KSI(N,G,K)

% loadings matrices for time-varying coefficients
% based on Canova and Ciccarelli (2013), page 22
nfac = N*G;
KSI_1 = zeros(K,N*G);
for i = 1:nfac
   KSI_1((i-1)*N*G+1:i*N*G,i)=1;   % Common factor for coefficients of eahc of the NG VAR eqs.
end
i_vec = zeros(N*G,N);
for i = 1:N
    i_vec((i-1)*G+1:i*G,i) = 1;
end
j_vec = zeros(N*G,G);
for j = 1:G
   j_vec(j:G:end,j) = 1; 
end
KSI_2 = zeros(K-N*G,N);
for i = 1:N
   KSI_2((i-1)*N*G*G+1:i*N*G*G,i) = repmat(i_vec(:,i),G,1);  % Common factor for coefficients of each country (N)
end
index_vec = zeros(N*G*G,G);
for j = 1:G
   index_vec((j-1)*N*G+1:j*N*G,j) = j_vec(:,j);
end

KSI_5 = zeros(K-N*G,N*G);
counter = 0;
for ii = 1:G
   index = (N*G+1)*(ii-1)+1:N*G*G+G:N*G*N*G;
   for jj = 1:N
       counter  = counter + 1;
       KSI_5(index(jj),counter) = 1;
   end
end


KSI_3 = zeros(K-N*G,G);
KSI_3 = repmat(index_vec,N,1); % Common factor for coefficients of each variable (G)  

KSI_4 = [ones(K,1)];   % Common factor for ALL VAR coefficients, not intercepts

KSI_ALL = [KSI_2, KSI_3, KSI_4];
K_state = size(KSI_ALL,2);