function [D1, D2] = CDL(X1,X2,opts)

% A computationally efficient algorithm for learning a pair of coupled dictionaries {D1,D2} corresponding to the correlated joint dataset
% {X1, X2} so that D1*A = X1 and D2*A = X2. where A is the common sparse represenation matrix
%
% Optimization problem:
%  {D1, D2} = argmin_{D1,D2,A}   omega*||D1*A-X1||_F^2  +  (1-omega)*||D2*A-X2||_F^2
%               s.t. ||A_i||_0 < k , i, 1,...,N and
%                    ||D1_t||_2 = 1 ,||D2_t||_2 = 1 , t = 1,...,K
%
% Input variables:
%   opts.omage: tuning parameter (default: 0.5)
%   opts.K:    number of atoms in dictionaries (default: 4*max([size(X1,1),size(X1,1)])) 
%   opts.k:    maximum number of nonzero entries in columns of A  (default: max([size(X1,1),size(X1,1)])/4)
%   opts.Nit:  number of CDL iterations (default: 10) 
%   opts.eps:  approximation accuracy threshold (default: 1e-4)
%   opts.D1 and opts.D1:  initial dictionaries (default: DCT dictionaries)
%   opts.DCatom  ('true' or 'false') if true, keeps DC atom unchanged (default: true)

% Refrence:
%   F. G. Veshki and S. A. Vorobyov, "An Efficient Coupled Dictionary Learning Method," in IEEE Signal Processing Letters, vol. 26, no. 10,
%   pp. 1441-1445, Oct. 2019, doi: 10.1109/LSP.2019.2934045.
%
% (the mexOMP algorithm used for sparse approximation is taken from SPAMS toolbox (see mexOMP.m file))
%%
if nargin>1
    [m1,m2,K,D1,D2,Nit,omega,print,remMean,DCatom,param] = Parse_input(X1,X2,opts);
else
    error('Inputs are not not enough!')
end

if remMean==true
X1 = X1-mean(X1);
X2 = X2-mean(X2);
end

if DCatom == true
    k0 = 2;
    D1(:,1) = orth(ones(m1,1));
    D2(:,1) = orth(ones(m2,1));
else
    k0 = 1;
end

tic
for j = 1: Nit
    
    
    Xj = [sqrt(omega)*X1; sqrt(1-omega)*X2]; % scaled joint input
    Dj = [sqrt(omega)*D1; sqrt(1-omega)*D2]; % scaled joint dictionary
    
    A = mexOMP(Xj,Dj,param); % sparse coding phase
    
    for i = k0:K % dictionary update phase
        
        inds = find(A(i,:));
        if ~isempty(inds)
            Dt = D1;
            Dt(:,i) = 0;
            E_i_1 = X1(:,inds) - Dt*A(:,inds);
            Dt = D2;
            Dt(:,i) = 0;
            E_i_2 = X2(:,inds) - Dt*A(:,inds);
            
            D1(:,i) = E_i_1 * A(i,inds)';
            D2(:,i) = E_i_2 * A(i,inds)';
            
            D1(:,i) = D1(:,i)/norm(D1(:,i));
            D2(:,i) = D2(:,i)/norm(D2(:,i));
            
            A(i,inds) = [sqrt(omega)*D1(:,i); sqrt(1-omega)*D2(:,i)]'* [sqrt(omega)*E_i_1; sqrt(1-omega)*E_i_2];
        else % replacing the unused atoms
            [~,iind] = max(sum((X1-D1*A).^2).*sum((X2-D2*A).^2));
            D1(:,i) = X1(:,iind)/norm(X1(:,iind));
            D2(:,i) = X2(:,iind)/norm(X2(:,iind));
        end
    end
    
    if print == true
        r1 = norm(D1*A-X1,'fro')^2;
        r2 = norm(D2*A-X2,'fro')^2;
        NZ = nnz(A);
        if j==1
            fprintf('iter \t ||DA1-X1||_F^2 \t ||DA2-X2||_F^2 \t ||A||_0 \n');
        end
        fprintf('%3g \t %12.3e \t %15.3e \t %12.3e \t \n', j, r1,r2,NZ);
    end
    
end
toc
end

function [m1,m2,K,D1,D2,Nit,omega,print,remMean,DCatom,param] = Parse_input(X1,X2,opts)

[m1,N1] = size(X1);
[m2,N2] = size(X2);

if N1~=N2
    error('Input data should have equal number of columns')
end

if isfield(opts,'print')
    print = opts.print;
else
    print = false;
end

if isfield(opts,'remMean')
    remMean = opts.remMean;
else
    remMean = true;
end

if isfield(opts,'DCatom')
    DCatom = opts.DCatom;
else
    DCatom = true;
end

if isfield(opts,'Nit')
    Nit = opts.Nit;
else
    Nit = 10;
end

if isfield(opts,'k')
    param.L = opts.k;
else
    param.L = max([m1,m2])/4;
end

if isfield(opts,'eps')
    param.eps = opts.eps;
else
    param.eps = 1e-4;
end

if isfield(opts,'K')
    K = opts.K;
else
    K = 4*max([m1,m2]);
end

if isfield(opts,'omega')
    omega = opts.omega;
else
    omega = 0.5;
end

if isfield(opts,'D1') && isfield(opts,'D2')
    D1 = opts.D1;
    D2 = opts.D2;
else
    D1 = DCT(K,sqrt(m1)); 
    D2 = DCT(K,sqrt(m2));
end

end