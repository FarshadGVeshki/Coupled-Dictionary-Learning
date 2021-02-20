# Coupled-Dictionary-Learning
Matlab code for the CDL method in 2019 paper "An Efficient Coupled Dictionary Learning Method"

 A computationally efficient algorithm for learning a pair of coupled dictionaries {D1,D2} corresponding to the correlated joint dataset
 {X1, X2} so that D1*A = X1 and D2*A = X2. where A is the common sparse represenation matrix

 Optimization problem:
 
  {D1, D2} = argmin_{D1,D2,A}   omega*||D1*A-X1||_F^2  +  (1-omega)*||D2*A-X2||_F^2
  
               s.t. ||A_i||_0 < k , i, 1,...,N and
               
                   ||D1_t||_2 = 1 ,||D2_t||_2 = 1 , t = 1,...,K

 Input variables:
 
   opts.omaga: tuning parameter (default: 0.5)
   
   opts.K:    number of atoms in dictionaries (default: 4*max([size(X1,1),size(X1,1)])) 
   
   opts.k:    maximum number of nonzero entries in columns of A  (default: max([size(X1,1),size(X1,1)])/4)
   
   opts.Nit:  number of CDL iterations (default: 10) 
   
   opts.eps:  approximation accuracy threshold (default: 1e-4)
   
   opts.D1 and opts.D1:  initial dictionaries (default: DCT dictionaries)
   
   opts.DCatom  ('true' or 'false') if true, keeps DC atom unchanged (default: true)

 Refrence:
   F. G. Veshki and S. A. Vorobyov, "An Efficient Coupled Dictionary Learning Method," in IEEE Signal Processing Letters, vol. 26, no. 10,
   pp. 1441-1445, Oct. 2019, doi: 10.1109/LSP.2019.2934045.

 (the mexOMP algorithm used for sparse approximation is taken from SPAMS toolbox (see mexOMP.m file))
