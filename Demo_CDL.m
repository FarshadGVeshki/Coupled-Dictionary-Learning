%% joint sparse approximation (common sparse approximation) via learning coupled dictionaries
% Refrence:
%   F. G. Veshki and S. A. Vorobyov, "An Efficient Coupled Dictionary Learning Method," in IEEE Signal Processing Letters, vol. 26, no. 10,
%   pp. 1441-1445, Oct. 2019, doi: 10.1109/LSP.2019.2934045.

% (the algorithms used for patch extraction and reconstruction and
% dictionary visualization are taken from SPAMS toolbox (see mexExtractPatches.m and mexCombinePatches.m file))
%%
clear
clc
%% input data

% training data
I_org_tr = imresize(double(rgb2gray(imread('peppers.png'))),1)/255;
I_blur_tr = imgaussfilt(I_org_tr,4); % blurred version of I1

figure(1)
subplot 221
imshow(I_org_tr)
title('training data 1')
subplot 222
imshow(I_blur_tr)
title('training data 2')

% test data
I_org_test = imresize(double(rgb2gray(imread('pears.png'))),1)/255;
I_blur_test = imgaussfilt(I_org_test,4); % blurred version of I1

subplot 223
imshow(I_org_test)
title('testing data 1')
subplot 224
imshow(I_blur_test)
title('testing data 2')

%% Learning coupled dictionaries using training data

p = 8; % patch size
ss = 2; % sliding step
Eps = 1e-4; %approximation threshold
w = 0.5; % tuning parameter omega
param.omega = w; 
opts.eps = Eps; 
opts.k = 10; % maximim number of nonzeros in sparse vectors
opts.K = 64; % number of atoms in dictionaries
opts.Nit = 20; % number of CDL iterations
opts.remMean = true; % removing mean from the samples
opts.DCatom = true; % first atom is DC atom
opts.print = true; % printing the results

X_org_tr = mexExtractPatches(I_org_tr,p,ss);
X_blur_tr = mexExtractPatches(I_blur_tr,p,ss);
V = var([X_org_tr;X_blur_tr]); % patches with low variance are removed from traing data
X_org_tr(:,V<Eps) = [];
X_blur_tr(:,V<Eps) = [];

[D_org, D_blur] = CDL(X_org_tr,X_blur_tr,opts);

ID1 = displayPatches(D_org);
ID2 = displayPatches(D_blur);

figure(2)
subplot 121
imshow(ID1)
title('dictionary 1')
subplot 122
imshow(ID2)
title('dictionary 2')

%% approximation and reconstruction of test data using a common sparse representation

X_org_test = mexExtractPatches(I_org_test,p,ss);
X_blur_test = mexExtractPatches(I_blur_test,p,ss);

param.L = opts.k;
param.eps = Eps;

% finding common sparse representation:
A = mexOMP([sqrt(w)*X_org_test; sqrt(1-w)*X_blur_test],[sqrt(w)*D_org; sqrt(1-w)*D_blur],param);

X_org_rec = D_org*A;
X_blur_rec = D_blur*A;

I_org_rec = mexCombinePatches(X_org_rec,zeros(size(I_org_test)),p,0,ss);
I_blur_rec = mexCombinePatches(X_blur_rec,zeros(size(I_org_test)),p,0,ss);

figure(3)
subplot 221
imshow(I_org_test)
title('testing data 1')
subplot 222
imshow(I_blur_test)
title('testing data 2')
subplot 223
imshow(I_org_rec)
title('reconstructed data 1')
xlabel(['psnr ' num2str(psnr(I_org_rec,I_org_test))])
subplot 224
imshow(I_blur_rec)
title('reconstructed data 2')
xlabel(['psnr ' num2str(psnr(I_blur_rec,I_blur_test))])

