function [C, G, Out]= solving_TSCLMVSC(X, cls_num, gt, opts)   

%% Note: Tensorized Specificity and Consistency for 
%% Latent Multi-View Subspace Clustering
% Input:
%   X:          feature matrices
%   cls_num:    number of clusters
%   gt:         ground truth clusters
%   opts:       optional parameters
%               - maxIter: max iteration
%               - lambda1, lambda2, etc:  hyper-parameter
%               - mu: penalty parameter
%               - epsilon: stopping tolerance
% Outout:
%   C:          clusetering results
%   G:          affinity matrix
%   Out:        other output information, e.g. metrics

%% Parameter settings
K = length(X);   % number of views
N = size(X{1},2); % sample number

% Default
flag_debug = 0;
epsilon = 1e-6;
mu = 1e-5; 
max_mu = 1e10; 
pho_mu = 2;
maxIter = 200;


if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'lambda1');       lambda1 = opts.lambda1;         end
if  isfield(opts, 'lambda2');       lambda2 = opts.lambda2;         end
if  isfield(opts, 'lambda3');       lambda3 = opts.lambda3;         end
if  isfield(opts, 'm');             m1 = opts.m;                    end
if  isfield(opts, 'mu');            mu = opts.mu;                   end
if  isfield(opts, 'max_mu');        max_mu = opts.max_mu;           end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end

%% Initialize...
Z = cell(1,K);
W = Z;
E1 = Z;
Y1 = Z;
S = Z;
P = Z;
PTP = Z;

Y2 =cell(1,K+1);
Z = Y2;
J = Z;

chg1 = cell(1,K); 
chg2 = cell(1,K+1);

d1 = cell(1,K);
d2 = cell(1,K);

for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    J{k} = zeros(N,N);
    E1{k} = zeros(size(X{k},1),N); 
    Y1{k} = zeros(size(X{k},1),N);
    Y2{k} = zeros(N,N);
    PTP{k} = zeros(m1,m1);
    S{k} = zeros(m1,N);
    P{k} = zeros(size(X{k},1),m1);
end
C = zeros(m1,N);
Z_C = zeros(N,N);
Y3 = zeros(N,N);

Z{K+1} = zeros(N,N);
J{K+1} = zeros(N,N);
Y2{K+1}=Y3;

Isconverg = 0;
iter = 0;
while(Isconverg == 0)
    if flag_debug == 1
       fprintf('----processing iter %d--------\n', iter + 1);
    end
    %% 1-------------------Update P^k------------------------------- 
    for k=1:K
        G1 = S{k}+C;
        Q1 = (1/mu*Y1{k}+X{k}-E1{k})';
        W1 = G1*Q1;
        [U,~,V] = svd (W1,'econ'); 
        P{k} = V*U'; 
    end

    %% 2-------------------Update S^k-------------------------------
    temp1 = zeros(m1,N);
    for k=1:K
        PTP{k} = P{k}'*P{k};
        A1 = PTP{k}; 
        B1 = 2*lambda2/mu*((eye(N)-Z{k})*(eye(N)-Z{k})')+eye(N)*1e-10;
        temp2 = P{k}'*(X{k}-E1{k}+Y1{k}/mu);
        temp1 = temp1 + temp2;
        C1 = temp2-PTP{k}*C;
        S{k} = sylvester(A1, B1, C1);
    end

    %% 3-------------------Update C---------------------------------
    A1 = zeros(m1,m1);
    for k=1:K
        A1 = A1 + PTP{k};
    end
    B1 = 2*lambda3/mu*((eye(N)-Z_C)*(eye(N)-Z_C)')+eye(N)*1e-10;
    temp3 = zeros(m1,N);
    for k=1:K
        temp3 = temp3-PTP{k}*S{k};
    end
    C1 = temp1 +temp3; 
    C =  sylvester(A1, B1, C1);

    %% 4-------------------Update Z^k-------------------------------
    for k=1:K
        HTH = S{k}'*S{k};
        tmp = 2*lambda2*HTH - Y2{k} + mu*J{k} ;
        Z{k} = (2*lambda2*HTH + mu*eye(N))\tmp;
    end 

    %% 5-------------------Update Z_C-------------------------------
    CTC = C'*C;
    tmp1 = 2*lambda3*CTC -Y3 + mu*J{K+1};
    Z_C = (2*lambda3*CTC + mu*eye(N))\tmp1;

    %% 6-------------------Update E^k-------------------------------
    C1 = [];
    for k=1:K   
        tmp1 = X{k} - P{k}*(S{k}+C) + Y1{k}/mu;
        C1 = [C1; tmp1];
    end
    [Econcat] = solve_l1l2(C1,lambda1/mu);
    start = 1;
    for k=1:K
        E1{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end

    %% 7-------------------Update J_tensor--------------------------
    Z{K+1} = Z_C;
    Z_tensor = cat(3, Z{:,:});
    Y2_tensor = cat(3, Y2{:,:});
    temp3 = Z_tensor + Y2_tensor/mu;
    [J_tensor, ~] = logDet_Shrink(temp3, 1/mu, 3); % Logdet
   
    %% 8-------------------Update auxiliary variable------------------
    for k=1:K
        J{k} = J_tensor(:,:,k); 
        d1{k} = X{k}-P{k}*(S{k}+C)-E1{k};
        d2{k} = Z{k} - J{k};
        Y1{k} = Y1{k} + mu*(d1{k});
        Y2{k} = Y2{k} + mu*(d2{k});
    end
    J{K+1} = J_tensor(:,:,K+1);
    d2{K+1} = Z{K+1} - J{K+1};
    Y2{K+1} = Y2{K+1} + mu*(d2{K+1});
    Y3  = Y2{K+1};

    %% ------------------- Converge check ----------------------------
    Isconverg = 1;
    for k=1:K
        chg1{k}=norm(d1{k},inf);
        if (chg1{k}>epsilon)
            if flag_debug==1
               fprintf('norm_X   %7.10f     \n', chg1{k});
            end
            Isconverg = 0;
        end
    end
    for k=1:K+1
        chg2{k}=norm(d2{k},inf);
        if (chg2{k}>epsilon)
            if flag_debug==1
               fprintf('norm_Z_J %7.10f   \n', chg2{k});
            end
            Isconverg = 0;
        end
    end

    if (iter>maxIter)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
end

%% ---------------- Clustering --------------------------------------
A = zeros(N,N);
for k=1:K
    A = A + abs(Z{k})+abs(Z{k}');
end
A = A/K;
A = A+abs(Z{K+1})+abs(Z{K+1}');
G = A;
C = SpectralClustering(A,cls_num);
[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,~,~,~]=RandIndex(gt,C);
purity=compute_purity(gt,C);

%% ---------------- Record ------------------------------------------
Out.NMI = nmi;
Out.AR = AR;
Out.ACC = ACC;
Out.recall = r;
Out.precision = p;
Out.fscore = f;
Out.purity=purity;
end