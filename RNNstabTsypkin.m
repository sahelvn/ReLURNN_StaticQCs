function [alpha, info]=RNNstabTsypkin(G)
% function [alpha, info]=RNNstabTsypkin(G)
%
% This function analyzes the stability boundary of a recurrent
% neural network (RNN) of the form:
%      v = G (w+d)
%      w = alpha*Phi(v) where Phi:R^nv -> R^nv is a repeated nonlinearity
% The repeated nonlinearity is assumed to apply a scalar function 
% elementwise:
%    w_k = phi( v_k ) where phi is the scalar nonlinearity
% It is assumed the nonlinearity lies in the sector [0,1].
% The goal is to find the maximum value of alpha for which this system
% is stable. The code uses the Tsypkin criterion.
%
% Inputs:
% G is discrete-time plant with input w and outputs v
% 
% Outputs
% alpha is a lower bound on the maximum stability boundary
% info contains solver information.

% Grab state-matrices 
[A,B,C,D] = ssdata(G);
if norm(D)>0
    % XXX PJS -- I should double check if the condition can be
    % modified to handle D~=0.
    error('Tsypkin requires D=0');
end

% Absolute and relative bisection tolerance
rtol = 1e-3;
atol = 1e-3;

% Set upper and lower bounds on the series gain
alphaUB =  200;
alphaLB = 0;

% Use bisection to find maximum alpha for which LMI condition is feasible
infoLB = [];
while alphaUB-alphaLB > rtol*alphaUB + atol

    % Set alpha 
    alpha = (alphaUB + alphaLB)/2;

    % Scale the system inputs
    Bs = B*alpha;

    % Test feasbility of circle condition
    infos = LocalTsypkin(A,Bs,C);

    % Check feasibility
    if isequal(infos.status,'Solved')
        % Feasible condition: System is stable with alpha
        alphaLB = alpha;
        infoLB = infos;
    else
        % Infeasible condition: System is unstable with alpha
        alphaUB = alpha;
    end
end

% Store data for output
alpha = alphaUB;
info = infoLB;
info.alphaBounds = [alphaLB alphaUB];


%% Local function: Tsypkin Criterion
function info = LocalTsypkin(A,Bs,C)

% Construct augmented system
[nv,nx] = size(C);
Aa = [A zeros(nx,nv); C zeros(nv,nv)];
Ba = [Bs; zeros(nv,nv)];
S = [C zeros(nv,nv)];

cvx_begin sdp quiet
    variable P(nx+nv,nx+nv) symmetric
    variable Q0(nv,nv) diagonal
    variable Q1(nv,nv) diagonal
    
    % Q0, Q1 diagonal with non-negative entries
    for i=1:nv
        Q0(i,i)>=0;
        Q1(i,i)>=0;
    end
    
    % Storage matrix is positive semidefinite
    P >= eye(nx+nv);
   
    % Matrix for Lyapunov function difference: V(k+1) - V(k)
    dV = [Aa Ba]'*P*[Aa Ba] - blkdiag(P,zeros(nv));
        
    % Matrix for quadratic contraint, [VN; WN]' Mqc [VN; WN]
    % Rfac is defined such that: 
    %   [v(k+1); v(k); w(k+1)] = Rfac*[x(k+1); v(k); w(k+1)];
    Rfac = blkdiag(C,eye(nv),eye(nv));
    M = [zeros(2*nv) [Q0+Q1;-Q1]; [Q0+Q1 -Q1] -2*Q0];    
    Mqc = Rfac'*M*Rfac;

    % LMI Condition    
    dV + Mqc <=0;
cvx_end

% Store data for output
info.P = P;
info.Q0 = Q0;
info.Q1 = Q1;
info.M = M;
info.status = cvx_status;

