function [alpha, info]=RNNstabReLU(G, N)
% function [alpha, info]=RNNstabReLU(G, N)
%
% This function analyzes the stability boundary of a ReLU recurrent
% neural network (RNN) of the form:
%      v = G (w+d)
%      w = alpha*Phi(v) where Phi:R^nv -> R^nv is a repeated nonlinearity
% The repeated nonlinearity is assumed to apply a scalar function 
% elementwise:
%    w_k = phi( v_k ) where phi is the scalar nonlinearity
% The goal is to find the maximum value of alpha for which this system
% is stable. The code uses results in the reference below.
%
% Sahel Vahedi Noori, Bin Hu, Geir Dullerud, and Peter Seiler, "Stability 
% and Performance Analysis of Discrete-Time ReLU Recurrent Neural 
% Networks," submitted to the 2024 CDC.
%
% Inputs:
% G is discrete-time plant with inputs w and outputs v
% N is the time horizon for the lifting.
% 
% Outputs
% alpha is a lower bound on the maximum stability boundary
% info contains solver information.

% Check inputs
if nargin==1
    N=1;  % Default is no lifting
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

    % Scale the system
    Gs = G*alpha;

    % Lift system
    nd = 0;
    ne = 0;
    GN = liftPlant(Gs,N,nd,ne);

    % Test feasbility of ReLU QC
    info = LocalReLUQC(GN);
    
    % Check feasibility
    if isequal(info.status,'Solved')
        % Feasible condition: System is stable with alpha
        alphaLB = alpha;
        infoLB = info;
    else
        % Infeasible condition: System is unstable with alpha
        alphaUB = alpha;
    end
end

% Store data for output
alpha = alphaUB;
info = infoLB;
info.alphaBounds = [alphaLB alphaUB];

%% Local function: Circle Criterion
function info = LocalReLUQC(G)

% Dimensions
[A,B,C,D] = ssdata(G);
[nx,m] = size(B);

% Test Feasbility
% This uses CVX to implement the LMI condition in the reference.
cvx_begin sdp quiet
    variable P(nx,nx) symmetric
    variable Q2(m,m) symmetric
    variable Q3(m,m) symmetric
    variable Qtil(m,m)    
    
    % Q2 and Q3 have non-negative entries
    for i=1:m
        for j=1:m
            Q2(i,j)>=0;
            Q3(i,j)>=0;
        end
    end

    % Qtil is a Metzler matrix, i.e. all off-diag entries are non-neg.
    for i=1:m
        for j=1:m
            if i ~= j
                Qtil(i,j)>=0;
            end
        end
    end

    % Storage matrix is positive semidefinite
    P >= eye(nx);
    
    % Matrix for Lyapunov function difference: V(k+1) - V(k)
    dV = [A B]'*P*[A B] - blkdiag(P,zeros(m));
        
    % Matrix for quadratic contraint, [VN; WN]' Mqc [VN; WN]
    % Rfac is defined such that: [VN; WN] = Rfac*[x; WN; DN] 
    Rfac = [C D; zeros(m,nx) eye(m)];
    M = [Q2 -Qtil'-Q2'; -Qtil-Q2 Q2+Q3+Qtil+Qtil'];    
    Mqc = Rfac'*M*Rfac;

    % LMI Condition    
    dV + Mqc <=0;
cvx_end
    
% Store data for output
info.P = P;
info.Q2 = Q2;
info.Q3 = Q3;
info.Qtil = Qtil;
info.M = M;
info.status = cvx_status;


