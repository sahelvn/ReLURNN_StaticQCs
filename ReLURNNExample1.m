%% Example 1: Stability analysis for max series gain of a ReLU RNN
% Discrete-time exapmles from:
% Carrasco, et al. "Convex searches for discrete-time Zames–Falb
% multipliers." IEEE Transactions on Automatic Control 65.11
% (2019): 4538-4553.
%

%% Examples
% See results in Table II of the reference
z = tf('z');

Example = 6;
% I introduced minus signs in all plants as our condition assumes
% positive feedback but the reference assumes negative feedback.
switch Example
    case 1
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [0.7935, 3.80, 13.5113, 36.1]
        Gd = -0.1*z/(z^2-1.8*z+0.81) ;
    case 2
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [0.1984, 0.2427, 1.1056, 2.7455]
        Gd = -(z^3-1.95*z^2+0.9*z+0.05) / (z^4-2.8*z^3+3.5*z^2 -2.412*z+.7209);
    case 3
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [0.1379, 0.1379, 0.3121, 0.3126]
        Gd = (z^3-1.95*z^2+0.9*z+0.05) / (z^4-2.8*z^3+3.5*z^2 -2.412*z+.7209);
    case 4
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [1.5312, 1.6911, 3.8240, 7.7907]
        Gd = -(z^4-1.5*z^3+0.5*z^2-0.5*z+0.5)/(4.4*z^5-8.957*z^4+9.893*z^3-5.671*z^2+2.207*z-0.5);
    case 5
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [1.0273, 1.0273, 2.4475, 2.4475]
        Gd = -(-0.5*z+0.1)/(z^3-0.9*z^2+0.79*z+0.089);
    case 6
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [0.6510, 0.6510, 1.0870, 1.0870]
        Gd = -(2*z+0.92)/(z^2-0.5*z);
    case 7
        %  [Circle, Tsypkin, Best ZF, Nyquist] = [0.1069, 0.1069, 0.5280, 1.1766]
        Gd = -(1.341*z^4-1.221*z^3+0.6285*z^2-0.5618*z+0.1993)/(z^5-0.935*z^4+0.7697*z^3-1.118*z^2+0.6917*z-0.1352);
end
Gd = ss(Gd);

fprintf('\n------------Example # = %d', Example);

% Lifted ReLU QCs
NR = 5;
alphaR = zeros(NR,1);
for N=1:NR
    alphaR(N) = RNNstabReLU(Gd, N);
end
fprintf('\n alphaR(1) = %4.3f \t alphaR(end) = %4.3f',alphaR(1),alphaR(end));

% Lifted Slope-restricted Doubly Hyperdominance condition
alphaD = zeros(NR,1);
for N=1:NR
    alphaD(N) = RNNstabDH(Gd, N);
end
fprintf('\n alphaD(1) = %4.3f \t alphaD(end) = %4.3f',alphaD(1),alphaD(end));

% Circle: Lifting does not improve
alphaC = zeros(2,1);
alphaC(1) = RNNstabCircle(Gd, 1);
alphaC(2) = RNNstabCircle(Gd, NR);
fprintf('\n alphaC(1) = %4.3f \t alphaC(end) = %4.3f',alphaC(1),alphaC(2));

% Tsypkin: No lifting
alphaT = RNNstabTsypkin(Gd);
fprintf('\n alphaT = %4.3f \n',alphaT);

%% Plot results
figure(1)
plot(1:NR,alphaR,'b-x', 1:NR,alphaD,'c-v',...
    [1 NR],alphaC,'r-o',[1 NR],alphaT*[1 1],'k--');
xlabel('N');
ylabel('\alpha');
legend('ReLU','DH','Circle','Tsypkin','Location','Best')
grid on; xlim([1 NR])
if exist('garyfyFigure','file'), garyfyFigure, end

