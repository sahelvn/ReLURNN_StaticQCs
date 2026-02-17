%% Example: Gain analysis of a ReLU RNN

%% Data

% Dimensions
nx = 4;
nv = 2;
nw = nv;
nd = 3;
ne = 3;

% Random continuous-time example
% Shift the eigenvalues so that the slowest one has real part = -1
%rng('default');
%rng(0)  % Same as default
%rng(12)
%rng(45)  % DH can't even prove stability for N=5
rng(67)
A = randn(nx,nx);
e = max(real(eig(A)));
A = A-(e+1)*eye(nx);
B = randn(nx,nw+nd);
C = randn(nv+ne,nx);
D = zeros(nv+ne,nw+nd);
Gc = ss(A,B,C,D);

% Discretize the dynamics
dt = 0.1;
Gd = c2d(Gc,dt);
[Ad,Bd,Cd,Dd] = ssdata(Gd);

% Truncate the coefficients to 2 significant digits
Nsig = 2;
Ad = round(Ad,Nsig);
Bd = round(Bd,Nsig);
Cd = round(Cd,Nsig);
Gd = ss(Ad,Bd,Cd,0,1);

%% Compute stability margin using lifted ReLU QCs and DH QCs
RunStab = false;  % Set = true to run stability analysis
if RunStab
    NR = 5;
    alphaR = zeros(NR,1);
    for N=1:NR
        alphaR(N) = RNNstabReLU(Gd(1:nv,1:nw), N);
        %[N alphaR(N)]
    end
    fprintf('\n alphaR(1) = %4.3f \t alphaR(end) = %4.3f\n',alphaR(1),alphaR(end));

    alphaD = zeros(NR,1);
    for N=1:NR
        alphaD(N) = RNNstabDH(Gd(1:nv,1:nw), N);
        %[N alphaD(N)]
    end
    fprintf('\n alphaD(1) = %4.3f \t alphaD(end) = %4.3f\n',alphaD(1),alphaD(end));
end

%% Compute gain using lifted ReLU QCs and DH QCs
Ng = 5;
gamR = zeros(Ng,1);
for N=1:Ng
    gamR(N) = RNNgainReLU(Gd, N,nd);
    %[N gamR(N)]
end
fprintf('\n gamR(1) = %4.3f \t gamR(end) = %4.3f\n',gamR(1),gamR(end));


gamD = zeros(Ng,1);
for N=1:Ng
    gamD(N) = RNNgainDH(Gd, N,nd);
    %[N gamD(N)]
end
fprintf('\n gamD(1) = %4.3f \t gamD(end) = %4.3f\n',gamD(1),gamD(end));

%% Plot results
if RunStab
    figure(1)
    plot(1:NR,alphaR,'b-x', 1:NR,alphaD,'c-v');
    xlabel('N');
    ylabel('\alpha');
    legend('ReLU','DH','Location','Best');
    grid on;
    xlim([1 NR])
    if exist('garyfyFigure','file'), garyfyFigure, end
end

figure(2)
plot(1:Ng,gamR,'b-x',1:Ng,gamD,'c-v');
xlabel('N');
ylabel('\gamma');
legend('ReLU','DH','Location','Best');
grid on;
xlim([1 Ng])
if exist('garyfyFigure','file'), garyfyFigure, end
