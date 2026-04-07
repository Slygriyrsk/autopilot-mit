clear; clc; close all;

%longitudinal B747 params
Xu = -1.982e3; % partial derivative of the X (axial/forward) force with respect to forward speed perturbation u: X_u = ∂X/∂u
Xw = 4.025e3; %∂X/∂w (change in axial force from vertical velocity w).
Zu = -2.595e4; %∂Z/∂u (change in normal force Z from forward speed)
Zw = -9.030e4; %∂Z/∂w (normal force sensitivity to w)
Zq = -4.524e5; %∂Z/∂q (normal force sensitivity to pitch rate)
Zwd = 1.909e3; %derivative of Z with respect to vertical acceleration term ẇ (added‑mass / apparent mass term)
Mu = 1.593e4; %∂M/∂u (pitch moment change with u)
Mw = -1.563e5; %∂M/∂w (pitch moment change with w)
Mq = -1.521e7; %∂M/∂q (pitch damping stiffness from pitch rate)
Mwd = -1.702e4; %pitching moment derivative with respect to ẇ (rate coupling)
g = 9.81; theta0 = 0;

S = 511; %gross wing area (Cbar*b), where b is the wing span
cbar = 8.324; %this is mean chord
U0 = 235.9; % m/s nominal forward speed
Iyy = 0.449e8; %Iyy is interia along the aircraft pitch axis y
m = 2.83176e6 / g;
rho = 0.3045; %air density, L=1/2*rho*v^2*S*Cl (where Cl is the lift coeffecient)

Xdp = 0.3 * m * g; %change in X due to thrust/propulsive command
Zdp = 0; %normal force due to thrust/pedal etc
Mdp = 0; %pitching moment due to thrust
Xde = -3.818e-6 * (0.5 * rho * U0^2 * S); %elevator control derivative: change in X due to elevator deflection.
Zde = -0.3648    * (0.5 * rho * U0^2 * S); %normal force due to elevator.
Mde = -1.444     * (0.5 * rho * U0^2 * S * cbar); %pitching moment due to elevator

A = [ Xu/m, Xw/m, 0, -g;
      Zu/(m-Zwd), Zw/(m-Zwd), (Zq+m*U0)/(m-Zwd), 0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy, ...
      (Mw+Zw*Mwd/(m-Zwd))/Iyy, ...
      (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy, 0;
      0 0 1 0];

A_aug = [A zeros(4,1);
         0 1 0 U0 0]; %h = -w + Uo*0, so flip the w sign else -ve alt


B_aug = [ Xde/m,              Xdp/m;
          Zde/(m-Zwd),        Zdp/(m-Zwd);
          (Mde + Zde*Mwd/(m-Zwd))/Iyy, (Mdp + Zdp*Mwd/(m-Zwd))/Iyy;
          0, 0; 
          0, 0 ];

Bw_aug = [ -Xu/m,                     -Xw/m,                0;
           -Zu/(m-Zwd),               -Zw/(m-Zwd),          0;
           (-Mu - Zu*Mwd/(m-Zwd))/Iyy, (-Mw - Zw*Mwd/(m-Zwd))/Iyy, -Mq/Iyy;
            0,                         0,                   0;
            0,                         0,                   0 ];

Ts = 0.01;
sysd = c2d(ss(A_aug,[B_aug Bw_aug],eye(5),0),Ts);
Ad = sysd.A;
Bd = sysd.B(:,1:2);
Bwd = sysd.B(:,3:5);

Ad_aug = [Ad, Bwd; 
        zeros(3,5), eye(3)];

C_aug = [eye(5), zeros(5,3)]; 

%training here
numSim = 500;
N = 500;
t = (0:N-1)*Ts;

XTrain = cell(numSim,1);
YTrain = cell(numSim,1);

allX = [XTrain{:}];
muX = mean(allX, 2); sigmaX = std(allX, 0, 2);

allY = [YTrain{:}];
muY = mean(allY, 2); sigmaY = std(allY, 0, 2);

for s = 1:numSim
    XTrain{s} = (XTrain{s} - muX) ./ sigmaX;
    YTrain{s} = (YTrain{s} - muY) ./ sigmaY;
end

%for kalman filter only
for s=1:numSim
    X=zeros(5,1); x_hat=zeros(5,1); P=eye(5);
    Q=diag([1e-10, 1e-10, 1e-11, 1e-13, 1e-13]);
    R=diag([ (1e-1)^2, (1e-1)^2, (1e-1)^2, (1e-2)^2, (1)^2 ]);
    L=chol(R,'lower');
    
    innov_store=zeros(5,N);
    gust_store=zeros(3,N);

    g_mag=2+5*randn(3,1);
    g_freq=0.1+1.4*rand(3,1);
    g_phase = 2*pi*rand(3,1);
    
    for k=1:N
        u=[0;0];
        %gust=[10;5;0];, this won't work because model will se this only
        %for 12000 simulations no learning

        gust = [g_mag(1) * sin(2*pi*g_freq(1)*t(k) + g_phase(1));
                g_mag(2) * sin(2*pi*g_freq(2)*t(k) + g_phase(2));
                0]; 
        
        X = Ad*X + Bd*u + Bwd*gust;
        z = X + L*randn(5,1);
        
        x_pred = Ad*x_hat + Bd*u;
        P_pred = Ad*P*Ad' + Q;
        
        y = z - x_pred;
        
        S = P_pred + R;
        K = P_pred / S;
        
        x_hat = x_pred + K*y;
        %P = (eye(5)-K)*P_pred;

        %Applying Joseph equation
        I=eye(5);
        P=(I-K)*P_pred*(I-K)'+K*R*K';
        P=(P+P')/2;

        
        innov_store(:,k)=y;
        gust_store(:,k)=gust;
    end
    
    XTrain{s}=innov_store;
    YTrain{s}=gust_store;
end

allData = [];
for s=1:numSim
    allData = [allData XTrain{s}];
end

mu = mean(allData,2);
sigma = std(allData,0,2)+1e-6;

for s=1:numSim
    XTrain{s}=(XTrain{s}-mu)./sigma;
end

%normalize op
allG=[];
for s=1:numSim
    allG=[allG YTrain{s}];
end

mu_g=mean(allG,2);
sigma_g=std(allG,0,2)+1e-6;

for s=1:numSim
    YTrain{s}=(YTrain{s}- mu_g) ./ sigma_g;
end

%gru hyperparameters
inputSize=5; %(5 state residuals + 2 contorl inputs)
hiddenSize=32;
outputSize=3; %(gusts)

rng(0)

net.Wz=randn(hiddenSize,inputSize)*0.1;
net.Uz=randn(hiddenSize,hiddenSize)*0.1;
net.bz=zeros(hiddenSize,1);

net.Wr=randn(hiddenSize,inputSize)*0.1;
net.Ur=randn(hiddenSize,hiddenSize)*0.1;
net.br=zeros(hiddenSize,1);

net.Wh=randn(hiddenSize,inputSize)*0.1;
net.Uh=randn(hiddenSize,hiddenSize)*0.1;
net.bh=zeros(hiddenSize,1);

net.Wo=randn(outputSize,hiddenSize)*0.1;
net.bo=zeros(outputSize,1);

%BPTT (Back Propagation Through Time)
epochs=10;
lr=0.016;
windowSize = 30; %truncating the long seq into segments of 50 timesteps each to calculate gradient and update weights

loss_history=zeros(epochs,1);

for epoch=1:epochs
    totalLoss=0;
    for s=1:numSim
        Xseq=XTrain{s};
        Gseq=YTrain{s};
        N = size(Xseq, 2); 
        
        h_current = zeros(hiddenSize,1); 
        
        for st = 1:windowSize:N
            windowEnd = min(st + windowSize - 1, N);
            actualWinSize = windowEnd - st + 1;
            
            H_win = zeros(hiddenSize, actualWinSize + 1);
            H_win(:,1) = h_current; 
            Z = zeros(hiddenSize, actualWinSize);
            Rg = zeros(hiddenSize, actualWinSize);
            Htilde = zeros(hiddenSize, actualWinSize);
            Ghat = zeros(outputSize, actualWinSize);
            
            %=Forward Pass
            for k = 1:actualWinSize
                idx = st + k - 1; % Global index
                x = Xseq(:,idx);
                hprev = H_win(:,k);
                
                z = sig(net.Wz*x + net.Uz*hprev + net.bz);
                r = sig(net.Wr*x + net.Ur*hprev + net.br);
                htil = tanh(net.Wh*x + net.Uh*(r.*hprev) + net.bh);
                h = (1-z).*hprev + z.*htil;
                
                H_win(:,k+1) = h;
                Z(:,k) = z; Rg(:,k) = r; Htilde(:,k) = htil;
                Ghat(:,k) = net.Wo*h + net.bo;
            end
            
            % Update h_current for the NEXT window (carry over)
            h_current = H_win(:, end);
            
            %Backward Pass
            grads = init_grads(net);
            dh_next = zeros(hiddenSize,1); % Truncation happens here
            
            for k = actualWinSize:-1:1
                idx = st + k - 1; % Global index
                err = Ghat(:,k)-Gseq(:,k);
                totalLoss=totalLoss+sum(err.^2);
                dy=2*err;
                
                grads.dWo = grads.dWo + dy*H_win(:,k+1)';
                grads.dbo = grads.dbo + dy;
                
                dh = net.Wo'*dy + dh_next;
                
                z=Z(:,k); r=Rg(:,k); htil=Htilde(:,k); hprev=H_win(:,k);
                
                dh_til = dh.*z.*(1-htil.^2);
                dz = dh.*(htil-hprev).*z.*(1-z);
                dr = (net.Uh'*dh_til).*hprev.*r.*(1-r);
                
                grads.dWh = grads.dWh + dh_til*Xseq(:,idx)';
                grads.dUh = grads.dUh + dh_til*(r.*hprev)';
                grads.dWz = grads.dWz + dz*Xseq(:,idx)';
                grads.dUz = grads.dUz + dz*hprev';
                grads.dWr = grads.dWr + dr*Xseq(:,idx)';
                grads.dUr = grads.dUr + dr*hprev';
                
                %dh next from the prev time step k-1
                dh_next = dh .* (1 - z) + net.Uz' * dz + net.Ur' * dr + (net.Uh' * dh_til) .* r;
            end
            
            %update weights after every window
            grads = clip_grads(grads,1);
            net = apply_update(net,grads,lr);
        end
    end
    
    fprintf('Epoch %d | MSE: %.6f\n', epoch, totalLoss/(numSim * N * 3));
    fprintf('Accuracy : %.6f\n', 100*sqrt(totalLoss/(numSim*N*3)));
    loss_history(epoch)=totalLoss/numSim;
end

%KF + GRU
P=eye(5);
Q=diag([1e-7, 1e-7, 1e-9, 1e-10, 1e-10]);
R=diag([ (1e-1)^2, (1e-1)^2, (1e-1)^2, (1e-2)^2, (1)^2 ]);
L=chol(R,'lower');

Q_aug = blkdiag(Q, diag([1e-1, 1e-3, 1e-2]));  % tune gust process noise as needed

% Measurement noise (same as before)
R_kf = R;

% Initialize augmented state & covariance
X_aug = zeros(8,1);   % [x; w]
P_aug = blkdiag(P, eye(3)*10);

h = zeros(hiddenSize,1);

gust_est = zeros(3,N);
gust_true = zeros(3,N);
gust_est_kf = zeros(3,N);

X_true_hist = zeros(5,N);    % true plant states
xhat_hist    = zeros(5,N);    % KF+GRU state estimates (x_hat)
x_aughat_hist = zeros(8,N);  %kalman only

figure
plot(1:epoch,loss_history,'LineWidth',2);
xlabel('Epoch');
ylabel('Loss');
title("Training Loss");
grid on;

%gust start and end, since my total time is 10 sec only Ts*N
t1=1; t2=2;
alpha=2;

X = zeros(5,1); x_hat = zeros(5,1);
P = eye(5);

for k=1:N
    s = 0.5*(tanh(alpha*(t(k)-t1)) - tanh(alpha*(t(k)-t2)));
    true_g = s*[8*sin(2*pi*0.5*t(k));5;0];
    u = [0;0];
    
    %true plant
    X = Ad*X + Bd*u + Bwd*true_g;
    z = X + L*randn(5,1);
    
    
    %KF+GRU path (ground model) [physics only]
    x_pred = Ad*x_hat + Bd*u;
    P_pred = Ad*P*Ad' + Q;
    
    y = z - x_pred;
    x_input = (y - mu) ./ sigma;
    
    [g_est,h] = gru_forward(x_input,h,net);
    g_est_denorm = g_est .* sigma_g + mu_g;
    
    %BUG: gust already there, model knows
    %x_pred = x_pred + Bwd*g_est_denorm;
    
    S = P_pred + R;
    S = (S+S')/2;
    K = (P_pred / S);
    
    x_hat = x_pred + K*(z - x_pred);
    P = (eye(5)-K)*P_pred;
    
    gust_est(:,k) = g_est_denorm;
    gust_true(:,k) = true_g;
    
    %KF-only (augmented KF)
    % Predict for augmented state
    u_aug = [Bd*u; zeros(3,1)];  % no direct control on gust states
    X_aug_pred = Ad_aug * X_aug + u_aug;  % includes gust influence in Ad_aug
    P_aug_pred = Ad_aug * P_aug * Ad_aug' + Q_aug;
    
    y_aug = z - C_aug * X_aug_pred;
    S_aug = C_aug * P_aug_pred * C_aug' + R_kf;
    S_aug = (S_aug + S_aug')/2;
    K_aug = P_aug_pred * C_aug' / S_aug;
    
    X_aug = X_aug_pred + K_aug * y_aug;
    P_aug = (eye(size(P_aug)) - K_aug * C_aug) * P_aug_pred;
    
    gust_est_kf(:,k) = X_aug(6:8);
    
    X_true_hist(:,k) = X;
    xhat_hist(:,k) = x_hat;
    x_aughat_hist(:,k) = X_aug; 
end

state_names = {'u (m/s)', 'w (m/s)', 'q (rad/s)', 'theta (rad)', 'h (m)'};

figure('Name','State Tracking: True vs KF+GRU vs Augmented-KF')
for i = 1:5
    subplot(5,1,i)
    plot(t, X_true_hist(i,:), 'k', 'LineWidth', 1.5); hold on
    plot(t, xhat_hist(i,:), 'b--', 'LineWidth', 1.2);
    % augmented KF states are the first 5 elements of x_aug estimate:
    %plot(t, squeeze(x_aughat_hist(i,:)), 'r:','LineWidth',1.2);
    ylabel(state_names{i})
    if i==1
        legend('True','KF+GRU (x\_hat)','KF-only (augmented KF)')
    end
    grid on
    if i==5
        xlabel('time (s)')
    end
end
sgtitle('State tracking comparison across simulation')

figure
subplot(3,1,1)
plot(t,gust_true(1,:),'k','LineWidth',2); hold on
plot(t,gust_est_kf(1,:),'r--')
plot(t,gust_est(1,:),'b')
legend('True','KF only','KF+GRU')
title('Longitudinal Gust')

subplot(3,1,2)
plot(t,gust_true(2,:),'k','LineWidth',2); hold on
plot(t,gust_est_kf(2,:),'r--')
plot(t,gust_est(2,:),'b')
legend('True','KF only','KF+GRU')
title('Normal Gust')

subplot(3,1,3)
plot(t,gust_true(3,:),'k','LineWidth',2); hold on
plot(t,gust_est_kf(3,:),'r--')
plot(t,gust_est(3,:),'b')
legend('True','KF only','KF+GRU')
title('Pitch Gust')

%{
function y=sig(x)
y=1./(1+exp(-x));
end
%}
%more stable to prevent overflow
function y = sig(x)
    y = zeros(size(x));
    idx = x >= 0;
    y(idx) = 1 ./ (1 + exp(-x(idx)));
    %for -ve val, use an alternative form to avoid exp(large positive)
    y(~idx) = exp(x(~idx)) ./ (1 + exp(x(~idx)));
end


function [g,h]=gru_forward(x,hprev,net)
    z=1./(1+exp(-(net.Wz*x+net.Uz*hprev+net.bz)));
    r=1./(1+exp(-(net.Wr*x+net.Ur*hprev+net.br)));
    htil=tanh(net.Wh*x+net.Uh*(r.*hprev)+net.bh);
    h=(1-z).*hprev+z.*htil;
    g=net.Wo*h+net.bo;
end

function grads=init_grads(net)
fields=fieldnames(net);
for i=1:length(fields)
    if startsWith(fields{i},'W')||startsWith(fields{i},'U')||startsWith(fields{i},'b')
        grads.(['d' fields{i}])=zeros(size(net.(fields{i})));
    end
end
end

function grads = clip_grads(grads, threshold)
    total_norm = 0;
    fields = fieldnames(grads);
    for i = 1:length(fields)
        total_norm = total_norm + sum(grads.(fields{i})(:).^2);
    end
    total_norm = sqrt(total_norm);
    
    if total_norm > threshold
        scale = threshold / (total_norm + 1e-6);
        for i = 1:length(fields)
            grads.(fields{i}) = grads.(fields{i}) * scale;
        end
    end
end

function net=apply_update(net,grads,lr)
fields=fieldnames(grads);
for i=1:length(fields)
    name=fields{i}(2:end);
    net.(name)=net.(name)-lr*grads.(fields{i});
end
end