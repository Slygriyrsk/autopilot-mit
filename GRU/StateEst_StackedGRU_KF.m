clear; clc; close all;

% Simulation parameters
Ts = 0.01;
T  = 20;
N  = round(T/Ts);

% B747 model parameters
Xu  = -1.982e3;
Xw  =  4.025e3;
Zu  = -2.595e4;
Zw  = -9.030e4;
Zq  = -4.524e5;
Zwd =  1.909e3;
Mu  =  1.593e4;
Mw  = -1.563e5;
Mq  = -1.521e7;
Mwd = -1.702e4;
g   =  9.81;
theta0 = 0;
S    = 511;
cbar = 8.324;
U0   = 235.9;
Iyy  = 0.449e8;
m    = 2.83176e6 / g;
rho  = 0.3045;

Xdp = 0.3 * m * g;
Zdp = 0;
Mdp = 0;
Xde = -3.818e-6        * (0.5 * rho * U0^2 * S);
Zde = -0.3648          * (0.5 * rho * U0^2 * S);
Mde = -1.444           * (0.5 * rho * U0^2 * S * cbar);

% Continuous-time state-space (5-state augmented with altitude h)
A = [ Xu/m,           Xw/m,                   0,                        -g;
      Zu/(m-Zwd),     Zw/(m-Zwd),             (Zq+m*U0)/(m-Zwd),        0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy, ...
      (Mw+Zw*Mwd/(m-Zwd))/Iyy, ...
      (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy,         0;
      0, 0, 1, 0 ];

A_aug = [A,           zeros(4,1);
         0, 1, 0, U0, 0];          % h_dot = -w + U0*theta (small-angle)

B_aug = [ Xde/m,                           Xdp/m;
          Zde/(m-Zwd),                     Zdp/(m-Zwd);
          (Mde+Zde*Mwd/(m-Zwd))/Iyy,      (Mdp+Zdp*Mwd/(m-Zwd))/Iyy;
          0,                               0;
          0,                               0 ];

Bw_aug = [ -Xu/m,                    -Xw/m,                   0;
           -Zu/(m-Zwd),              -Zw/(m-Zwd),              0;
           (-Mu-Zu*Mwd/(m-Zwd))/Iyy, (-Mw-Zw*Mwd/(m-Zwd))/Iyy, -Mq/Iyy;
            0,                        0,                        0;
            0,                        0,                        0 ];

% Discretize
sysd = c2d(ss(A_aug, [B_aug, Bw_aug], eye(5), 0), Ts);
Ad   = sysd.A;
Bd   = sysd.B(:, 1:2);
Bwd  = sysd.B(:, 3:5);

% Measure u, w, q, theta (states 1-4); altitude h not directly measured here
C = [1 0 0 0 0;
     0 1 0 0 0;
     0 0 1 0 0;
     0 0 0 1 0];


x_true = zeros(5, N);
x_kf   = zeros(5, N);
x_gru  = zeros(5, N);

x_true(:,1) = [5; 10; 0.01; 0.08; 200];
x_kf(:,1)   = x_true(:,1);
x_gru(:,1)  = x_true(:,1);

% Control inputs
u    = zeros(2, N);
udot = zeros(2, N);
for k = 2:N
    udot(:,k) = 0.01 * randn(2,1);
    u(:,k)    = u(:,k-1) + Ts * udot(:,k);
end

% KF noise covariances
%{
Q = diag([1e-10, 1e-10, 1e-11, 1e-13, 1e-13]);
R = diag([(1e-1)^2, (1e-1)^2, (1e-1)^2, (1e-2)^2]);
%}
Q=diag([1e-7, 1e-7, 1e-9, 1e-10, 1e-10]);
R=diag([ (1e-1)^2, (1e-1)^2, (1e-1)^2, 1 ]);
P = eye(5);

%KF sim
for k = 2:N
    wg = zeros(3,1);
    if (k*Ts)>=10 && (k*Ts)<=15
        wg(1)= 10*sin(2*pi*(k*Ts));
        wg(2)=5*cos(2*pi*(k*Ts));
        wg(3)= 1.5;
    end

    x_true(:,k) = Ad*x_true(:,k-1) + Bd*u(:,k-1) + Bwd*wg;

    y = C*x_true(:,k) + 0.02*randn(4,1);

    x_pred = Ad*x_kf(:,k-1) + Bd*u(:,k-1);
    P_pred = Ad*P*Ad' + Q;
    K_gain = P_pred*C' / (C*P_pred*C' + R);
    x_kf(:,k) = x_pred + K_gain*(y - C*x_pred);
    P = (eye(5) - K_gain*C) * P_pred;
end

%GRU Architecture
sig      = @(x) 1./(1+exp(-x));
sig_der  = @(f) f.*(1-f);
tanhf    = @(x) tanh(x);
tanh_der = @(f) 1 - f.^2;

nx  = 7;    % 5 KF states + 2 udot control inputs
nh1 = 8;
nh2 = 16;
ny  = 5;

lr = 1e-4;

% Xavier initialization
lim1 = sqrt(6/(nh1+nx));
Wr1 = -lim1 + 2*lim1*rand(nh1,nx);  Ur1 = -lim1 + 2*lim1*rand(nh1,nh1);  br1 = zeros(nh1,1);
Wz1 = -lim1 + 2*lim1*rand(nh1,nx);  Uz1 = -lim1 + 2*lim1*rand(nh1,nh1);  bz1 = zeros(nh1,1);
Wh1 = -lim1 + 2*lim1*rand(nh1,nx);  Uh1 = -lim1 + 2*lim1*rand(nh1,nh1);  bh1 = zeros(nh1,1);

lim2 = sqrt(6/(nh2+nh1));
Wr2 = -lim2 + 2*lim2*rand(nh2,nh1); Ur2 = -lim2 + 2*lim2*rand(nh2,nh2);  br2 = zeros(nh2,1);
Wz2 = -lim2 + 2*lim2*rand(nh2,nh1); Uz2 = -lim2 + 2*lim2*rand(nh2,nh2);  bz2 = zeros(nh2,1);
Wh2 = -lim2 + 2*lim2*rand(nh2,nh1); Uh2 = -lim2 + 2*lim2*rand(nh2,nh2);  bh2 = zeros(nh2,1);

Wy = 0.01*randn(ny, nh2);
by = zeros(ny, 1);

%normalize inputs
all_inputs = zeros(nx, N-2);
for kk = 2:N-1
    all_inputs(:, kk-1) = [x_kf(:,kk); udot(:,kk)];
end
xmin = min(all_inputs, [], 2);
xmax = max(all_inputs, [], 2);
xrng = xmax - xmin;
xrng(xrng < 1e-8) = 1e-8;  % avoid divide-by-zero

% Target scale for normalizing the residual output
target_scale = [5; 5; 0.1; 0.1; 500];

%train gru
epochs = 100;

for ep = 1:epochs
    h1 = zeros(nh1, 1);
    h2 = zeros(nh2, 1);
    total_loss = 0;
    total_err  = zeros(ny, 1);

    for k = 2:N-1
        inp      = [x_kf(:,k); udot(:,k)];
        inp_norm = 2*(inp - xmin)./xrng - 1;

        %saving prev hid states for gradients
        h1_prev = h1;
        h2_prev = h2;

        %fwd GRU layer 1
        r1       = sig(  Wr1*inp_norm + Ur1*h1_prev + br1 );
        z1       = sig(  Wz1*inp_norm + Uz1*h1_prev + bz1 );
        h_tilde1 = tanhf( Wh1*inp_norm + Uh1*(r1.*h1_prev) + bh1 );
        h1_new   = (1-z1).*h1_prev + z1.*h_tilde1;

        %fwd GRU layer 2
        r2       = sig(  Wr2*h1_new + Ur2*h2_prev + br2 );
        z2       = sig(  Wz2*h1_new + Uz2*h2_prev + bz2 );
        h_tilde2 = tanhf( Wh2*h1_new + Uh2*(r2.*h2_prev) + bh2 );
        h2_new   = (1-z2).*h2_prev + z2.*h_tilde2;

        %out
        y_hat = Wy*h2_new + by;

        %target is residual from true state and kf prediction
        % GRU learns the correction dx such that x_true ≈ Ad*x_kf + Bd*u + dx*target_scale
        y_true = (x_true(:,k+1) - (Ad*x_kf(:,k) + Bd*u(:,k))) ./ target_scale;

        e = y_hat - y_true; %err in prediction
        total_err  = total_err + abs(e);
        total_loss = total_loss + e'*e;

        %BPTT
        dWy = e * h2_new';
        dby = e;

        %gradient into h2_new
        dh2 = Wy' * e;

        %layer 2 gradients
        % h2_new = (1-z2).*h2_prev + z2.*h_tilde2
        dz2=dh2 .* (h_tilde2 - h2_prev) .* sig_der(z2);
        dh_tilde2=dh2 .* z2 .* tanh_der(h_tilde2);
        % r2 gates h2_prev inside h_tilde2: d(h_tilde2)/d(r2) = Uh2'*dh_tilde2 .* h2_prev
        dr2=(Uh2' * dh_tilde2) .* h2_prev .* sig_der(r2);

        %layer 2 wt updates
        Wy  = Wy  - lr * dWy;
        by  = by  - lr * dby;
        Wz2 = Wz2 - lr * (dz2 * h1_new');
        Uz2 = Uz2 - lr * (dz2 * h2_prev');
        bz2 = bz2 - lr * dz2;
        Wh2 = Wh2 - lr * (dh_tilde2 * h1_new');
        Uh2 = Uh2 - lr * (dh_tilde2 * (r2.*h2_prev)');
        bh2 = bh2 - lr * dh_tilde2;
        Wr2 = Wr2 - lr * (dr2 * h1_new');
        Ur2 = Ur2 - lr * (dr2 * h2_prev');
        br2 = br2 - lr * dr2;

        % ---- Gradient flowing into h1_new (from Layer 2) ----
        % h2_new depends on h1_new through r2, z2, h_tilde2
        % dL/d(h1_new) = Wz2'*dz2 + Wh2'*dh_tilde2 + Wr2'*dr2
        dh1 = Wz2' * dz2 + Wh2' * dh_tilde2 + Wr2' * dr2;

        % ---- Layer 1 gate gradients ----
        dz1       = dh1 .* (h_tilde1 - h1_prev) .* sig_der(z1);
        dh_tilde1 = dh1 .* z1 .* tanh_der(h_tilde1);
        dr1       = (Uh1' * dh_tilde1) .* h1_prev .* sig_der(r1);

        % ---- Update Layer 1 weights ----
        Wz1 = Wz1 - lr * (dz1 * inp_norm');
        Uz1 = Uz1 - lr * (dz1 * h1_prev');
        bz1 = bz1 - lr * dz1;
        Wh1 = Wh1 - lr * (dh_tilde1 * inp_norm');
        Uh1 = Uh1 - lr * (dh_tilde1 * (r1.*h1_prev)');
        bh1 = bh1 - lr * dh_tilde1;
        Wr1 = Wr1 - lr * (dr1 * inp_norm');
        Ur1 = Ur1 - lr * (dr1 * h1_prev');
        br1 = br1 - lr * dr1;

        % ---- Advance hidden states ----
        h1 = h1_new;
        h2 = h2_new;
    end

    fprintf('Epoch %3d | Loss = %.6f | Error norm = %.6f\n', ...
        ep, total_loss/(N-2), norm(total_err/(N-2)));
end

%GRU interface pass
h1 = zeros(nh1, 1);
h2 = zeros(nh2, 1);

for k = 2:N-1
    inp      = [x_kf(:,k); udot(:,k)];
    inp_norm = 2*(inp - xmin)./xrng - 1;

    % Layer 1
    r1       = sig(  Wr1*inp_norm + Ur1*h1 + br1 );
    z1       = sig(  Wz1*inp_norm + Uz1*h1 + bz1 );
    h_tilde1 = tanhf( Wh1*inp_norm + Uh1*(r1.*h1) + bh1 );
    h1       = (1-z1).*h1 + z1.*h_tilde1;

    % Layer 2
    r2       = sig(  Wr2*h1 + Ur2*h2 + br2 );
    z2       = sig(  Wz2*h1 + Uz2*h2 + bz2 );
    h_tilde2 = tanhf( Wh2*h1 + Uh2*(r2.*h2) + bh2 );
    h2       = (1-z2).*h2 + z2.*h_tilde2;

    y_hat = Wy*h2 + by;
    dx    = y_hat .* target_scale;

    %x_gru(:,k+1) = Ad*x_gru(:,k) + Bd*u(:,k) + dx;, if gru does small err
    %the err multiplies by Ad blasting the value of gru curve
    %x_gru(:,k+1) = x_kf(:,k+1) + dx;
    x_gru(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx;
end
x_gru(:,N) = x_gru(:,N-1);

%Accuracy
fprintf('\n-------- RMSE Accuracy --------\n');
state_labels = {'u', 'w', 'q', 'theta', 'h'};
idx = 2:N;
for j = 1:5
    rmse_kf  = sqrt(mean((x_true(j,idx) - x_kf(j,idx)).^2));   % FIX: was .^2 on wrong bracket
    rmse_gru = sqrt(mean((x_true(j,idx) - x_gru(j,idx)).^2));
    fprintf('%5s  |  KF RMSE = %8.3f  |  GRU RMSE = %8.3f\n', ...
        state_labels{j}, rmse_kf, rmse_gru);
end

t      = (0:N-1)*Ts;
labels = {'u (m/s)', 'w (m/s)', 'q (rad/s)', '\theta (rad)', 'h (m)'};

figure('Name','State Estimation Comparison','NumberTitle','off');
for i = 1:5
    subplot(5,1,i)
    plot(t, x_true(i,:), 'k',   'LineWidth', 1.2); hold on;
    plot(t, x_kf(i,:),   'r--', 'LineWidth', 1.0);
    plot(t, x_gru(i,:),  'b-.', 'LineWidth', 1.0);
    ylabel(labels{i});
    grid on;
    %ylim([-100, 300]);
    if i == 1
        legend('True', 'KF', 'KF+GRU', 'Location', 'best');
    end
end
xlabel('Time (s)');
sgtitle('B747 Longitudinal State Estimation');