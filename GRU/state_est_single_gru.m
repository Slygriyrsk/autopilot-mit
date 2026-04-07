clear; clc; close all;

Ts = 0.01;        
T  = 20;         
N  = round(T/Ts);


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
S    = 511;          % gross wing area (m^2)
cbar = 8.324;        % mean aerodynamic chord (m)
U0   = 235.9;        % nominal forward speed (m/s)
Iyy  = 0.449e8;      % pitch moment of inertia (kg.m^2)
m    = 2.83176e6/g;  % aircraft mass (kg)
rho  = 0.3045;       % air density (kg/m^3)

% Control derivatives
Xdp = 0.3*m*g;
Zdp = 0;
Mdp = 0;
Xde = -3.818e-6 * (0.5*rho*U0^2*S);
Zde = -0.3648   * (0.5*rho*U0^2*S);
Mde = -1.444    * (0.5*rho*U0^2*S*cbar);


A = [ Xu/m,                          Xw/m,                              0,                                   -g;
      Zu/(m-Zwd),                    Zw/(m-Zwd),                        (Zq+m*U0)/(m-Zwd),                   0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy,      (Mw+Zw*Mwd/(m-Zwd))/Iyy,          (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy,    0;
      0,                             0,                                 1,                                   0 ];

% Altitude augmentation: h_dot = -w + U0*theta  (small-angle approximation)
A_aug = [ A,          zeros(4,1);
          0, -1, 0, U0, 0 ];

B_aug = [ Xde/m,                          Xdp/m;
          Zde/(m-Zwd),                    Zdp/(m-Zwd);
          (Mde+Zde*Mwd/(m-Zwd))/Iyy,     (Mdp+Zdp*Mwd/(m-Zwd))/Iyy;
          0,                             0;
          0,                             0 ];

Bw_aug = [ -Xu/m,                        -Xw/m,                          0;
           -Zu/(m-Zwd),                  -Zw/(m-Zwd),                    0;
           (-Mu-Zu*Mwd/(m-Zwd))/Iyy,    (-Mw-Zw*Mwd/(m-Zwd))/Iyy,     -Mq/Iyy;
            0,                            0,                              0;
            0,                            0,                              0 ];


sysd = c2d(ss(A_aug, [B_aug, Bw_aug], eye(5), 0), Ts);
Ad   = sysd.A;
Bd   = sysd.B(:, 1:2);
Bwd  = sysd.B(:, 3:5);


C = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0 ];


x_true = zeros(5, N);
x_kf   = zeros(5, N);
x_gru  = zeros(5, N);

x_true(:,1) = [250; 50; 0.2; 0.1; 2000];
x_kf(:,1)   = x_true(:,1);
x_gru(:,1)  = x_true(:,1);

u    = zeros(2, N);
udot = zeros(2, N);
for k = 2:N
    udot(:,k) = 0.01 * randn(2,1);
    u(:,k)    = u(:,k-1) + Ts*udot(:,k);
end

Q=diag([1e-7, 1e-7, 1e-9, 1e-10, 1e-10]);
R=diag([ (1e-1)^2, (1e-1)^2, (1e-1)^2, 1 ]);
P = eye(5);


for k = 2:N
    % Sinusoidal wind gust active between 100s and 200s
    wg = zeros(3,1);
    if (k*Ts) >= 10 && (k*Ts) <= 15
        wg(1) = 10  * sin(2*pi*(k*Ts));
        wg(2) =  5  * cos(2*pi*(k*Ts));
        wg(3) =  1.5;
    end

    % True state propagation (with gust)
    x_true(:,k) = Ad*x_true(:,k-1) + Bd*u(:,k-1) + Bwd*wg;

    % Noisy measurement of [u; w; q; theta]
    y = C*x_true(:,k) + 0.02*randn(4,1);

    % KF predict
    x_pred = Ad*x_kf(:,k-1) + Bd*u(:,k-1);
    P_pred = Ad*P*Ad' + Q;

    % KF update
    K_gain    = P_pred*C' / (C*P_pred*C' + R);
    x_kf(:,k) = x_pred + K_gain*(y - C*x_pred);
    P         = (eye(5) - K_gain*C) * P_pred;
end


sig      = @(x) 1./(1+exp(-x));   
sig_der  = @(f) f.*(1-f);
tanh_der = @(f) 1 - f.^2;

nx = 7;
nh = 32; 
ny = 5; 

lr = 1e-6;

%Xavier
lim = sqrt(6 / (nh + nx));
Wr = -lim + 2*lim*rand(nh, nx);   % reset gate   - input weight
Ur = -lim + 2*lim*rand(nh, nh);   % reset gate   - recurrent weight
br = zeros(nh, 1);                 % reset gate   - bias

Wz = -lim + 2*lim*rand(nh, nx);   % update gate  - input weight
Uz = -lim + 2*lim*rand(nh, nh);   % update gate  - recurrent weight
bz = zeros(nh, 1);                 % update gate  - bias

Wh = -lim + 2*lim*rand(nh, nx);   % candidate    - input weight
Uh = -lim + 2*lim*rand(nh, nh);   % candidate    - recurrent weight
bh = zeros(nh, 1);                 % candidate    - bias

Wy = 0.01*randn(ny, nh);           % output layer - weight
by = zeros(ny, 1);                 % output layer - bias

all_inputs = zeros(nx, N-2);
for kk = 2:N-1
    all_inputs(:, kk-1) = [x_kf(:,kk); udot(:,kk)];
end
xmin = min(all_inputs, [], 2);
xmax = max(all_inputs, [], 2);
xrng = xmax - xmin;
xrng(xrng < 1e-8) = 1e-8;  


target_scale = [50; 50; 0.1; 0.1; 1000];

epochs = 100;

for ep = 1:epochs

    h          = zeros(nh, 1); 
    total_loss = 0;
    total_err  = zeros(ny, 1);

    for k = 2:N-1
        inp      = [x_kf(:,k); udot(:,k)];
        inp_norm = 2*(inp - xmin)./xrng - 1;   

        h_prev = h;
        r = sig(Wr*inp_norm + Ur*h_prev + br);

        z = sig(Wz*inp_norm + Uz*h_prev + bz);

        h_tilde = tanh(Wh*inp_norm + Uh*(r.*h_prev) + bh);

        h_new = (1-z).*h_prev + z.*h_tilde;

        y_hat_lin = Wy*h_new + by;

        y_hat     = tanh(y_hat_lin);
        y_true = (x_true(:,k+1) - (Ad*x_kf(:,k) + Bd*u(:,k))) ./ target_scale;
        y_true = max(-1, min(1, y_true));

        e          = y_hat - y_true;
        total_err  = total_err + abs(e);
        total_loss = total_loss + e'*e;

        delta_out = e .* (1 - y_hat.^2);

        dWy = delta_out * h_new';
        dby = delta_out;

        dh = Wy' * delta_out;                          

        dz = dh .* (h_tilde - h_prev) .* sig_der(z);

        dh_tilde = dh .* z .* tanh_der(h_tilde);


        dr = (Uh' * dh_tilde) .* h_prev .* sig_der(r);

        all_grads   = [dWy(:); dby; dz; dh_tilde; dr];
        gnorm       = norm(all_grads);
        clip_thresh = 1.0;
        if gnorm > clip_thresh
            scale_g  = clip_thresh / gnorm;
            dWy      = dWy      * scale_g;
            dby      = dby      * scale_g;
            dz       = dz       * scale_g;
            dh_tilde = dh_tilde * scale_g;
            dr       = dr       * scale_g;
        end


        Wy = Wy - lr * dWy;
        by = by - lr * dby;

        Wz = Wz - lr * (dz * inp_norm');
        Uz = Uz - lr * (dz * h_prev');    

        Wh = Wh - lr * (dh_tilde * inp_norm');
        Uh = Uh - lr * (dh_tilde * (r.*h_prev)');
        bh = bh - lr * dh_tilde;

        Wr = Wr - lr * (dr * inp_norm');
        Ur = Ur - lr * (dr * h_prev');
        br = br - lr * dr;

        h = h_new;
    end

    fprintf('Epoch %3d | Loss = %.6f | Error norm = %.6f\n', ...
        ep, total_loss/(N-2), norm(total_err/(N-2)));
end

h = zeros(nh, 1); 

for k = 2:N-1
    inp      = [x_kf(:,k); udot(:,k)];
    inp_norm = 2*(inp - xmin)./xrng - 1;

    r       = sig(  Wr*inp_norm + Ur*h  + br );
    z       = sig(  Wz*inp_norm + Uz*h  + bz );
    h_tilde = tanh( Wh*inp_norm + Uh*(r.*h)  + bh );
    h       = (1-z).*h + z.*h_tilde;

    y_hat = tanh(Wy*h + by);        
    dx    = y_hat .* target_scale;

    %x_gru(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx;
    x_gru(:,k+1) = x_kf(:,k+1) + dx;
end
x_gru(:,N) = x_gru(:,N-1);   

fprintf('\n-------- RMSE Accuracy --------\n');
state_labels = {'u', 'w', 'q', 'theta', 'h'};
idx = 2:N;
for j = 1:5
    rmse_kf  = sqrt(mean((x_true(j,idx) - x_kf(j,idx)).^2));
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
