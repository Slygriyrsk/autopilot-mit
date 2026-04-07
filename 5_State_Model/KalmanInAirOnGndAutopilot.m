%units: speed: m/s, altitude: m, angle: rad
clear; clc; close all;

%mode = 'vel_hold'; %keep the mode as 1,2,3 arr
map = dictionary([1, 2, 3], ["theta_hold", "vel_hold", "alt_hold"]);
current_mode = 2;

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

A = [ Xu/m,           Xw/m,               0,            -g*cos(theta0);
      (Zu)/(m-Zwd),  (Zw)/(m-Zwd), (Zq + m*U0)/(m-Zwd), -m*g*sin(theta0)/(m-Zwd);
      (Mu + Zu*Mwd/(m-Zwd))/Iyy, (Mw + Zw*Mwd/(m-Zwd))/Iyy, (Mq + (Zq + m*U0)*Mwd/(m-Zwd))/Iyy, -m*g*sin(theta0)*Mwd/( (m-Zwd)*Iyy );
      0, 0, 1, 0 ];

A_aug = [A, zeros(4,1);
         0, -1, 0, U0, 0];

B = [ Xde/m,              Xdp/m;
      Zde/(m-Zwd),        Zdp/(m-Zwd);
      (Mde + Zde*Mwd/(m-Zwd))/Iyy, (Mdp + Zdp*Mwd/(m-Zwd))/Iyy;
      0, 0 ];

B_aug = [B;
         0, 0];

Bw = [ -Xu/m,                     -Xw/m,                0;
       -Zu/(m-Zwd),               -Zw/(m-Zwd),          0;
       (-Mu + (-Zu*Mwd/(m-Zwd)))/Iyy, (-Mw + (-Zw*Mwd/(m-Zwd)))/Iyy, -Mq/Iyy;
        0,                         0,                   0 ];

Bw_aug = [Bw;
          0, 0, 0];

C_aug = eye(5);
D_aug = zeros(5, size([B_aug, Bw_aug],2));

Ts = 0.01;
sys_aug = ss(A_aug, [B_aug, Bw_aug], C_aug, D_aug);
sysd_aug = c2d(sys_aug, Ts);

Ad = sysd_aug.A;
Bd = sysd_aug.B(:,1:2);
Bwd = sysd_aug.B(:,3:5);

% On-ground model uses same dynamics but no gust (we will design KF using this)
Ad_gnd = Ad;
Bd_gnd = Bd;
C_gnd = C_aug;

N = 12000;
t = (0:N-1)*Ts;
rad2deg = 180/pi;

u = zeros(1, N);
if isKey(map, current_mode)
    curr_str=map(current_mode);
    switch curr_str
        case "theta_hold"
            u(1:3000) = 0.5;
            u(3001:6000) = -0.5;
            u(6001:9000) = 0.5;
            u(9001:N) = -0.5;
        case "vel_hold"
            u(1:3000) = 1;
            u(3001:6000) = -1;
            u(6001:9000) = 1;
            u(9001:N) = -1;
        case "alt_hold"
            u(1:3000) = 2;
            u(3001:6000) = -2;
            u(6001:9000) = 2;
            u(9001:N) = -2;            
        otherwise
            u(1:3000) = 1;
            u(3001:6000) = -1;
            u(6001:9000) = 1;
            u(9001:N) = -1;
    end
end

Kp_theta = -2; Ki_theta = -0.8; Kd_theta = 0.5;
int_theta = 0;

Kp_alt = 0.005; Ki_alt = 1e-6;
int_alt = 0;

Kp_vel = -0.1; Ki_vel = -0.001;
int_vel = 0;

Kp_thrust = 0.1;
Ki_thrust = 0;
int_thrust = 0;

%cahge the trust limits here
thrust_min = -5;
thrust_max =  5;

delta_max = 0.5;
delta_min = -0.5;

% True in-air state
X = zeros(5,1);

x_store = zeros(5, N);
delta_store = zeros(1, N);

theta_ref_store = zeros(1, N);
alt_ref_store = zeros(1, N);
vel_ref_store = zeros(1, N);
thrust_store = zeros(1, N);

alt_err_store = zeros(1,N);

%kalman on ground
Q_kf = diag([1e-10, 1e-10, 1e-11, 1e-13, 1e-13]); % process noise covariance
%Q_kf = diag([1e-4, 1e-4, 1e-5, 1e-6, 1e-3]);
%R_kf = ((5e-3)^2) * eye(5); % measurement noise covariance
%increase R_kf to make it trust more on the model, tune these values to
%adapt the kalman estimation
R_kf = diag([ (1e-1)^2, (1e-1)^2, (1e-1)^2, (1e-2)^2, (1)^2 ]);
P = eye(5); % initial state covariance
x_hat = zeros(5,1); % initial estimate

k_store = zeros(5,N); % store diagonal of K
k_colalt=zeros(5,N);
k_rowalt=zeros(N,5);
x_hat_store = zeros(5,N); % store estimates
innov_store = zeros(5,N);

rng('default');

%lower triangle of matrix R using cholesky factorization
L_R = chol(R_kf, 'lower');

innov_window = 50;
innov_buf = zeros(5, innov_window);
buf_ptr = 0;
alpha_R = 0.95;

K_full_store = zeros(5,5,N);
Krownorm_store = zeros(5,N);
R_original=R_kf;

gust_start=30;
gust_end=60;

gust_store=zeros(3,N);
gust_est_store = zeros(3, N);
Ad_aug_gnd=blkdiag(Ad_gnd,eye(3));
Bd_aug_gnd=[Bd_gnd; zeros(3,2)];
C_aug_gnd=[C_gnd, zeros(5,3)];
Q_gust=diag([1, 1, 1]);
Q_kf_aug=blkdiag(Q_kf,Q_gust);
P_aug=eye(8);
x_hat_aug=zeros(8,1);

for k = 1:N
    theta_doublet_ref = 5 * (pi/180) * u(k);
    alt_doublet_ref = 10 * u(k);
    vel_doublet_ref =5 * u(k);

    if isKey(map,current_mode)
        mode_str=map(current_mode);
        if strcmp(mode_str, 'theta_hold')
            theta_ref = theta_doublet_ref;
            %alt_ref = X(5);
            alt_ref = x_hat(5);
            %vel_ref = X(1);
            vel_ref = x_hat(1);
        elseif strcmp(mode_str, 'alt_hold')
            alt_ref = alt_doublet_ref;
            alt_err = alt_ref - x_hat(5);
            alt_err_store(k) = alt_err;
            int_alt = int_alt + alt_err * Ts;
            theta_ref = Kp_alt * alt_err + Ki_alt * int_alt;
            theta_ref = max(min(theta_ref, 0.1745), -0.1745);
            %vel_ref = X(1);
            vel_ref = x_hat(1);
        elseif strcmp(mode_str, 'vel_hold')
            vel_ref = vel_doublet_ref;
            vel_err = vel_ref - x_hat(1);
            int_vel = int_vel + vel_err * Ts;
            thrust_unsat = Kp_thrust * vel_err + Ki_thrust * int_vel;
            thrust = max(min(thrust_unsat, thrust_max), thrust_min);
            if abs(thrust - thrust_unsat) > 1e-9
                int_vel = int_vel - vel_err * Ts;
            end
            theta_ref=0;
            %alt_ref = X(5);
            alt_ref = x_hat(5);
        else
            theta_ref = theta_doublet_ref;
            alt_ref = alt_doublet_ref;
            vel_ref = vel_doublet_ref;
        end
    end

    p_err = theta_ref - x_hat(4);
    int_theta = int_theta + p_err * Ts;
    delta_unsat = Kp_theta * p_err + Ki_theta * int_theta + Kd_theta * x_hat(3);
    delta = max(min(delta_unsat, delta_max), delta_min);
    if abs(delta - delta_unsat) > 1e-9
        int_theta = int_theta - p_err * Ts;
    end

    if ~exist('thrust','var')
        thrust = 0;
    end

    u_vec = [delta; thrust];

    %in-air model has gust/disturbance
    w_sine = 10*sin(2*pi*k*Ts);
    u_sine = 15*sin(2*pi*k*Ts);
    if t(k)>=gust_start && t(k)<=gust_end
        gust = [4; w_sine; 0];
    else
        gust = [0; 0; 0];
    end
    
    X = Ad * X + Bd * u_vec + Bwd * gust;

    %measurement from in-air, with noise that is fed into ground KF
    z = X + L_R * randn(5,1);

    %kalman on-gnd without gust and do the prediction, could not use true
    %state because it will be trivial then for the model to estimate, X
    %estimated

    x_pred = Ad_aug_gnd * x_hat_aug + Bd_aug_gnd * u_vec;            
    P_pred = Ad_aug_gnd * P_aug * Ad_aug_gnd' + Q_kf_aug;          

    % measurement z (already computed above): 5x1
    y_tilde = z - C_aug_gnd * x_pred;   % innovation

    % update innovation buffer
    buf_ptr = buf_ptr + 1;
    if buf_ptr > innov_window, buf_ptr = 1; end
    innov_buf(:, buf_ptr) = y_tilde;

    valid_count = min(k, innov_window);
    recent = innov_buf(:, 1:valid_count);
    if valid_count > 2
        emp_innov_cov = cov(recent.');  % 5x5 (if enough samples)
    else
        emp_innov_cov = diag(var(recent,0,2));  % fallback
    end

    % Optional adaptive R (simple constrained update)
    S_theory = C_aug_gnd * P_pred * C_aug_gnd' + R_kf;
    if trace(emp_innov_cov) > trace(S_theory)
        R_new = alpha_R * R_kf + (1-alpha_R) * (emp_innov_cov - C_aug_gnd*P_pred*C_aug_gnd');
        R_new = (R_new + R_new')/2;
        R_new = diag(max(diag(R_new), 1e-8));  % keep diagonal, enforce positive entries
        R_kf = R_new;
    else
        R_kf = alpha_R * R_kf + (1-alpha_R) * R_original;
    end

    % K gain and update (use standard formula)
    S = C_aug_gnd * P_pred * C_aug_gnd' + R_kf;    % 5x5
    K = P_pred * C_aug_gnd' / S;                   % 8x5

    % Update state and covariance (Joseph form)
    x_hat_aug = x_pred + K * y_tilde;
    P_aug = (eye(8) - K * C_aug_gnd) * P_pred * (eye(8) - K * C_aug_gnd)' + K * R_kf * K';
    x_hat = x_hat_aug(1:5);
    delta_store(k) = delta;          % store command for plotting
    thrust_store(k) = thrust;

    % store values
    gust_store(:,k) = gust;                        % true gust (3x1)
    gust_est_store(:,k) = x_hat_aug(6:8);          % store both gust states (2x1)
    x_store(:,k) = X;
    x_hat_store(:,k) = x_hat_aug(1:5);
    innov_store(:,k) = y_tilde;
    k_store(:,k) = diag(K(1:5,1:5));               % keeps your previous extraction

    theta_ref_store(k) = theta_ref;
    alt_ref_store(k) = alt_ref;
    vel_ref_store(k) = vel_ref;
end

figure('Name','Hold comparisons','NumberTitle','off','Units','normalized','Position',[0.05 0.05 0.9 0.85]);

subplot(3,1,1);
plot(t, rad2deg*theta_ref_store, 'r--','LineWidth',1.2); hold on;
plot(t, rad2deg*x_store(4,:), 'b-','LineWidth',1.2);
plot(t, rad2deg*x_hat_store(4,:), 'k:','LineWidth',1.2);
ylabel('\theta (deg)');
legend('Ref \theta','Measured \theta','KF estimate \theta','Location','best');
grid on;
title('Pitch Angle: reference vs measured vs KF estimate');

subplot(3,1,2);
plot(t, alt_ref_store, 'g--','LineWidth',1.2); hold on;
plot(t, x_store(5,:), 'b-','LineWidth',1.2);
plot(t, x_hat_store(5,:), 'm-.','LineWidth',1.2);
ylabel('Altitude (m)');
legend('Ref h','Measured h','KF estimate h','Location','best');
grid on;
title('Altitude: reference vs measured vs KF estimate');

subplot(3,1,3);
plot(t, vel_ref_store, 'r--','LineWidth',1.2); hold on;
plot(t, x_store(1,:), 'b-','LineWidth',1.2);
plot(t, x_hat_store(1,:), 'k:','LineWidth',1.2);
ylabel('Forward speed u (m/s)');
xlabel('Time (s)');
legend('Ref u','Measured u','KF estimate u','Location','best');
grid on;
title('Forward Velocity: reference vs measured vs KF estimate');

figure('Name','Commands','NumberTitle','off');
subplot(2,1,1);
plot(t, delta_store, 'k-','LineWidth',1.2);
ylabel('\delta_e (rad)');
xlabel('Time (s)');
grid on;
title('Elevator command');

subplot(2,1,2);
plot(t, thrust_store, 'm-','LineWidth',1.2);
ylabel('Thrust (normalized)');
xlabel('Time (s)');
grid on;
title('Thrust command');

% Kalman gain plots
figure('Name', 'Kalman Gain Analysis for Wind Gust');
subplot(2,1,1);
plot(t, k_store(1,:), 'r', 'LineWidth', 1.2); hold on;
plot(t, k_store(2,:), 'g', 'LineWidth', 1.2); 
plot(t, k_store(3,:), 'b', 'LineWidth', 1.2);
ylabel('Gain Magnitude');
title('Kalman Gains (Sensitivity to Disturbance)');
legend('K_u (Fwd Vel)', 'K_w (Vertical Vel)', 'K_q (Pitch Rate)');
grid on;
ylim([0, 0.02]);

subplot(2,1,2);
plot(t, k_store(4,:), 'm', 'LineWidth', 1.2); hold on;
plot(t, k_store(5,:), 'k', 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Gain Magnitude');
legend('K_{\theta} (Pitch)', 'K_h (Altitude)');
grid on;
ylim([0,0.01]);

%figure; plot(t, k_colalt'); title('Effect of alt meas on states'); legend('u','w','q','\theta','h');
%figure; plot(t, k_rowalt); title('How measurements influence alt estimate'); legend('z1','z2','z3','z4','z5');

% Estimates vs truth for selected states
figure('Name', 'Kalman Filter State Estimates');
subplot(3,1,1);
plot(t, x_store(1,:), 'r--', 'LineWidth', 1.0); hold on; % True from in-air
plot(t, x_hat_store(1,:), 'k', 'LineWidth', 1.2);           % Estimated (gnd KF)
ylabel('Vel u (m/s)');
legend('True (Air)', 'Estimated (Gnd)');
title('Recovery of Forward Velocity from Gust Interference');
grid on;

subplot(3,1,2);
plot(t, x_store(4,:)*180/pi, 'r--', 'LineWidth', 1.0); hold on;
plot(t, x_hat_store(4,:)*180/pi, 'k', 'LineWidth', 1.2);
ylabel('Theta (deg)');
legend('Ref \theta','Measured \theta','KF estimate \theta','Location','best');
grid on;

subplot(3,1,3);
plot(t, x_store(5,:), 'r--', 'LineWidth', 1.0); hold on;
plot(t, x_hat_store(5,:), 'k', 'LineWidth', 1.2);
ylabel('Altitude (m)');
xlabel('Time (s)');
grid on;

figure;
plot(t, gust_store(1,:), 'LineWidth', 1.5);
hold on;
plot(t, gust_store(2,:), 'LineWidth', 1.5);
plot(t, gust_store(3,:), 'LineWidth', 1.5);
hold off;
xlabel('Time (s)');
ylabel('Gust Value');
title('Gust Input Over Time');
legend('Gust u', 'Gust w', 'Gust q', 'Location', 'best');
grid on;
xlim([0 120]); % Focus on the relevant time window

figure('Name', 'Ground Station: Gust Detection & Identification', 'NumberTitle', 'off');

%fwd gust(u)
subplot(2,1,1);
plot(t, gust_store(1,:), 'r', 'LineWidth', 1.5); hold on;
plot(t, gust_est_store(1,:), 'k--', 'LineWidth', 1.2);
ylabel('u_{gust} (m/s)');
legend('True In-Air Gust', 'Ground-Station Estimate');
title('Detection of Longitudinal Gust (Axial)');
grid on;

%vertical gust (w)
subplot(2,1,2);
plot(t, gust_store(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, gust_est_store(2,:), 'k--', 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('w_{gust} (m/s)');
legend('True In-Air Gust', 'Ground-Station Estimate');
title('Detection of Vertical Gust (Heave)');
grid on;

%logging the status of on ground autopilot
% Post-simulation analysis print
max_err_u = max(abs(gust_store(1,:) - gust_est_store(1,:)));
fprintf('--- Ground Autopilot Performance Report ---\n');
fprintf('Ground Station tracked In-Air Forward Gust with Max Error: %.4f m/s\n', max_err_u);
if max_err_u < 0.5
    fprintf('Status: SUCCESS - Ground System accurately identified In-Air disturbance.\n');
else
    fprintf('Status: WARNING - High estimation error. Adjust Q_gust tuning.\n');
end
