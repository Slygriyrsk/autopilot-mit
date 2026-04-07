clear; clc; close all;

%aircraft model A,B,C,D
Xu = -1.982e3; Xw = 4.025e3;
Zu = -2.595e4; Zw = -9.030e4; Zq = -4.524e5; Zwd = 1.909e3;
Mu = 1.593e4; Mw = -1.563e5; Mq = -1.521e7; Mwd = -1.702e4;

g = 9.81; theta0 = 0;
S = 511; cbar = 8.324;
U0 = 235.9;
Iyy = 0.449e8;                 
m = 2.83176e6 / g;             
rho = 0.3045;
Xdp = 0.3 * m * g;             
Zdp = 0;
Mdp = 0;
Xde = -3.818e-6 * (0.5 * rho * U0^2 * S);
Zde = -0.3648    * (0.5 * rho * U0^2 * S);
Mde = -1.444     * (0.5 * rho * U0^2 * S * cbar);

%state space matrices
A = [ Xu/m,           Xw/m,               0,            -g*cos(theta0);
      (Zu)/(m-Zwd),  (Zw)/(m-Zwd), (Zq + m*U0)/(m-Zwd), -m*g*sin(theta0)/(m-Zwd);
      (Mu + Zu*Mwd/(m-Zwd))/Iyy, (Mw + Zw*Mwd/(m-Zwd))/Iyy, ...
         (Mq + (Zq + m*U0)*Mwd/(m-Zwd))/Iyy, -m*g*sin(theta0)*Mwd/( (m-Zwd)*Iyy );
      0, 0, 1, 0 ];

%B is 4x2 [Δe;Δp]
B = [ Xde/m,              Xdp/m;
      Zde/(m-Zwd),        Zdp/(m-Zwd);
      (Mde + Zde*Mwd/(m-Zwd))/Iyy, (Mdp + Zdp*Mwd/(m-Zwd))/Iyy;
      0, 0 ];

% Form: Bw * [Ug; Wg; Qg], where the gust will affect the aircraft motion 
%on all aspects like fwd gust, vertical gust and pitch rate gust
Bw = [ -Xu/m,                     -Xw/m,                0;
       -Zu/(m-Zwd),               -Zw/(m-Zwd),          0;
       (-Mu + (-Zu*Mwd/(m-Zwd)))/Iyy, (-Mw + (-Zw*Mwd/(m-Zwd)))/Iyy, -Mq/Iyy;
        0,                         0,                   0 ];


%pitch rate(q) = C([0 0 1 0]) else pitch angle(theta) = C([0 0 0 1])
C = [0 0 1 1]; %now i am only outputting theta

%on gnd reference model same as in air
A_gnd=A;
B_gnd=B;
C_gnd=C;

%10 ms sampling time
Ts = 0.01;

B_combined=[B, Bw];  %the first 2 rows are inputs and last 3 rows are Bw
syscont=ss(A,B_combined,C,0);
sysdes=c2d(syscont,Ts);

%descrete time response for in-air and on-gnd
%sysd = c2d(ss(A,B,C,0),Ts);
Ad = sysdes.A; 
Bd = sysdes.B(:,1:2); 
Bwd= sysdes.B(:,3:5);
Cd = sysdes.C;

sysd_g = c2d(ss(A_gnd,B_gnd,C_gnd,0),Ts);
Ad_g = sysd_g.A;
Bd_g = sysd_g.B;
Cd_g = sysd_g.C;

N = 12000; %no of samples
t = (0:N-1)*Ts;
rad2deg=180/pi; %because the signal we measure is degree in the flight

%fault time period from 5sec to 7 sec
fault_start = 500;
fault_end   = 700;

Q = diag([1e-4, 1e-4, 1e-6, 1e-6]);
R = (25e-3)^2;

Xair = zeros(4,1);
Xgnd = zeros(4,1);
Pg = eye(4);

%u=(1.5*pi/180) * ones(1,N); %theta_Ref not u into radians
u = zeros(1, N);
u(1:3000) = 1.5*(pi/180);
u(3001:6000) = -1.5*(pi/180);
u(6001:9000)=1.5*(pi/180);
u(9001:N)=-1.5*(pi/180);
dp = zeros(1, N); %for thrust elevation

x_air=zeros(4,N);
x_ground=zeros(4,N);

%to store the output and parameters
store_u=zeros(1,N);
store_w=zeros(1,N);
store_q=zeros(1,N);
store_theta=zeros(1,N);

%to store inputs
store_delta=zeros(1,N);
store_thrust=zeros(1,N);

%to store the kalman gain
store_k=zeros(4,N);

%to store the residuals
residualgnd=zeros(1,N);

%to store the dequantized ops like theta or q
y_dequantize=zeros(1,N);

residual_buffer = zeros(1, 100); % To store recent innovations within the 100 window size
R_original = R;

%for p controller only, where i will follow the feedback flow of our o/p
%(theta) to our i/p and then multiply by Kp and that thing will be fed to
%our aircraft model

%our plant transfer function would be the model we have A,B,C,Dmatrice
sys = ss(A, B, C, 0);
%assigned the names for inputs here
sys.InputName = {'Elevator', 'Thrust'};
sys.OutputName = {'Pitch Angle'};
G_all=tf(sys);%aircraft model transfer function with i/p as delta e, disp(G_all)
G=G_all(1,1); %this is my G(s) for input1 (delta e) to output1 for (theta), dcgain(G) in cmd window

%kp must be set negative because the Mde (pitching moment due to elevator)
%itself -ve and will work opposite means if delta e = (kp*error) > 0, then
%the tail will lift and nose go down and as the aircraft as well.
Kp = -2;
Ki = -0.8636;
Kd = 0.5; %this is q i am putting inside
%Cp=pid(Kp,Ki,Kd); %propotional gain

%T=feedback(Cp*G, 1);

i_err=0;

for k=1:N
    %PID controller here below
    %my theta ref is u only
    theta_air=Xair(4); %theta
    q_air=Xair(3);%pitch rate
    p_err=u(k)-theta_air;
    i_err=i_err+p_err*Ts; %0.01 is my sampling time
    
    %now this delta will go to my plant model G(s) as input
    %deltaE=Kp*p_err+Ki*i_err+Kd*q_air;
    %limiting delta E so it won't jump unrealistic
    deltaE_unsat = Kp * p_err + Ki * i_err + Kd * q_air;

    % anti-windup and saturate
    deltaE = max(min(deltaE_unsat, 0.5), -0.5);
    if abs(deltaE - deltaE_unsat) > 1e-9
        i_err = i_err - p_err * Ts;
    end

    if k>=fault_start && k<=fault_end
        pitchF=Xair(3)+0.0;
        thetaF=Xair(4)+0.0;
        %deltaF=u(k)+0.05;
        deltaF=deltaE+0.0;
        thrustF=dp(k)+0.0;
    else
        pitchF=Xair(3);
        thetaF=Xair(4);
        %deltaF=u(k);
        deltaF=deltaE;
        thrustF=dp(k);
    end
    
    u_vec=[deltaF; thrustF];
    gust=[10; 0; 0]; %[Ug,Wg,Qg]
    Xair=Ad*Xair+Bd*u_vec+Bwd*gust;
    Yair=Cd*Xair;

    uq=quant7(Xair(1),[-10,10]);
    wq=quant7(Xair(2),[-10,10]);
    qq=quant7(pitchF, [-1,1]);
    thetaq=quant7(thetaF,[-0.5,0.5]);

    deltaq=quant7(deltaF, [-0.3, 0.3]);
    thrustq=quant7(thrustF, [-0.2, 0.2]);

    %now store it
    store_u(k)=uq;
    store_w(k)=wq;
    store_q(k)=qq;
    store_theta(k)=thetaq;
    store_delta(k)=deltaq;
    store_thrust(k)=thrustq;

    word = pack32(uq,wq,qq,thetaq,deltaq);
    [uq,wq,qq,thetaq,deltaq]=unpack32(word);
    
    yq=dequant7(thetaq,[-0.5,0.5]);
    y_dequantize(k)=yq;

    %now on the on ground model
    %u_vec_gnd=[u(k); dp(k)];
    u_vec_gnd=[deltaF; dp(k)];
    Xpred=Ad_g*Xgnd+Bd_g*u_vec_gnd;
    Ppred=Ad_g*Pg*Ad_g'+Q;

    residual=yq-Cd_g*Xpred;
    residual_buffer = [residual_buffer(2:end), residual];
    moving_resid_var = var(residual_buffer);

    theoretical_var = Cd_g * Ppred * Cd_g' + R_original;
    if moving_resid_var > theoretical_var
        %R_adaptive = moving_resid_var;
        R_adaptive=max(R_original, min(moving_resid_var, R_original*100));
    else
        R_adaptive = R_original;
    end

    K = Ppred * Cd_g' / (Cd_g * Ppred * Cd_g' + R_adaptive);
    Xgnd = Xpred + K * residual; 
    Pg = (eye(4) - K * Cd_g) * Ppred;

    store_k(:,k)=K;
    x_air(:,k)=Xair;
    x_ground(:,k)=Xgnd;
    residualgnd(k)=residual;
end

%{
figure;
step(T,30);
title("close loop pitch angle control");
grid on;
%}

%this is our aircraft pitch control
figure('Name', 'Aircraft pitch control', 'NumberTitle', 'off');

%pitch angle theta
subplot(3,1,1);
plot(t, u * (pi/180), 'r--', 'LineWidth', 2); hold on; 
plot(t, x_air(4,:) * (pi/180), 'b', 'LineWidth', 1.5); 
grid on;
ylabel('Pitch Angle (rad)');
legend('Reference R(s)', 'Actual \theta', 'Location', 'best');
title('Step 1: Tune Gains to Match Actual to Reference');

%fwd vel
subplot(3,1,2);
plot(t, x_air(1,:)*2, 'LineWidth', 1.5, 'Color', 'b');
grid on; xlim([min(t) max(t)]); % Dynamic limit to ensure data is visible
ylabel('u (knots)');
title('Forward Velocity');

%vertical vel
subplot(3,1,3);
plot(t, x_air(2,:)*2, 'LineWidth', 1.5, 'Color', 'r');
grid on; xlim([min(t) max(t)]);
ylabel('w (knots)');
xlabel('Time (sec)');
title('Vertical Velocity');

%below are the velocity details
figure('Name', 'Velocity Detail', 'Color', 'w');

subplot(3,1,1);
plot(t, rad2deg*x_air(4,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, rad2deg*u, 'r--', 'LineWidth', 1);
ylabel('\theta (deg)');
title('Doublet Velocity Analysis');
grid on; xlim([25 65]);
legend('Actual \theta', 'Ref');

subplot(3,1,2);
plot(t, x_air(1,:)*2, 'Color', [0 0.5 0], 'LineWidth', 1.5);
ylabel('u (knots)');
grid on; xlim([25 65]);

subplot(3,1,3);
plot(t, x_air(2,:)*2, 'm', 'LineWidth', 1.5);
ylabel('w (knots)');
xlabel('Time (sec)');
grid on; xlim([25 65]);


step_info = stepinfo(x_air(4,:), t, u(end));
fprintf('Rise Time: %.3f sec\n', step_info.RiseTime);
fprintf('Overshoot: %.2f%%\n', step_info.Overshoot);
fprintf('Settling Time: %.3f sec\n', step_info.SettlingTime);
