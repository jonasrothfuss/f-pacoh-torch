% clc

%% Model Setup (Configuration start)
%Ts   = 5e-5;  % Sampling Time
%Ctime = 1e-3; % Sampling time function generator

%xstart = 0; % mm
%xstop = 100; % mm
%travel = (xstop-xstart)/1000; % m
%jerk = 1e3; % m/s^3
%acc = 10; % m/s^2
%vmax = 0.1; % m/s

% Controller
%SLPKP  = 200;          % Position Loop Proportional Gain (Spii: 1200)(default: 20)
%SLVKP  = 600;          % Velocity Loop Proportional Gain (Spii: 200)(default: 100)
%SLVKI  = 1000;         % Velocity Loop Integrator Gain (Spii: 1000)(default: 100)
%SLAFF = 0;	           % Acceleration Feedforward Gain (1 compensates nearly perfectly)

%% Configuration end

%RPOS = s_curve(stepsize,vmax,acc,jerk,Ctime);
%RPOS(:,2) = RPOS(:,2)*1e3;

% Miscelaneous definitions
% Ts = 1/20000; % DSP sampling time

load('cogging_data.mat')
cog_pos = cog_pos*1000;

% Flag to set source of acceleration feed-forward:
% 1 - downsampled according to CTIME
% 0 - derivative at 20[kHz] of RVEL
% AccFFSource = 0;            

% Plant Model
s = tf('s')/2/pi;
%y0 = RPOS(1,2);

K = 5.6234e+5;
% K = 2.3961e+6;
% fnotch_c = 2500;
% fnotch_w = 800;
fpole = 10000;
fdeadt = 0.002/2/pi;
fres_w11 = 0.1;
fres_w12 = 0.1;
fres11 = 390/(sqrt(1-2*fres_w11^2));
fres12 = 400/(sqrt(1-2*fres_w12^2));
fres_w21 = 0.03;
fres_w22 = 0.05;
fres21 = 475/(sqrt(1-2*fres_w21^2)); % y dependent
fres22 = 500/(sqrt(1-2*fres_w22^2)); % y dependent
fres_w31 = 0.03;
fres_w32 = 0.06;
fres31 = 690/(sqrt(1-2*fres_w31^2)); %x dependent (slightly), y dependent
fres32 = 800/(sqrt(1-2*fres_w32^2)); % x dependent (slightly), y dependent
fres_w41 = 0.03;
fres_w42 = 0.04;
fres41 = 870/(sqrt(1-2*fres_w41^2)); % avg
fres42 = 900/(sqrt(1-2*fres_w42^2)); % avg
fres_w51 = 0.03;
fres_w52 = 0.06;
fres51 = 1050/(sqrt(1-2*fres_w51^2)); % avg
fres52 = 1100/(sqrt(1-2*fres_w52^2)); % avg

% sys_int = tf(K,[1, 0 0]);
% sys_notch2 = tf([1 0 fnotch_c^2],[1 fnotch_w fnotch_c^2]); %Notch filter
sys_pole1 = tf([1],[1/fpole 1],'inputdelay',fdeadt);
% sys_res1 = tf([1/(fres11^2) 2*fres_w11/(fres11) 1],[1/(fres12^2) 2*fres_w12/(fres12) 1]);
% sys_res2 = tf([1/(fres21^2) 2*fres_w21/(fres21) 1],[1/(fres22^2) 2*fres_w22/(fres22) 1]);
% sys_res3 = tf([1/(fres31^2) 2*fres_w31/(fres31) 1],[1/(fres32^2) 2*fres_w32/(fres32) 1]);
% sys_res4 = tf([1/(fres41^2) 2*fres_w41/(fres41) 1],[1/(fres42^2) 2*fres_w42/(fres42) 1]);
% sys_res5 = tf([1/(fres51^2) 2*fres_w51/(fres51) 1],[1/(fres52^2) 2*fres_w52/(fres52) 1]);

sys_int = K/s^2;
% sys_notch2 = (s^2/fnotch_c^2 + 1)/(s^2/fnotch_c^2 + fnotch_w*s/fnotch_c^2 + 1);
sys_res1 = (s^2/fres11^2 + 2*fres_w11/fres11*s +1)/(s^2/fres12^2 + 2*fres_w12/fres12*s +1);
sys_res2 = (s^2/fres21^2 + 2*fres_w21/fres21*s +1)/(s^2/fres22^2 + 2*fres_w22/fres22*s +1);
sys_res3 = (s^2/fres31^2 + 2*fres_w31/fres31*s +1)/(s^2/fres32^2 + 2*fres_w32/fres32*s +1);
sys_res4 = (s^2/fres41^2 + 2*fres_w41/fres41*s +1)/(s^2/fres42^2 + 2*fres_w42/fres42*s +1);
sys_res5 = (s^2/fres51^2 + 2*fres_w51/fres51*s +1)/(s^2/fres52^2 + 2*fres_w52/fres52*s +1);


sys_plant = sys_int*sys_res1*sys_res2*sys_res3*sys_res4*sys_res5*sys_pole1;
sys_corr = (7.3e-5*s+1)^2;

% MFLAGS
% MFLAGS1 = 0;            % 0=closed loop, 1=open loop

COUNTS_PER_MM = 2500*2^12;  %  Encoder Resolution [counts/mm] 
SLVRAT = 1;            % Gear ratio
XVEL = COUNTS_PER_MM*3000; %COUNTS_PER_MM*250;  % Max Velocity (default: COUNTS_PER_MM*1000 [counts/sec])
SLVLI  = 50;            % Velocity Loop Integrator Limit [percentage] (Spii: 40)(default: 100)

SLVSOF = 1000;          % Second Order Low-Pass Filter Bandwidth (Spii: 4000)(default: 700)
SLVSOFD = 0.707;         % Second Order Low-Pass Filter Damping (Spii: 0.71)(default: 0.707)
MFLAGS15 = 0;           % 0=enable, 1 = disable low-pass filter

SLVNFRQ = 520;          % Notch Filter Frequency (default: 400)
SLVNWID = 30;           % Notch Filter Width (default: 30)
SLVNATT = 2;            % Notch Filter Attenuation (default: 5)
MFLAGS14 = 1;           % 0=disable, 1 = enable Notch filter 

SLVB0NF = 750;
SLVB0ND = 0.1;
SLVB0DF = 750;
SLVB0DD = 0.2;
MFLAGS16 = 1;       

SLVB1NF = 500;
SLVB1ND = 0.2;
SLVB1DF = 450;
SLVB1DD = 0.2;
MFLAGS26 = 0;   

% SLFRCD = 0;            % Dynamic Friction Feedforward (default: 0)
% DCOM = 0;         % Constant Drive Command (2^15 = 10A)(default: 0)
% XCUR  = 100;            % Torque limit [percentage] (default: 30)

%  Feedforwards and Scale Factor
% V_FF = min(2^21/Ts/XVEL,1/Ts);
% V_SF = V_FF*SLVRAT;                   
% A_FF = SLAFF;                  
% FRIC_DYN = SLFRCD/XVEL/100*32766;

scal_fac = 2^21/XVEL;      % Scaling factor Gains
Kamp = 10;                 % Current Amplifier gain, A/V
Km = 36.6;                 % Motor Force const. gemessen! (N/Vin (DOUT)]
Lm = 0.0043;            % Induktanz [H] (Terminal to Terminal) (datasheet: 0.0091)                
Mm = 2.6;              % Motormasse [kg]
q = 2^15/10;
IKP = 600;
cor_fac = 1;



Kv_ff = 1;%*scal_fac;
% Ka_ff = SLAFF/2e7;
% offizielle Skalierung:
% Ka_ff = SLAFF/2e7/32766;
% manuelles Optimieren:
% Ka_ff = 4.5e-8;

V_LI= SLVLI/100;
V_SF = 1;%scal_fac*SLVRAT;

% Cogging
scale_cog = 1*scal_fac*Km*Lm; %1*scal_fac/(Km*Kamp*Mm); % scaling factor for cogging (might depend on motor mass: scale_cog = 1/Mm)
pos_os = 8.5; %[in mm] 40 works really well! 8.5 is better

% Position Loop  
%Kp = SLPKP;%SLPKP*scal_fac;

% Velocity Loop  
%Kv = SLVKP*scal_fac*SLVRAT/(1024);%SLVKP/2^10;%SLVKP*scal_fac*SLVRAT;
%Ki = SLVKI/2^16;%SLVKI/2^16*Ts/5e-5;%SLVKI*Ts;

%% Run model
%param = [Kp,Kv,Ki];
%[T_settle,TV] = costfun_ARGUS_sim(param,RPOS,Ts);

%% Plot resulted signal

% figure(1), plot(FPOS.Time, FPOS.Data, RPOS(:,1), RPOS(:,2))
% title('RPOS and FPOS')
% 
% figure(2), plot(PE.Time, PE.Data)
% title('PE')