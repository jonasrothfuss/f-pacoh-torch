function Refpos= s_curve(travel,speed,acc,jerk,ts)
% S_CURVE Motion profile generator for a 3rd order polynomial S-curve.
% inputs:
% - travel: relative travel distance [m]
% - speed: maximum velocity [m/s]
% - acc: maximum acceleration [m/s^2]
% - jerk: maximum jerk [m/s^3]
% - ts: sampling period [s] (optional)
% outputs:
% - final_time: duration of the motion profile [s]
%

% input processing
if nargin < 5
    ts = 1/20e3;            % default sampling period [s] -> 20 kHz
end
dx = travel; % relative position command [m]
vm = speed;  % maximum speed [m/s]
am = acc;    % maximum acceleration [m/s^2]
jm = jerk;   % maximum jerk [m/s^3]

% variables storing values during integration process
x  = 0;     % position
t  = 0;     % time

% vectors storing the final motion profile
tt = [];    % time
xx = [];    % position
vv = [];    % velocity
aa = [];    % acceleration
jj = [];    % jerk
ss = [];    % phase (goes from 1 to 7)

% evaluate phase durations, i.e. t1 for phase 1, t2 for phase 2, etc.
t1 = min([(0.5*dx/jm)^(1/3), (vm/jm)^(1/2), am/jm]);
am = min([am, jm*t1]);
t2 = min([(vm-am*t1)/am, (-3/2*am*t1+sqrt((3/2*am*t1)^2 -4*(0.5*am)*(am*t1^2-dx/2)))/am]);
t3 = t1;
vm = min([vm, jm*t1^2 + jm*t1*t2]);
t4 = (dx-(2*(am*t1^2 + 1/2*am*t2^2 + 3/2*am*t1*t2)))/vm;
t5 = t1;
t6 = t2;
t7 = t1;

% calculate position, velocity, accelereation at the end of each phase,
% which is needed to gerenate the motion profile with numerical integration
[x1,v1,a1] = CalculatePhase(t1, jm, 0, 0, 0);
[x2,v2,a2] = CalculatePhase(t2, 0, a1, v1, x1);
[x3,v3,~ ] = CalculatePhase(t3, -jm, a2, v2, x2);
[x4,v4,a4] = CalculatePhase(t4, 0, 0, v3, x3); 
[x5,v5,a5] = CalculatePhase(t5, -jm, 0, v4, x4);
[x6,v6,a6] = CalculatePhase(t6, 0, a5, v5, x5);     

% convert the phase duration into the times at the end of each phase
t2 = t1+t2;
t3 = t2+t3;
t4 = t3+t4;
t5 = t4+t5;
t6 = t5+t6;
t7 = t6+t7;
counter = 0;

% calculate profile
while (x < dx)
    counter = counter + 1;
%     disp(counter)
    if (t < t1)      % phase 1 (j=jm, a>0)
      s = 1;
      [x,v,a,j]=CalculatePhase(t, jm, 0, 0, 0);
      
    elseif (t < t2)  % phase 2 (j=0, a=am)
      s = 2;
      [x,v,a,j]=CalculatePhase(t-t1, 0, am, v1, x1);
      
    elseif (t < t3)  % phase 3  (j=-jm, a>0)
      s = 3;
      [x,v,a,j]=CalculatePhase(t-t2, -jm, a2, v2, x2);
      
    elseif (t < t4)  % phase 4 (j=0, a=0, v=vmax)
      s = 4;
      [x,v,a,j]=CalculatePhase(t-t3, 0, 0, vm, x3);
      
    elseif (t < t5)  % phase 5 (j=-jm, a<0)
      s = 5;
      [x,v,a,j]=CalculatePhase(t-t4, -jm, a4, v4, x4);
      
    elseif (t < t6)  % phase 6 (j=0; a=-am)
      s = 6;
      [x,v,a,j]=CalculatePhase(t-t5, 0, -am, v5, x5);
      
    elseif (t < t7)  % phase 7 (j=jm, a<0)
      s = 7;
      [x,v,a,j]=CalculatePhase(t-t6, jm, a6, v6, x6);
      
    else  % end of trajectory
        x = dx;
        s = 0;
    end

    % Concatenate the new values
    tt = [tt,t];
    xx = [xx,x];
    vv = [vv,v];
    aa = [aa,a];
    jj = [jj,j];
    ss = [ss,s];
    t = t+ts;
end

% plot results
n=5;
% figure(1)
% subplot(n,1,1);
% plot(tt,xx); grid on; ylabel('x');
% title('S-Curve');
% subplot(n,1,2);
% plot(tt,vv); grid on; ylabel('v');
% subplot(n,1,3);
% plot(tt,aa); grid on; ylabel('a');
% subplot(n,1,4);
% plot(tt,jj); grid on; ylabel('j');
% subplot(n,1,5);
% plot(tt,ss); grid on; ylabel('phase');

time = tt;

%save to text file
Refpos = [tt',xx'];
% save('profile.txt','Refpos','-ascii')
% save('profile.mat','time','xx','vv','aa','jj')

% The output is the final time
final_time=max(tt);
end

function [x,v,a,j]=CalculatePhase(dt, j0, a0, v0, x0)
% Calculate the motion profile through integration. Note that the jerk is
% assumed to be constant.
% Inputs:
% - dt: time of integration [s]
% - j0: constant jerk during integration period [m/s^3]
% - a0: initial acceleration [m/s^2]
% - v0: initial velocity [m/s]
% - x0: initial position [m]
% Outputs:
% - x: final position [m]
% - v: final velocity [m/s]
% - a: final acceleration [m/s^2]
% - j: constant jerk [m/s^3]
%

    j = j0;
    a = j0 * dt + a0;
    v = 1/2*j0*dt^2 + a0*dt + v0;
    x = 1/6*j0*dt^3 + 1/2*a0*dt^2 + v0*dt + x0;
end
