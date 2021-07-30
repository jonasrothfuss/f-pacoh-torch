function [val,con1] = costfun_ARGUS_sim(param,RPOS,Ts)
% Calculates different performance metrics for the "X_axis_new" model w.r.t
% the control parameters param and the reference position signal RPOS

global Kp Kv Ki
Kp = param(1);
Kv = param(2);
Ki = param(3);

bounds = [2.5e-6,5e-6,10e-6,20e-6]; % in mm
stepsize = abs(RPOS(end,2)-RPOS(1,2));

% Run Simulink model
simout = sim('Argus_Simulink');
FPOS = FPOSsim.signals.values;
PE = PEsim.signals.values;
ICOM = ICOMsim.signals.values;

tol = 1e-9;

xstart = RPOS(1,2);

idx_moving = find(all([abs(RPOS(:,2)-xstart) < abs(stepsize)-tol, abs(RPOS(:,2)-xstart) > 0+tol],2));
idx_standstill = find(any([abs(RPOS(:,2)-xstart-stepsize) < tol, abs(RPOS(:,2)-xstart) < tol],2));

% Move time
T_move = RPOS(idx_moving(end)+1,1)-RPOS(idx_moving(1),1);

% Settling time
idx_set = zeros(length(bounds),1);
ss_pos = FPOS(end); % Steady state position
for i = 1:1:length(bounds)
    idx_set(i) = find(abs(PE) > bounds(i),1,'last');
end
T_settle = Ts*mean(idx_set);% - RPOS(idx_moving(1),1);

% TV
run_pos_der = [0;diff(PE(idx_moving(end)+1:end))];
TV = Ts*sum(abs(run_pos_der));

val = T_settle-T_move;
con1 = TV;

end

