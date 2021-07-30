% Runs the Argus Simulink model. Parameters have to be set externaly
%% Calculate RPOS
RPOS = s_curve(stepsize,vmax,acc,jerk,Ctime);
RPOS(:,2) = RPOS(:,2)*1e3;

%% Rescale Gains
[Kp,Kv,Ki,Ka_ff] = scale_gains(SLPKP,SLVKP,SLVKI,SLAFF,scal_fac,SLVRAT);
param = [Kp,Kv,Ki];

%% Run Simulation
[T_settle,TV] = costfun_ARGUS_sim(param,RPOS,Ts);