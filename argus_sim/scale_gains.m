function [Kp,Kv,Ki,Ka_ff] = scale_gains(SLPKP,SLVKP,SLVKI,SLAFF,scal_fac,SLVRAT)
%Rescales the gains to their unit in the Argus Simulink model
Kp = SLPKP;
Kv = SLVKP*scal_fac*SLVRAT/(1024);
Ki = SLVKI/2^16;
Ka_ff = SLAFF/2e7;

end

