import matlab
import matlab.engine
import time

def RunSim_Argus(eng,parameters):
    # Load parameters into the MATLAB workspace
    eng.workspace['Ts'] = matlab.double([parameters['Ts']])
    eng.workspace['Ctime'] = matlab.double([parameters['Ctime']])
    eng.workspace['stepsize'] = matlab.double([parameters['stepsize']])
    eng.workspace['jerk'] = matlab.double([parameters['jerk']])
    eng.workspace['acc'] = matlab.double([parameters['acc']])
    eng.workspace['vmax'] = matlab.double([parameters['vmax']])
    eng.workspace['SLPKP'] = matlab.double([parameters['SLPKP']])
    eng.workspace['SLVKP'] = matlab.double([parameters['SLVKP']])
    eng.workspace['SLVKI'] = matlab.double([parameters['SLVKI']])
    eng.workspace['SLAFF'] = matlab.double([parameters['SLAFF']])

    # Run simulation
    eng.Argus_RunSimulation(nargout=0)

    # Load evaluation results from MATLAB workspace to Python
    T_settle = eng.workspace['T_settle']
    TV = eng.workspace['TV']

    return T_settle,TV

if __name__ == "__main__":
    # create MATLAB session and load static parameters for simulation
    t_eng_start = time.time()
    eng = matlab.engine.start_matlab()
    t_eng_stop = time.time()
    print("Engine setup time: ",t_eng_stop-t_eng_start)

    t_paramset_start = time.time()
    eng.Argus_Parameters(nargout=0)
    t_paramset_stop = time.time()
    print("Parameter setting time", t_paramset_stop-t_paramset_start)

    t_simulation_start = time.time()
    parameters = {'Ts': 5e-5, # Sampling time simulation in s
                  'Ctime': 1e-3, # Sampling time function generator (reference position - RPOS) in s
                  'stepsize': 0.1, # RPOS stepsize in m (max. feasible 0.2)
                  'jerk': 1e3, # RPOS max. jerk in m/(s^3)
                  'acc': 10, # RPOS max. acceleration in m/(s^2)
                  'vmax': 0.1, # RPOS maximum velocity in m/s
                  'SLPKP': 200, # Proportional gain position controller (nominal 200)
                  'SLVKP': 600, # Proportional gain velocity controller (nominal 600)
                  'SLVKI': 1000, # Integral gain velocity controller (nominal 1000)
                  'SLAFF': 0} # Acceleration feedforward gain

    T_settle, TV = RunSim_Argus(eng, parameters)
    t_simulation_stop = time.time()
    print("Simulation time: ", t_simulation_stop-t_simulation_start)
    print("Settling time: ", T_settle)
    print("Total variation: ", TV)

    eng.quit()