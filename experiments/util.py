import sys
import os
import json
import numpy as np
from itertools import cycle
from config import BASE_DIR, DATA_DIR

""" Custom Logger """
import sys

class Logger:

    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

""" Async executer """
import multiprocessing

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

def _start_process(target, args=None):
    if args:
        p = multiprocessing.Process(target=target, args=args)
    else:
        p = multiprocessing.Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass


""" Command generators """

def generate_base_command(module, flags=None, unbuffered=True):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    if unbuffered:
        base_cmd = interpreter_script + ' -u ' + base_exp_script
    else:
        base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag, setting in flags.items():
            if type(setting) == bool:
                if setting:
                    base_cmd += f" --{flag}"
            else:
                base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list, num_cpus=1, num_gpus=1, dry=False, n_hosts=1, mem=6000, long=False, mode='local',
                          experiment_name='run', promt=True, oci_shape='VM.Standard.E3.Flex.16', run_ids=None):

    if mode == 'leonhard':
        cluster_cmds = []
        bsub_cmd = 'bsub ' + \
                   f'-W {23 if long else 3}:59 ' + \
                   f'-R "rusage[mem={mem}]" ' + \
                   f'-R "rusage[ngpus_excl_p={num_gpus}]" ' + \
                   f'-n {num_cpus} ' + \
                   f'-R "span[hosts={n_hosts}]" '

        for python_cmd in command_list:
            cluster_cmds.append(bsub_cmd + python_cmd)

        if promt:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            for cmd in cluster_cmds:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'oci':
        import oci_launcher.mode as mode
        import oci_launcher.mount as mount
        from datetime import datetime
        from oci_launcher.launch import launch_python

        if promt:
            answer = input(f"About to launch {len(command_list)} OCI instances. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            def launch_command_oci(command, run_id):
                target_file = command.split(" ")[1]
                cmd_args = " ".join(command.split(" ")[2:])

                REMOTE_PYTHON_INTERPRETER = '/home/ubuntu/miniconda3/envs/meta-bo/bin/python -u'

                # import oci_launcher.ssh as ssh
                # from oci_launcher.config import SSH_KEY_FILE
                # mode_ssh = mode.SSH(
                # credentials=ssh.SSHCredentials(hostname='152.67.66.222', username='ubuntu', identity_file=SSH_KEY_FILE),
                # )

                mode_oci = mode.OCIMode(oci_shape=oci_shape, run_id=run_id)

                MODE = mode_oci

                mounts = [
                    mount.MountLocal(local_dir=BASE_DIR + '/', mount_point='/home/ubuntu/meta-bo-febo',
                                     pythonpath=True, filter_dir=['runs/*']),
                    mount.MountDropbox(mount_point='/home/ubuntu/meta-bo-febo/runs', dropbox_path='/meta-bo-febo_runs',
                                       output=True,
                                       skip_folders='runs')
                ]

                launch_python(target=target_file, mount_points=mounts,
                              target_mount_dir='/home/ubuntu/meta-bo-febo',
                              mode=MODE, args=cmd_args, verbose=True,
                              python_cmd=REMOTE_PYTHON_INTERPRETER, tmux=True,
                              stdout_file='/home/ubuntu/meta-bo-febo/runs/%s.out' % run_id,
                              install_packages=['psutil'])

            if run_ids is None:
                run_ids = ['%s_%s_%s_%s'%(experiment_name, datetime.now().strftime("%d-%m-%y_%H:%M:%S"),
                                          str(abs(cmd.__hash__())), np.random.randint(10**4)) for cmd in command_list]
            else:
                run_ids = ['%s_%s_%s' % (exp_id, datetime.now().strftime("%d-%m-%y_%H:%M:%S"), np.random.randint(10**4)) for exp_id in run_ids]
            assert len(run_ids) == len(command_list)
            exec = AsyncExecutor(n_jobs=len(command_list))
            exec.run(launch_command_oci, command_list, run_ids)

    elif mode == 'local':
        if promt:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)

    elif mode == 'local_async':
        if promt:
            answer = input(f"About to launch {len(command_list)} commands in {num_cpus} local processes. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if answer == 'yes':
            if dry:
                for cmd in command_list:
                    print(cmd)
            else:
                exec = AsyncExecutor(n_jobs=num_cpus)
                cmd_exec_fun = lambda cmd: os.system(cmd)
                exec.run(cmd_exec_fun, command_list)
    else:
        raise NotImplementedError

""" Hashing and Encoding dicts to JSON """

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def hash_dict(d):
    return str(abs(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).__hash__()))


if __name__ == '__main__':
    load_meta_data('/home/jonasrothfuss/Dropbox/Eigene_Dateien/ETH/02_Projects/16_Inspire_Meta_BO/inspire_safe_meta_bo/data/gp_ucb_meta_data/RandomMixtureMetaEnv_20_tasks_20_samples.json')

