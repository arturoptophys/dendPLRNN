from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py
    
    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.
    """
    args = []
    args.append(Argument('experiment', ['MansourData']))
    args.append(Argument('data_path', ['C:\\Users\\as153\\Documents\\Python\\dendPLRNN\\BPTT_TF\\Experiments\\myData\\MansourM003_201111_res25_kernelS_20_norm_1.npy']))
    #args.append(Argument('experiment', ['TestRunFreiPoseData']))
    #args.append(Argument('data_path', ['/home/artur.schneider/Data2Test/datasets/Rat480_190823_res100_kernelS_5_norm_1.npy']))
    #args.append(Argument('experiment', ['TestRunFreiPoseBEHData']))
    #args.append(Argument('data_path', ['O:\\archive\\users\\as153\\Durstewitz\\FreiPose\\BEHRat480_190823_res30_kernelS_4_trafo_Local_norm_1.npy']))
    args.append(Argument('use_gpu', [0])) # may wanna use gpu here
    args.append(Argument('dim_z', [150,300], add_to_name_as="M"))
    args.append(Argument('n_bases', [100], add_to_name_as="B"))
    args.append(Argument('fix_obs_model', [1])) # keep at one Projects overblown dimentions back to input dims
    args.append(Argument('mean_centering', [0])) # not needed for denrd-PLRNN
    args.append(Argument('learn_z0', [1]))
    args.append(Argument('n_epochs', [3000]))
    args.append(Argument('teacher_forcing_interval', [5,15], add_to_name_as="tau"))
    args.append(Argument('seq_len', [100,500], add_to_name_as="T"))      #200 -500 is a good heuristic
    args.append(Argument('latent_model', ['dendr-PLRNN'], add_to_name_as="Model"))
    args.append(Argument('learning_rate', [1e-3], add_to_name_as="LR"))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs =1
    # number of runs to run in parallel
    n_cpu = 8
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))
    #create_runsforNodes(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))
