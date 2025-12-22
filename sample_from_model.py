import os
import argparse
from DatasetCreator.loadGraphDatasets.HCPDatasetGenerator import plot
from train import TrainMeanField
import numpy as np
import igraph as ig
from EnergyFunctions.HCPEnergy import HCPEnergyClass
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Switch ray into local mode for debugging')
parser.add_argument('--multi_gpu', action='store_true', help='wheter to use multi gpu or not, KEEP IT ALWAYS TRUE')
parser.add_argument('--mode', default='Diffusion', choices = ["Diffusion"], help='Define the Approach')
parser.add_argument('--EnergyFunction', default='MIS', choices = ["MaxCut", "MIS", "MVC", "MaxCl", "WMIS", "MDS", "MaxClv2", "TSP", "IsingModel", "SpinGlass", "SpinGlass", "HCP"], help='Define the EnergyFunction of the IsingModel')
parser.add_argument('--IsingMode', default='RB_iid_100', choices = ["Gset","BA_large","RB_iid_small", "RB_iid_dummy", "BA_dummy", "RB_iid_large" ,"RRG_200_k_=all", "BA_small","TSP_random_100", 
                                                                    "TSP_random_20", "COLLAB", "IMDB-BINARY", "RB_iid_100_dummy" , "RB_iid_200", "RB_iid_100", "NxNLattice_4x4", "NxNLattice_8x8", "NxNLattice_16x16", "NxNLattice_10x10", "SpinGlassUniform_10x10", "SpinGlass_16x16", "NxNLattice_24x24", "NxNLattice_32x32", "HCP_dummy"], help='Define the Training dataset')
parser.add_argument('--graph_mode', default='normal', choices = ["normal", "TSPModel", "Transformer", "UNet"], help='Use U-Net or normal GNN, TSP model is a graph based implementation of the transformer, transformer is to be prefered')
parser.add_argument('--train_mode', default='REINFORCE', choices = ["REINFORCE", "PPO", "Forward_KL"], help='Use U-Net or normal GNN')
parser.add_argument('--AnnealSchedule', default='linear', choices = ["linear", "cosine", "exp"], help='Define the Annealing Schedule')
parser.add_argument('--temps', default=[0.], type = float, help='Define gridsearch over Temperature', nargs = "+")
parser.add_argument('--T_target', default=0., type = float, help='Define target temperature')
parser.add_argument('--N_warmup', default=0, type = int, help='Define gridsearch over Number of Annealing steps')
parser.add_argument('--N_anneal', default=[2000], type = int, help='Define gridsearch over Number of Annealing steps', nargs = "+")
parser.add_argument('--N_equil', default = 0, type = int, help='Define gridsearch over Number of Equil steps')
parser.add_argument('--lrs', default=[5e-5], type = float, help='Define gridsearch over learning rate', nargs = "+")
parser.add_argument('--lr_schedule', default="cosine", choices = ["cosine", "cosine_warm_restarts", "None"], help='use learning rate schedule or not')
parser.add_argument('--seed', default=[123], type = int, help='Define dataset seed', nargs = "+")
parser.add_argument('--GPUs', default=["0"], type = str, help='Define Nb', nargs = "+")
parser.add_argument('--n_hidden_neurons', default=[64], type = int, help='number of hidden neurons', nargs = "+")
parser.add_argument('--n_rand_nodes', default=2, type = int, help='define node embedding size')
parser.add_argument('--stop_epochs', default=10000, type = int, help='define early stopping')
parser.add_argument('--n_diffusion_steps', default=[9], type = int, help='define number of diffusion steps', nargs = "+")
parser.add_argument('--time_encoding', default="one_hot", type = str, help='encoding of diffusion steps')
parser.add_argument('--noise_potential', default = ["annealed_obj"], type = str, choices = ["bernoulli", "boltzmann_noise", "diffusion", "annealed_obj", "categorical", "combined"], help='define the diffusion mode', nargs = "+")
parser.add_argument('--n_basis_states', default=[10], type = int, help='number of states per graph', nargs = "+")
parser.add_argument('--n_test_basis_states', default=8, type = int, help='number of states per graph during test time')
parser.add_argument('--batch_size', default=[30], type = int, help='number of graphs within a batch', nargs = "+")
parser.add_argument('--minib_diff_steps', default=1, type = int, help='minibatch size in diffusion steps in PPO or forward KL')
parser.add_argument('--minib_basis_states', default=10, type = int, help='minibatch size in basis states in PPO or forward KL')
parser.add_argument('--inner_loop_steps', default=1, type = int, help='number of inner loop steps in PPO or forward KL')
parser.add_argument('--n_GNN_layers', default=[8], type = int, help='num of GNN Layers', nargs = "+")
parser.add_argument('--project_name', default= "", type = str, help='define project name')
parser.add_argument('--beta_factor', default=[1.], type = float, help='desfine noise strength', nargs = "+")
parser.add_argument('--loss_alpha', default=0.0, type = float, help='rel weighteing between forward and reverse KL')
parser.add_argument('--MCMC_steps', default=0, type = int, help='number of MCMC steps')
parser.add_argument('--mov_average', default=0.0009, type = float, help='moving_average for RL')
parser.add_argument('--TD_k', default=3, type = float, help='TD_k for PPO')
parser.add_argument('--clip_value', default=0.2, type = float, help='clip_value for PPO')
parser.add_argument('--value_weighting', default=0.65, type = float, help='value_func weighting for PPO')
parser.add_argument('--mem_frac', default= ".90", type = str, help='memory fraction')
parser.add_argument('--diff_schedule', default= "own", type = str, help='define diffusion schedule')
parser.add_argument('--proj_method', default= "None", choices = ["CE", "feasible", "None"], type = str, help='define projection method')
parser.add_argument('--linear_message_passing', action='store_true')
parser.add_argument('--no-linear_message_passing', dest='linear_message_passing', action='store_false')
parser.add_argument('--relaxed', action='store_true')
parser.add_argument('--no-relaxed', dest='relaxed', action='store_false')
parser.add_argument('--time_conditioning', action='store_true')
parser.add_argument('--no-time_conditioning', dest='time_conditioning', action='store_false')
parser.add_argument('--deallocate', action='store_true')
parser.add_argument('--no-deallocate', dest='time_conditioning', action='store_false')
parser.add_argument('--jit', action='store_true')
parser.add_argument('--no-jit', dest='jit', action='store_false')
parser.add_argument('--mean_aggr', action='store_true')
parser.add_argument('--no-mean_aggr', dest='mean_aggr', action='store_false')
parser.add_argument('--grad_clip', action='store_true')
parser.add_argument('--no-grad_clip', dest='grad_clip', action='store_false')
parser.add_argument('--graph_norm', action='store_true')
parser.add_argument('--no-graph_norm', dest='graph_norm', action='store_false')
parser.add_argument('--sampling-temp', default=0., type = float, help='define sampling temperature for asymptoticly unbiased estimations')
parser.add_argument('--n_sampling_rounds', default=5, type = int, help='how often the the basis states are sampled in a loop in unbiased estimations')
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--no-bfloat16', dest='bfloat16', action='store_false')

parser.set_defaults(bfloat16=False)
parser.set_defaults(CE=False)
parser.set_defaults(graph_norm=True)
parser.set_defaults(grad_clip=True)
parser.set_defaults(mean_aggr=True)
parser.set_defaults(relaxed=True)
parser.set_defaults(time_conditioning=True)
parser.set_defaults(deallocate=False)
parser.set_defaults(jit=False)
parser.set_defaults(linear_message_passing=True)
parser.set_defaults(multi_gpu=True)
parser.set_defaults(GPUs=["2"])
parser.set_defaults(train_mode="PPO")
args = parser.parse_args()

def meanfield_run():

    resources_per_trial = 1.
    devices = args.GPUs
    n_workers = int(len(devices)/resources_per_trial)

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if(len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str, type(device_str))
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    print("Init ray in local_mode!")

    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True,)# precision=4


    run(flexible_config = {"use_sample": 2, "jit": False, "dataset_name": "HCP_dummy", "problem_name": "HCP", "edge_updates": True, "mode_node_edge": "edge", "N_anneal": args.N_anneal[0], "load_wandb_id": "m68s1bnd","n_random_node_features": 16, "n_diffusion_steps": args.n_diffusion_steps[0], "minib_diff_steps": args.minib_diff_steps, "minib_basis_states": args.minib_basis_states, "N_basis_states": args.n_basis_states[0], "train_mode": args.train_mode}, overwrite = True) # "load_wandb_id": "oz5t74ww"






def run( flexible_config, overwrite = True):

    config = {
        "mode": "Diffusion",  # either Diffusion or MeanField
        "dataset_name": "HCP_dummy",
        "problem_name": "HCP",
        "jit": True,
        "wandb": True,

        "seed": 123,
        "lr": 1e-3,
        "batch_size": 30, # H
        "N_basis_states": 100, # n_s

        "random_node_features": True,
        "n_random_node_features": 5,
        "relaxed": False,

        "T_max": 0.01,
        "T_target": 0.,
        "N_warmup": 0,
        "N_anneal": 2000,
        "N_equil": 0,
        "stop_epochs": 2000,

        ### TODO rework network and remove edge updates
        "n_hidden_neurons": 64,
        "n_features_list_prob": [64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [10],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [30],
        "n_features_list_decode": [64],
        "n_message_passes": 5,
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
        "edge_updates": False,
        "n_diffusion_steps": 1,
        "beta_factor": 1,
        "noise_potential": "annealed_obj",

        "time_conditioning": True,

        "project_name": args.project_name,
        "mean_aggr": False,
        "grad_clip": True,
        "messeage_concat": False,
        "graph_mode": "normal",
        "loss_alpha": 0.0,
        "MCMC_steps": 0,
        "train_mode": "PPO",
        "inner_loop_steps": 2,
        "minib_diff_steps": 3,
        "minib_basis_states": 10,
        "graph_norm": False,
        "proj_method": "None",
        "diff_schedule": "DiffUCO",
        "mov_average": 0.05,
        "sampling_temp": 1.,
        "n_sampling_rounds": 5,
        "n_test_basis_states": 400,
        "bfloat16": False,
        "AnnealSchedule": "linear",
        "time_encoding": "one_hot",
        "lr_schedule": "cosine",
        "TD_k": 3,
        "clip_value": 0.2,
        "value_weighting": 0.65,

        "mode_node_edge": "node",
        "load_wandb_id": "m68s1bnd",
        "use_sample": -1,
        "node_emb_type": "sin_positional",
        "augment_rooms": False
    }

    if(overwrite):
        for key in flexible_config:
            if(key in config.keys()):
                config[key] = flexible_config[key]
            else:
                raise ValueError("key does not exist")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_frac)
    if(args.deallocate):
        pass
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    config["n_bernoulli_features"] = [10, 20, 30, 100][config["use_sample"]]
    config["T_max"] = 0.01 * 10. / config["n_bernoulli_features"]
    config["T_target"] = config["T_max"] * 0.66

    config["embedding_dim"] = [32, 48, 64, 128][config["use_sample"]]

    # from jax import config
    # config.update("jax_enable_x64", True)

    train = TrainMeanField(config, config.get("load_wandb_id", None), load_best_parameters="load_wandb_id" in config)

    log_dict = train.sample(100)

    #plot_time_progression(log_dict)
    plot_samples(log_dict)
    

def plot_time_progression(log_dict, select_sample=0):
    time_progression = log_dict["bin_sequence"][0, :, :, :, 0]

    T, D, N = time_progression.shape

    graph_batch = log_dict["graph_batch"]

    node_gr_idx = jnp.repeat(jnp.arange(graph_batch["graphs"][0].graph.n_node.shape[1]), graph_batch["graphs"][0].graph.n_node[0], axis=0, total_repeat_length=graph_batch["graphs"][0].graph.n_node.sum())

    for t in range(T):
        sample = time_progression[t, :, select_sample].astype(int)

        graph_batch = log_dict["graph_batch"]

        _, e_dict, _ = HCPEnergyClass.calculate_Energy(None, graph_batch["graphs"][0], sample, node_gr_idx)

        cleaned = {k: v[0].item() for k, v in e_dict.items()}
        total = sum(cleaned.values())
        print(f"sample {T-t-1}: {total}({cleaned})")
        plot(None, graph_batch["graphs"][0].graph.globals["node_types"].squeeze(),f"energy_time_progression_{t}.png", solution_nodes=log_dict["X_0"][0, :, 0, 0], meta_graph=graph_batch["graphs"][0])


def plot_samples(log_dict):
    samples = log_dict["X_0"]

    for i in range(samples.shape[2]):
        sample = samples[0, :, i, 0]
        sample = jnp.ones_like(sample)

        graph_batch = log_dict["graph_batch"]

        plot(None, graph_batch["graphs"][0].graph.globals["node_types"].squeeze(),f"plots/plot_full_{i}.png", solution_nodes=log_dict["X_0"][0, :, i, 0], meta_graph=graph_batch["graphs"][0])

if __name__ == "__main__":
    meanfield_run()
