import numpy as np
from time import time
import tensorflow as tf
import pandas as pd
from math import ceil
import os
import sys, getopt
from utils import nn_score, graph_model_converter, input_transformer_eran, batch_normal_kernel
from eran_net_reader import read_tf_net 
from sampling_tools import ImportanceSplittingLpBatch
from envir_def import EFFICIENT_STAT_PATH
LOG_DIR = EFFICIENT_STAT_PATH+"logs/mnist/"
DATA_DIR = EFFICIENT_STAT_PATH+"data/MNIST/"
NETS_DIR = DATA_DIR+"test_nets/"
DATA_PATH = DATA_DIR+"mnist.npz"

DIM = 784
gaussian_gen = lambda N: np.random.normal(size =(N,DIM))
print(f"Number GPU used: {len(tf.config.list_physical_devices('GPU'))}")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path = DATA_PATH)
flatten_mnist_tf = lambda x: x.astype(np.float64).reshape((x.shape[0],np.prod(x.shape[1:])))/np.float64(255)
X_test_f = flatten_mnist_tf(X_test)

n_rep = 10
N=2
p_c=10**-40
T=50
alpha=1-10**-4
epsilon_range = [0.1,0.2,0.3]
if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv,"n:N:T:p:a:",["n_repeat=","p_c=","T=","N=","alpha=","epsilon_range=","epsilon="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        
        sys.exit(2)
    for opt, arg in opts:
        print(f"opt:{opt}, arg:{arg}")
        if opt in ('-n', "--n_repeat"):
          n_repeat= int(arg)
        elif opt in ("-N", "--N"):
            N = int(arg)
        elif opt in ("-a", "--alpha"):
            alpha= arg
        elif opt in ("-T", "--T"):
            T= int(arg)
        elif opt in ("-p", "--p_c"):
            p_c= float(arg)
        elif opt=="--epsilon_range":
            epsilon_range = [float(e) for e in  arg.strip('[').strip(']').split(',')]
        elif opt=="--epsilon":
            epsilon_range = [float(arg)]


name_method = f"Last Particle|N={N}|p_c={p_c}|T={T}"
print(name_method)
print(epsilon_range)

net_eps_img_results = []
count = 0

TOTAL_RUN = n_rep*len(epsilon_range)*6
avg_=0
i=0
for net_name in os.listdir(NETS_DIR):
    i+=1
    net_file = NETS_DIR+net_name
    print(f'network file:{net_file}')
    sep = net_name.split('.')
    model_name,extension="_".join(sep[:-1]), sep[-1]
    pyt = extension=='pyt'
    print(f'model name:{model_name}')
    with tf.compat.v1.variable_scope(model_name):    
        model, is_conv,mean, std,=read_tf_net(net_file, shape=(None,DIM),is_trained_with_pytorch=pyt,batch_mode=True)
        
    if mean!=0:
        X_test_n = (X_test_f-mean)
    else:
        X_test_n = X_test_f
    if std!=0:
        X_test_n = X_test_f/std
    else:
        X_test_n = X_test_f
    model_infer = graph_model_converter(model, variable_scope=model_name)
    logits_test = model_infer(X_test_n,by_batches = True, batch_size=200, verbose=1)
    y_pred_test = np.argmax(logits_test, 1)
    test_correct = y_pred_test==y_test
    true_indices = np.where(y_pred_test==y_test)[0]
    nb_examples = 100
    nb_systems = nb_examples
    big_batch_size= 256
    x_test, Y_test = X_test_n[true_indices][:nb_examples], y_test[true_indices][:nb_examples]
    for j in range(len(epsilon_range)):
        epsilon = epsilon_range[j]
        input_transform= lambda X,idx: input_transformer_eran(X,X_original = x_test[idx], mean=mean, std=std,eps=epsilon)
        total_score = lambda X,idx: nn_score(input_transform(X,idx), Y_test[idx], model_infer, mean=mean, std=std, several_labels=True)
        input_transform_big = lambda X: input_transformer_eran(X,X_original = np.repeat(x_test,N,axis=0), mean=mean, std=std, eps=epsilon)

        total_score_big = lambda X: nn_score(input_transform_big(X), np.repeat(Y_test,N),model=model_infer, mean=mean, std=std, several_labels=True, by_batches=True, batch_s = big_batch_size)
        aggr_results = []
        for k in range(n_rep):
            count+=1
            print(f'Run {k} on network {model_name} for epsilon={epsilon} started... \n (RUN {count}/{TOTAL_RUN}, Avg. time per run:{avg_}) ')
            t0= time()
            p_est, s_out = ImportanceSplittingLpBatch(gen =gaussian_gen, nb_system=nb_systems,N=N,  s=1.5, kernel_b = batch_normal_kernel,h_big = total_score_big, h=total_score,N=N, tau=0, p_c=p_c, T=T,
    alpha_test = alpha , p_c = p_c, verbose = 0,check_every=5, accept_ratio=0.25,  reject_thresh = 0.99, reject_forget_rate =0.9, gain_forget_rate=0.9, fast_d=3)
            t1=time()-t0
            local_result = [t1, s_out['Cert'], s_out['Calls'],p_est]
            aggr_results.append(local_result)
            avg_ = (count-1)/count*avg_ + (1/count)*t1
            print(f'Run {k} on network {model_name} for epsilon={epsilon} finished... \n (RUN {count}/{TOTAL_RUN}, Avg. time per run:{avg_}) ')

        aggr_results = np.array(aggr_results)
        p_estimates = np.vstack(aggr_results[:,-1])
        cert_ = np.vstack(aggr_results[:,1])
        avg_compute_time, avg_calls = aggr_results[:,0].mean(), int(aggr_results[:,2].mean())
        std_compute_time,  std_calls = aggr_results[:,0].std(), aggr_results[:,2].std()
        avg_pest = p_estimates.mean(axis=0)
        std_pest = p_estimates.std(axis=0)
        avg_cert = cert_.mean(axis =0) 
        CI = s_out['CI_est']
        for l in range(nb_examples):
            net_eps_img_result = {'network name': model_name, 'epsilon':epsilon, 'Image index (MNIST Test)':true_indices[l], 'Original label':Y_test[l] , 'Compute time':t1/nb_examples, 'Certified': bool(int(s_out['Cert'][l])), 'method': name_method, 'Calls': int(s_out['Calls']/nb_examples), 'P_est': p_est[l], 'CI': s_out['CI_est'][l], 'Std P_est':std_pest[l], 'Computie time std':std_compute_time ,'Calls std': std_calls, 'Avg. Compute time':avg_compute_time/nb_examples,'Avg. number of calls':ceil(avg_calls/nb_examples), 'Frequency Certified': avg_cert[l],'Avg. P_est':avg_pest[l]}
            net_eps_img_results.append(net_eps_img_result)

lp_mnist_results = pd.DataFrame(net_eps_img_results)
save_file = LOG_DIR+f"LP_N_{N}_pc_{-int(np.log10(p_c))}_T_{T}_n_{n_repeat}_MNIST_results.csv"
method_clean_name = f"LP_N_{N}_pc_{-int(np.log10(p_c))}_T_{T}"
with open(save_file, "w+") as file:
     lp_mnist_results.to_csv(file)
         
 