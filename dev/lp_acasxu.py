import numpy as np
import pandas as pd
import time

import os
import sys, getopt

from utils import normal_kernel
from acasxu_utils import onnx_to_model, read_acasxu_input_file, input_transformer_acasxu
from acasxu_utils import acasxu_prop1_score, acasxu_prop2_score, acasxu_prop3_score, acasxu_prop4_score, acasxu_prop5_score
from sampling_tools import ImportanceSplittingLp
import envir_def
DIM = 5

LOG_DIR = envir_def.EFFICIENT_STAT_PATH+"logs/acasxu/"
nets_path = envir_def.EFFICIENT_STAT_PATH+"data/acasxu/nets/"
acasxu_scores = [acasxu_prop1_score, acasxu_prop2_score, acasxu_prop3_score, acasxu_prop4_score, acasxu_prop5_score]
gaussian_gen = lambda N: np.random.normal(size =(N,DIM))
round_str = lambda x: '{:.3e}'.format(x)

n_repeat = 10
N= 2
p_c=10**-50
T=40
alpha=1-1e-3
index_properties = np.arange(1,6)
verb=0

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv,"n:N:T:p:a:",["n_repeat=","p_c=","T=","N=","alpha=","properties="])

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
        elif opt=="--properties":
            index_properties=[int(e) for e in  arg.strip('[').strip(']').split(',')]
        elif opt in ("--verbose", "-v"):
            verb = float(arg)



name_method = f"Last Particle|N={N}|p_c={p_c}|T={T}"
print(f"Tested method: {name_method}")


net_prop_results = [] 

count = 0
TOTAL_RUN = n_repeat*45*5
avg_=0
for net_name in os.listdir(nets_path):
    net_file = nets_path+net_name
    onnx_model = onnx_to_model(net_file)
    clean_name = net_name.split('.')[0]
    for j in index_properties:
        local_score = acasxu_scores[j-1]
        input_file = envir_def.EFFICIENT_STAT_PATH+f'data/acasxu/specs/acasxu_prop_{j}_input_prenormalized.txt'
        input_con= read_acasxu_input_file(input_file, verbose = 1)
        total_score_function = lambda X: local_score(onnx_model(input_transformer_acasxu(X,input_con)))
        aggr_results = []
        for i in range(n_repeat):
            count+=1
            
            t0 = time.time()
            p_est, s_out = ImportanceSplittingLp(gen =gaussian_gen, kernel = normal_kernel, h=total_score_function,N=N,     tau=0, p_c=p_c, alpha_test = alpha, verbose = 0,s=1.8,gain_forget_rate=0.8, reject_forget_rate=0.8,T=T)
            t1= time.time()-t0 
            local_result = [t1, s_out['Cert'], s_out['Calls'],p_est]
            aggr_results.append(local_result)
            avg_ = (count-1)/count*avg_ + (1/count)*t1
            print(f'Run {i+1} on network {clean_name} for property {j} is over (RUN {count}/{TOTAL_RUN}, Avg. time per run:{avg_}) ')
            eta_ = ((TOTAL_RUN-count)*avg_)/3600
            hours = int(eta_)
            minutes = (eta_*60) % 60
            seconds = (eta_*3600) % 60

            print("ETA %d:%02d.%02d" % (hours, minutes, seconds))
            

        aggr_results = np.array(aggr_results)
        p_estimates = aggr_results[:,-1]
        avg_compute_time, avg_calls = aggr_results[:,0].mean(), int(aggr_results[:,2].mean())
        avg_pest = p_estimates.mean()
        avg_cert = aggr_results[:,1].mean() 
        p_est_str = round_str(p_est)
        CI = s_out['CI_est']
        coverage_CI = ((p_estimates<=CI[1])*(p_estimates>=CI[0])).mean()
        
        net_prop_result = {'network name': clean_name, 'property':j, 'Compute time':t1, 'Certified': s_out['Cert'], 'method': name_method, 'Calls': s_out['Calls'], 'P_est': p_est_str, 'CI': s_out['CI_est'],
'Avg. Compute time':avg_compute_time,'Avg. number of calls':avg_calls, 'Frequency Certified': avg_cert,
    'Avg. P_est':round_str(avg_pest), 'Coverage CI': coverage_CI}
        net_prop_results.append(net_prop_result)

lp_acasxu_results= pd.DataFrame(net_prop_results)

save_file = LOG_DIR+f"LP_N_{N}_pc_{-int(np.log10(p_c))}_T_{T}_n_{n_repeat}_ACASXU_results.csv"
method_clean_name = f"LP_N_{N}_pc_{-int(np.log10(p_c))}_T_{T}_ACASXU"
with open(save_file, "w+") as file:
     lp_acasxu_results.to_csv(file)

