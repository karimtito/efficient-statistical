import os
import pandas as pd
from envir_def import EFFICIENT_STAT_PATH
LOG_DIR = EFFICIENT_STAT_PATH + "logs/acasxu/"
print(LOG_DIR)
net_prop_results = []
for log_name in os.listdir(LOG_DIR+'eran_logs/'):
    print(log_name)
    if log_name.split('.')[-1]!='txt':
        continue
    clean_log_name = log_name.split('.')[0].strip('eran_')
    prop_number=clean_log_name[-1]
    net_name = clean_log_name.strip(f'_prop_{prop_number}')
    f = open(LOG_DIR+'eran_logs/'+log_name,'r')
    lines = f.readlines()
    time_compute = float(lines[-2].split()[0])
    L= lines[-3].split()
    if 'Verified' in L:
        certified = True
    elif 'Failed' in L:
        certified = False
    elif 'Infeasible' in lines[-4].split() :
        certified = 'INFEASIBLE'
    else:
        certified = 'TIMEOUT'
        time_compute = 600

    net_prop_result = {'network name': net_name, 'property':int(prop_number),'Certified':certified,"Compute time": time_compute, 'method': 'DeepPoly + Complete MILP'}
    net_prop_results.append(net_prop_result)
    
eran_acasxu_df = pd.DataFrame(net_prop_results)
with open(LOG_DIR+"ERAN_ACASXU_results.csv", "w+") as file:
    eran_acasxu_df.to_csv(file)