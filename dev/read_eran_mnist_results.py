import os
import pandas as pd
from envir_def import EFFICIENT_STAT_PATH
LOG_DIR = EFFICIENT_STAT_PATH + "logs/MNIST/"
net_eps_img_results = []
for log_name in os.listdir(LOG_DIR+'eran_logs/'):
    if log_name.split('.')[-1]!='txt':
        continue
    clean_log_name = log_name.split('.')[0].strip('eran_logseran_')
    print(clean_log_name)
    eps_number=clean_log_name
    net_name = "_".join(clean_log_name.split('_')[:-3])
    f = open(LOG_DIR+'eran_logs/'+log_name,'r')
    lines = f.readlines()
    epsilon = float((list(filter(lambda k: 'epsilon' in k, lines))[0]).strip(',\n').split(' ')[-1])
    lines_img = list(filter(lambda k: 'img' in k, lines))
    lines_img = [ x for x in lines_img if 'incorrectly' not in x]
    time_compute = float((list(filter(lambda k: 'time: ' in k, lines))[-1]).strip('\n').split(' ')[-1])
    for img in lines_img:
        b = img.strip('\n').split(' ')
        index, status = b[1], b[2]
        certify = b[2]=='Verified'
        
   
        net_eps_img_result = {'network name': net_name, 'eps': epsilon,'Certified':certify,"Compute time": time_compute/100, 'method': 'DeepPoly + Complete MILP', 'Image index': int(index)}
        print(net_eps_img_result)
        net_eps_img_results.append(net_eps_img_result)
    
eran_mnist_df = pd.DataFrame(net_eps_img_results)
with open(LOG_DIR+"ERAN_MNIST_results.csv", "w+") as file:
    eran_mnist_df.to_csv(file)
