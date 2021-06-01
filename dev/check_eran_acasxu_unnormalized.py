import subprocess
import os 
from pathlib import Path

ERAN_PATH = "/home/karim-tito/eran/"
EFFICIENT_HOME = "/home/karim-tito/efficient-statistical-assessment/"
LOG_DIR = EFFICIENT_HOME+"logs/acasxu/"
dir_path = EFFICIENT_HOME+"data/acasxu/nets/"
os.chdir(ERAN_PATH+"tf_verify/")


for net_name in os.listdir(dir_path):
    net_file = dir_path+net_name
    clean_name = net_name.split('.')[0]
    
    for j in range(1,6):
        log_name = "eran_"+clean_name+f"_prop_{j}.txt"
        log_path = Path(LOG_DIR+log_name)
        if log_path.exists():
            continue
        command = f"python . --netname {net_file} --dataset acasxu --domain deeppoly --complete True --timeout_complete 100 --specnumber {j} --normalized_region False"
        with open(LOG_DIR+log_name, "w+") as output:
            subprocess.run(command.split(" "),stdout=output)
