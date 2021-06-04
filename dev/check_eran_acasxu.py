""" This scripts requires that the benchmark ERAN is installed 
    It checks all combination of the 45 ACAS Xu table compression neural networks and the 5 first properties. 
    The DNN files are in ONNX format and were downloaded from the VNNLIB website. 
    They do require input/output normalization. 
 """

import subprocess
import os 
from pathlib import Path
import sys, getopt
import envir_def



ERAN_PATH = None
LOG_DIR = envir_def.EFFICIENT_STAT_PATH+"logs/acasxu/eran_logs/"
NETS_DIR = envir_def.EFFICIENT_STAT_PATH+"data/acasxu/nets/"

index_properties = range(1,6)

timeout_complete = 600
if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv,"t:p:",["timeout_complete=","properties="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"        
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--properties"):
            index_properties=[int(e) for e in  arg.strip('[').strip(']').split(',')]
        elif opt in ("-t", "--timeout_complete"):
            timeout_complete=float(arg)


user_home_path = os.path.expanduser("~")
print(f"User home:{user_home_path}")
if 'ERAN_PATH' in os.environ:
    ERAN_PATH = os.environ['ERAN_PATH']
elif 'ERAN_HOME' in os.environ:
    ERAN_PATH = os.environ['ERAN_HOME']
else:
    ans = input("ERAN environnment variable (as ERAN_PATH or ERAN_HOME) was not found. Do you allow for automatical search of ERAN repo ? (y/n)")
    if ans.lower() in ("yes", "y", "o","oui","ok" ):
        search_res = sorted(Path(user_home_path).glob("**/eran"))
        if len(search_res)==0:
            raise RuntimeError("ERAN benchmark could not be found. \n Please install ERAN and define environnment variable as ERAN_PATH or ERAN_HOME.")
        elif len(search_res)==1:
            ERAN_PATH = str(search_res[0])+"/"
        else:
            i=input(f"Several path to eran clones were foud: \n {[str(path) for path in search_res]}. Choose one by submitting an index (between 1 and {len(search_res)}). \n i=")
            ERAN_PATH = str(search_res[int(i)-1])+"/"
    elif ans.lower() in ("no", "n", "non"):
        raise RuntimeError("ERAN benchmark could not be found. \n Please install ERAN and define an environnment variable as ERAN_PATH or ERAN_HOME.")

print(f"ERAN path: {ERAN_PATH}")


os.chdir(ERAN_PATH+"tf_verify/")


for net_name in os.listdir(NETS_DIR):
    net_file = NETS_DIR+net_name
    clean_name = net_name.split('.')[0]
    
    for j in index_properties:
        log_name = "eran_"+clean_name+f"_prop_{j}.txt"
        log_path = Path(LOG_DIR+log_name)
        if log_path.exists():
            continue
        command = f"python . --netname {net_file} --dataset acasxu --domain deeppoly --complete True --timeout_complete {timeout_complete} --specnumber {j} --normalized_region False"
        with open(LOG_DIR+log_name, "w+") as output:
            subprocess.run(command.split(" "),stdout=output)
