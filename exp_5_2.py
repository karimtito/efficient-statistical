import subprocess
import sys, getopt

from dev.envir_def import EFFICIENT_STAT_PATH
check_eran, check_lp=True, True

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv,"",["check_eran=","check_lp="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"        
        sys.exit(2)
    for opt, arg in opts:
        if opt=="--check_eran":
            check_eran = True if arg.lower() in ("true", "yes","y",'t') else False
        elif opt=="--check_lp":
            check_lp = True if arg.lower() in ("true", "yes","y",'t') else False
if check_lp:
    command = f"python {EFFICIENT_STAT_PATH}dev/lp_acasxu.py --n_repeat=10 --N=2 --T=40 --p_c=1e-50 --properties=[1,2,3,4,5]"
    subprocess.run(command.split())
if check_eran:
    command = f"python {EFFICIENT_STAT_PATH}dev/check_eran_acasxu.py --properties=[1,2,3,4,5]"
    subprocess.run(command.split())