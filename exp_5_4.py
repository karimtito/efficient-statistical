import subprocess
from dev.envir_def import EFFICIENT_STAT_PATH
command = f"python {EFFICIENT_STAT_PATH}dev/lp_imagenet.py --n_repeat=10 --N=2 --T=20 --p_c=1e-15 --epsilon_range=[0.02,0.03,0.06]"
subprocess.run(command.split())