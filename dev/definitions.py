from dev.check_eran_acasxu_unnormalized import EFFICIENT_HOME
import os
from sys import platform
import subprocess
if 'EFFICIENT_STAT_HOME' in os.environ:
    EFFICIENT_STAT_PATH = os.environ['EFFICIENT_STAT_HOME']
elif platform=='linux' or 'linux2' or 'darwin':
    command  = "export EFFICIENT_STAT_HOME" 
    subprocess.run(command.split(''))
elif platform=="win32":
    command = "set EFFICIENT_STAT_HOME"
    subprocess.run(command.split(''))

DATA_DIR = EFFICIENT_STAT_PATH + 'data/'
NB_TEST_SAMPLES=100



