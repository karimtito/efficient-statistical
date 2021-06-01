import os
from pathlib import Path
local_path = Path(os.path.dirname(os.path.realpath(__file__)))
EFFICIENT_STAT_PATH = str(local_path.parent.absolute())+'/'
#from sys import platform
#import subprocess
# if 'EFFICIENT_STAT_HOME' in os.environ:
#     EFFICIENT_STAT_PATH = os.environ['EFFICIENT_STAT_HOME']
# elif platform=='linux' or 'linux2' or 'darwin':
#     command  = "export EFFICIENT_STAT_HOME" 
#     subprocess.run(command.split(''))
# elif platform=="win32":
#     command = "set EFFICIENT_STAT_HOME"
#     subprocess.run(command.split(''))
print(EFFICIENT_STAT_PATH)
DATA_DIR = EFFICIENT_STAT_PATH + 'data/'
NB_TEST_SAMPLES=100
from time import sleep
sleep(1)


