from teste2 import *
from run import *
from report import *
from datetime import datetime

new_run_dir = f'./run_{datetime.now().strftime("%H_%M_%S")}'
os.makedirs(new_run_dir, exist_ok=True)
os.system(f'ln ./libdevice.10.bc {new_run_dir}/libdevice.10.bc')
os.chdir(new_run_dir)

teste2()
run()
report()
