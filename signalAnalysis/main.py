from teste_grande import *
from run import *
from report import *
from datetime import datetime

new_run_dir = f'./run_{datetime.now().strftime("%H_%M_%S")}'
os.makedirs(new_run_dir, exist_ok=True)
os.system(f'ln ./libdevice.10.bc {new_run_dir}/libdevice.10.bc')
os.chdir(new_run_dir)

teste_grande(dataset_path='/home/joaozenobio/PycharmProjects/SignalAnalysis/signalAnalysis/free_spoken_digit_dataset_master')
run(model_path='/home/joaozenobio/PycharmProjects/SignalAnalysis/signalAnalysis/run_19_33_16/model')
report()
