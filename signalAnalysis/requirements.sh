conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
pip install -r requirements.txt

cp $CONDA_PREFIX/lib/libdevice.10.bc .
