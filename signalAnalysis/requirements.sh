conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
conda install -c nvidia cuda-nvcc -y
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
pip install --upgrade pip
pip install -r requirements.txt
conda install -c conda-forge hdbscan -y

cp $CONDA_PREFIX/lib/libdevice.10.bc .
