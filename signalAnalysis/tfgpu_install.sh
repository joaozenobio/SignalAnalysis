conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install -r requirements.txt
pip install joblib==1.1.0
