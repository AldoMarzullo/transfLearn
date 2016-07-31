#source env/bin/activate
pip install -U scikit-learn
pip install Pillow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64"
