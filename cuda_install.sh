apt install aria2 -y
aria2c -x 15 -s 15 "https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux"
chmod +x cuda_10.0.130_410.48_linux.run
./cuda_10.0.130_410.48_linux.run
echo -e 'export PATH="/usr/local/cuda-10.0/bin:$PATH"\nexport LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"' >>~/.bashrc
source ~/.bashrc
echo -e 'export PATH=$PATH:/usr/local/cuda-10.0/bin\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64\nexport LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64' >>/etc/profile
source /etc/profile
