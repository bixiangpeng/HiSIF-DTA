FROM continuumio/anaconda3:latest
MAINTAINER bixiangpeng@stu.ouc.edu.cn

ARG env_name=HiSIF3

SHELL ["/bin/bash", "-c"]

WORKDIR /media/HiSIF-DTA

COPY . ./

RUN conda create -n $env_name python==3.7.7 \
&& source deactivate \
&& conda activate $env_name \
&& conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge \
&& pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
&& pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
&& pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
&& pip install torch_spline_conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html

RUN echo "source activate $env_name" > ~/.bashrc
ENV PATH /opt/conda/envs/$env_name/bin:$PATH
#&& pip install torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl \
#&& pip install torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl \
#&& pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl

CMD ["/bin/bash","inference.sh"]
