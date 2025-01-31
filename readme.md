<h1 align="center"> HOG-Diff: Higher-Order Guided Diffusion for Graph Generation </h1>




Official Implementations of HOG-Diff. 
In this work, we propose a novel Higher-order Guided
Diffusion (HOG-Diff) model that follows a coarse-to-fine generation curriculum and is guided by higher-order information, enabling the progressive generation of authentic graphs with inherent topological structures.

🚀 **We are continuously updating this repository and will release the complete code after the paper is accepted.**


<div align="center">
    <img src="figs/HOG-Diff-framework.png" alt="Framework">
</div>

---




## Environment Setup
This code was tested with PyTorch 2.0.0, cuda 11.8 and torch_geometrics 2.6.1

1️⃣ Download anaconda/miniconda if needed

2️⃣ Create and Activate a Python Virtual Environment
```
conda create -n HoGD_py39 python=3.9
conda activate HoGD_py39
```
3️⃣ Install Dependencies

Use the provided requirements.txt file to install all dependencies:
```
pip install -r requirements.txt
```

4️⃣ Compile the ORCA Program (for Graph Generation Evaluation)

For evaluating generic graph generation tasks, compile the [ORCA](http://www.biolab.si/supp/orca/orca.html) program by running the following command:

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```



# Running Experiments

To train and sample graphs using HOG-Diff, use the following commands:


```shell
CUDA_VISIBLE_DEVICES=0 python main.py --config config_name --mode train_ho
CUDA_VISIBLE_DEVICES=0 python main.py --config config_name --mode train_OU
CUDA_VISIBLE_DEVICES=0 python main.py --config config_name --mode sample
```

Replace config_name with the appropriate configuration file.

# Citation
Please cite our work if you find our code/paper is useful to your work. :
```latex

```


 
🍀 **Thank you for your interest in our work.** 🍀

If you have any questions or encounter any issues while using our code, please don't hesitate to raise an issue or reach out to us directly.

