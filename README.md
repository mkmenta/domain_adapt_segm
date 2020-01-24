# Learning to adapt class-specific features across domains for semantic segmentation
Master thesis available [here](https://arxiv.org/abs/2001.08311).
## Requirements
Implemented using PyTorch _v0.4.0_ and Python 3.

Datasets:
- Get MNIST from [yann.lecun.com](http://yann.lecun.com/exdb/mnist/)
- Get MNIST-M from [yaroslav.ganin.net](http://bit.ly/2nrlUAJ)
- Get ThinMNIST from [my GitHub](https://github.com/mkmenta/ThinMNIST)
## Training commands
Apart from the following arguments, you will need to specify an `exp_name`, `mnist_path`, `mnist_m_path` and `mnist_thin_path`.

Specify a GPU to run the code adding `CUDA_VISIBLE_DEVICES=X` before the command.

Example of training command for the FCN Segmenter baseline:
```
python train_segm_baseline.py
```

Example of training command for the SGAN-S baseline:
```
python main.py --mode train
```

Example of training command for the SGAN-S + Uncond. baseline:
```
python main.py --mode train --da_type uncond --df_num_down 2 --lambda_fdom 1 --lambda_frf 1
```

Example of training command for the SGAN-S + In. Cond.:
```
python main.py --mode train --da_type input_cond --df_num_up 2 --df_num_down 4 --lambda_frf 1
```

**Example of training command for the SGAN-S + Out. Cond.:**
```
python main.py --mode train --da_type output_cond --df_num_up 2 --lambda_frf 1 --lambda_fdom 1
```
