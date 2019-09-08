Pytorch optimized on sparse video
--------------------------------------------------------------------------------

During inference on sparse videos/images, where most pixels of the target have the value zero, many resulting values after the Conv2d() layer will be zeros no matter what weights does a Conv2d filter hold. In the original pytorch package, such redandunt operations are not being taken care of which leads to a longer inference time. This repository made some changes to the existed pytorch package for better performance in case of row-wise sparse images/videos without adding any possible error and not much overhead by providing a sparsity check towards the target. The inference time is cut off by 50% when the code is used in the example provided below. 

## Installation

### From Source

If you are installing from source, we highly recommend installing an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment.
You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro.

Once you have [Anaconda](https://www.anaconda.com/distribution/#download-section) installed, here are the instructions.

#### Install Dependencies

On Linux
```bash
# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda90 # or [magma-cuda92 | magma-cuda100 ] depending on your cuda version
```

#### Get the PyTorch Source
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

#### Install PyTorch
On Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

#### Replace file
Once you have your pytorch built, you should replace the file 'jit_avx512_common_convolution.cpp' to /pytorch/third_party/ideep/mkl-dnn/src/cpu/ with the file provided in this repository and do:

On Linux
```bash
python setup.py rebuild
```
To better visualize the performance, you can turn on the verbose model by:
```bash
export MKLDNN_VERBOSE= value
```
where vaule has the meaning that:
0	-> no verbose output (default)
1	-> primitive information at execution
2	-> primitive information at creation and execution

## Example
Reaching the dataset from https://drive.google.com/uc?export=download&confirm=E9YD&id=1TF6CsbFwhMo2Ry3LHQkgJPzJdN0ZS1XC
And run Sparse_Degree_Analysis_batch.py in the example folder
Comparison of the performance is available in the example folder.
