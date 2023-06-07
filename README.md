# VGG16-OpenCL


__This project is an implementation of the VGG16 model pre-trained on the [`CIFAR 10`](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for inference, written in `C` and `OpenCL C`.__

[`OpenCL`](https://www.khronos.org/opencl/) is a framework for writing programs that execute across heterogeneous platforms consisting of various processors and hardware accelerators. It provides programming languages and APIs to execute programs on these compute devices.

[`VGG16`](https://arxiv.org/abs/1409.1556) was proposed in the paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) by K. Simonyan and A. Zisserman and designed to work on 224 $\times$ 224 pixel input images. However, the model in this project has been adapted for `CIFAR 10` dataset, which includes 32 $\times$ 32 pixel images.

The model is a pre-trained and inference-only module, meaning you don't have to train it yourself.


## Prerequisites

You need the following installed on your machine:

   * An OpenCL compatible processor
   * Visual Studio (Optional)
   * OpenCL SDK

 > **PLEASE NOTE**: This project is written in Visual Studio on Windows. While it is possible to build this project in IDEs other than Visual Studio, you will need to create appropriate build configurations for your IDE and/or platform.


## Getting Started

   1. Clone this repository to your local machine:
      ```bash
      git clone https://github.com/sjleo1/VGG16-OpenCL.git
      cd VGG16-OpenCL
      ```
   2. Open the `.sln` file in Visual Studio
   3. Select `Build Solution` from the `Build` menu


## Program Outline

This program runs the classification module of VGG16, separately, in serial and in parallel. You will get an elapsed time for each classification operation after each result is compared to the correct answer.

### OpenCL Workflow

The following is a brief description of how an OpenCL program typically flows.

![OpenCL Workflow](./executing_programs.jpg)

   1. Selecting ***device***s and defining a ***context***: Context is the environment within which the *kernel*s are defined and execute.
   2. Creating ***command-queue***s: The *host* and the *OpenCL device*s interact with each other through *command*s posted by the host.
   3. Building ***program object***s: Program object is compiled and linked to generate kernels for OpenCL devices.
   4. Creating ***memory object***s: The *host program* defines memory objects required and pass them onto the arguments of kernels.
   5. Enqueueing ***command***s: Commands are enqueued to the command-queues to execute the kernels.

Please refer to [`OpenCL Guide`](https://github.com/KhronosGroup/OpenCL-Guide) for more detailed information.

### VGG16 Architecture

The VGG16 model gained its reputation from the use of small 3 $\times$ 3 convolutional filters throughout the network, which improved performance and reduced the computational complexity. The model in this project was adapted for images of 32 $\times$ 32 pixels, contrary to the original model that had been designed for that of the size of 224 $\times$ 224 pixels.


## Used OpenCL Optimization Techniques

   > TODO


## Results

### Sequential Code

| Processor | Memory | Elapsed Time (500 images) | Elapsed Time per Image (500 images) |
|:-:|:-:|:-:|:-:|
| `Intel i5-1240P` | 16 GB LPDDR5 | 467.427 seconds | 0.9349 seconds |
| `Intel i5-10400` | 32 GB DDR4 | 613.509 seconds | 1.2270 seconds |
| `Intel m3-6Y30` | 4 GB LPDDR3 | 1345.945 seconds | 2.6919 seconds |

### Parallel Code

| Processor | Dedicated Memory | Elapsed Time | Elapsed Time per Image |
|:-:|:-:|:-:|:-:|
| `Nvidia RTX 3060` | 12 GB GDDR6 | ? seconds | ? seconds |
| `Intel Iris Xe Graphics (80EU)` | No | ? seconds | ? seconds |
| `Intel UHD Graphics 630` | No | ? seconds | ? seconds |
| `Intel HD Graphics 515` | No | ? seconds | ? seconds |


## License

   > TODO