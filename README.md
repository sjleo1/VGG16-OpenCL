# VGG16-OpenCL


__`VGG16-OpenCL` is an implementation of the VGG16 model pre-trained on the [`CIFAR 10`](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for inference, written in `C` and `OpenCL C`.__

[`OpenCL`](https://www.khronos.org/opencl/) is a framework for writing programs that execute across heterogeneous platforms consisting of various processors and hardware accelerators. It provides programming languages and APIs to execute programs on these compute devices.

[`VGG16`](https://arxiv.org/abs/1409.1556) was proposed in the paper [*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) by K. Simonyan and A. Zisserman and designed to work on 224 $\times$ 224 pixel input images. However, the model in this project has been adapted for `CIFAR 10` dataset, which includes 32 $\times$ 32 pixel images.

__The goal of this project is to optimize the model to *`reduce inference time`* using OpenCL as much as possible.__

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

The single most important thing to consider when optimizing an OpenCL program is reducing global memory accesses as much as possible. Most of the techniques used in this project are variants of reducing global memory accesses.

   * **Tiled Convolution & SGEMM**
   * **More Work Per Thread**
   * **Using Vector Datatypes**
   * **Register Tiling**


## Performance Result

The test was performed on three different computers.

   * ***Computer 1** (Desktop)*

      | Run Type | Processor | Host Memory | Dedicated Memory | Elapsed Time | ET/Image |
      |:-:|:-:|:-:|:-:|:-:|:-:|
      | **Sequential** | `Intel i5-10400` | 32 GB DDR4  | - | 614 s (500 images) | 1.2270 seconds |
      | **OpenCL** | `Intel UHD Graphics 630` | DDR4 | No Dedicated Memory | 471 s (10000) | 0.0479 s |
      | **OpenCL** | `NVIDIA RTX 3060` | DDR4 | 12 GB GDDR6 | 29.1 s (10000) | 0.0029 s |
      |||||||
      | ***Performance Improvement*** ||||| $\times$ ***423*** |

   * ***Computer 2** (Laptop)*

      | Run Type | Processor | Host Memory | Dedicated Memory | Elapsed Time | ET/Image |
      |:-:|:-:|:-:|:-:|:-:|:-:|
      | **Sequential** | `Intel i5-1240P` | 16 GB LPDDR5 | - | 467 s (500) | 0.9349 s |
      | **OpenCL** | `Intel Iris Xe Graphics` (80EU) | LPDDR5 | No Dedicated Memory | 160 s (10000) | 0.0160 s |
      |||||||
      | ***Performance Improvement*** ||||| $\times$ ***58*** |

   * ***Computer 3** (Tablet)*

      | Run Type | Processor | Host Memory | Dedicated Memory | Elapsed Time | ET/Image |
      |:-:|:-:|:-:|:-:|:-:|:-:|
      | **Sequential** | `Intel m3-6Y30` | 4 GB LPDDR3  | - | 1345 s (500) | 2.6919 s |
      | **OpenCL** | `Intel HD Graphics 515` | LPDDR3 | No Dedicated Memory | 774 s (10000) | 0.0774 s |
      |||||||
      | ***Performance Improvement*** ||||| $\times$ ***35*** |


## License

   > TODO