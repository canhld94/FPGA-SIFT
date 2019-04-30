# README: FPGA-SIFT


#### Network Computing Lab (ncl.kaist.ac.kr)
- This repository contains some opencl kernels and host codes to run sift algorithm in an FPGA with OpenCL

#### READ ME ! for the contribution

- please fork the repository for your own.
- After update your codes, please send pull request to the PM.
- Please do not push any "IDE related files"
- Please do not push any "binary file - not complete one but temporary version"
- Please feel free to ask about this repo. 
- whenever you push, please attach issue number!
- Plz test your code before push your local repo.!

#### Project management
- bin/: prebuild program and report
- data/: data for testing
- include/: header files
- kernel/: kernel codes
- source/: host codes
- License: license file
- makefile and makeemconfig.mk: makefile

#### How to build the code
- Requirement: OpenCV version >= 3.2; SDAccel version >= 2017.1
- make options:  
>>>exe: build the host code  
>>>xclbin: build the kernel code  
>>>emconf: build the emulation configure file  
>>>clean: clean the host code  
>>>cleanall: clean everything  

### Release 0.1 - 20180710

- Image size: 1024x1024
- Only Gaussian pyramid was process on FPGA
- Kernel clock frequency: 200 MHz
- Device execution time: 19.28ms (including both FPGA processing time, image resizing time on host code and data transfer time)
- FPGA processing time only: ~ 8ms
- Total time for building Gaussian pyramid and DoG pyramid: ~24ms

