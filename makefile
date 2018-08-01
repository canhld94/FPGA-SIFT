include makeemconfig.mk

# compiler tools
XILINX_VIVADO_HLS ?= $(XILINX_SDX)/Vivado_HLS

SDX_CXX ?= $(XILINX_SDX)/bin/xcpp
XOCC ?= $(XILINX_SDX)/bin/xocc
EMCONFIGUTIL = $(XILINX_SDX)/bin/emconfigutil --od .
RM = rm -f
RMDIR = rm -rf

SDX_PLATFORM = xilinx:kcu1500:4ddr-xpr:4.0

INCLUDE += include

# host compiler global settings
CXXFLAGS += -std=c++0x -I/opt/Xilinx/SDx/2017.1/runtime/include/1_2/ -I/opt/Xilinx/SDx/2017.1/Vivado_HLS/include/ -I$(INCLUDE) -O3 -Wall -c -fmessage-length=0 -fopenmp
LDFLAGS += -lxilinxopencl -lopencv_imgcodecs -lopencv_highgui -lopencv_calib3d -lopencv_imgproc -lpthread -lrt -lstdc++ -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_cudafeatures2d -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/opt/Xilinx/SDx/2017.1/runtime/lib/x86_64 -fopenmp -gomp

# kernel compiler global settings
XOCC_OPTS = -t hw --platform /opt/kcu1500_custom_reference_design/xilinx_kcu1500_4ddr-xpr_4_0/xilinx_kcu1500_4ddr-xpr_4_0.xpfm --save-temps -O3 --report system

#
# OpenCL kernel files
#

BINARY_CONTAINERS += sift.xclbin

FPGA_BUILD_DIRS += fpga

SIFT_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s0.xo
ALL_KERNEL_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s0.xo
SIFT_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s1.xo
ALL_KERNEL_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s1.xo
SIFT_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s2.xo
ALL_KERNEL_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s2.xo
SIFT_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s3.xo
ALL_KERNEL_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s3.xo
SIFT_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s4.xo
ALL_KERNEL_OBJS += $(FPGA_BUILD_DIRS)/gaussian_s4.xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

#
# host files
#

HOST_OBJECTS += $(HOST_BUILD_DIRS)/main.o
HOST_OBJECTS += $(HOST_BUILD_DIRS)/sift.o
HOST_OBJECTS += $(HOST_BUILD_DIRS)/xcl2.o

HOST_EXE = sift.exe

HOST_BUILD_DIRS += host

EMCONFIG_FILE = emconfig.json

#
# primary build targets
#

.PHONY: exe xclbin emconf all clean cleanall

exe:
		@echo "*******************************************"
		@echo "          BUILDING HOST CODE...            "
		@echo "*******************************************"
		@mkdir -p $(HOST_BUILD_DIRS)
		$(SDX_CXX) $(CXXFLAGS) -o $(HOST_BUILD_DIRS)/main.o source/main.cpp
		$(SDX_CXX) $(CXXFLAGS) -o $(HOST_BUILD_DIRS)/sift.o source/sift.cpp
		$(SDX_CXX) $(CXXFLAGS) -o $(HOST_BUILD_DIRS)/xcl2.o source/xcl2.cpp
		$(SDX_CXX) -o $(HOST_EXE) $(HOST_OBJECTS) $(LDFLAGS)

xclbin:
		@echo "*******************************************"
		@echo "          BUILDING KERNEL CODE...          "
		@echo "*******************************************"
		@mkdir -p $(FPGA_BUILD_DIRS)
		@echo building kernel gaussian_s0...
		$(XOCC) $(XOCC_OPTS) -c -k gaussian_s0 --memory_port_data_width gaussian_s0:128 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_compile_sift_gaussian_s0 --max_memory_ports gaussian_s1 --max_memory_ports gaussian_s2 --max_memory_ports gaussian_s3 -o $(FPGA_BUILD_DIRS)/gaussian_s0.xo kernel/conv2d.cl
		-@$(RMDIR) .Xil
		
		@echo building kernel gaussian_s1...
		$(XOCC) $(XOCC_OPTS) -c -k gaussian_s1 --memory_port_data_width gaussian_s1:128 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_compile_sift_gaussian_s1 --max_memory_ports gaussian_s1 --max_memory_ports gaussian_s2 --max_memory_ports gaussian_s3 -o $(FPGA_BUILD_DIRS)/gaussian_s1.xo kernel/conv2d.cl
		-@$(RMDIR) .Xil
		
		@echo building kernel gaussian_s2...
		$(XOCC) $(XOCC_OPTS) -c -k gaussian_s2 --memory_port_data_width gaussian_s2:128 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_compile_sift_gaussian_s2 --max_memory_ports gaussian_s1 --max_memory_ports gaussian_s2 --max_memory_ports gaussian_s3 -o $(FPGA_BUILD_DIRS)/gaussian_s2.xo kernel/conv2d.cl
		-@$(RMDIR) .Xil
		
		@echo building kernel gaussian_s3...
		$(XOCC) $(XOCC_OPTS) -c -k gaussian_s3 --memory_port_data_width gaussian_s3:128 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_compile_sift_gaussian_s3 --max_memory_ports gaussian_s1 --max_memory_ports gaussian_s2 --max_memory_ports gaussian_s3 -o $(FPGA_BUILD_DIRS)/gaussian_s3.xo kernel/conv2d.cl
		-@$(RMDIR) .Xil
		
		@echo building kernel gaussian_s4...
		$(XOCC) $(XOCC_OPTS) -c -k gaussian_s4 --memory_port_data_width gaussian_s4:128 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_compile_sift_gaussian_s4 --max_memory_ports gaussian_s1 --max_memory_ports gaussian_s2 --max_memory_ports gaussian_s3 -o $(FPGA_BUILD_DIRS)/gaussian_s4.xo kernel/conv2d.cl
		-@$(RMDIR) .Xil
		
		@echo linking objects
		$(XOCC) $(XOCC_OPTS) -l --nk gaussian_s0:1  --nk gaussian_s1:1  --nk gaussian_s2:1  --nk gaussian_s3:1  --nk gaussian_s4:1 --xp misc:solution_name=$(FPGA_BUILD_DIRS)/_xocc_link_sift --kernel_frequency 200  --xp misc:map_connect=add.kernel.gaussian_s0_1.M_AXI_GMEM.core.OCL_REGION_0.M00_AXI --xp misc:map_connect=add.kernel.gaussian_s1_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI --xp misc:map_connect=add.kernel.gaussian_s1_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI --xp misc:map_connect=add.kernel.gaussian_s2_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI --xp misc:map_connect=add.kernel.gaussian_s2_1.M_AXI_GMEM1.core.OCL_REGION_0.M02_AXI --xp misc:map_connect=add.kernel.gaussian_s3_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI --xp misc:map_connect=add.kernel.gaussian_s3_1.M_AXI_GMEM1.core.OCL_REGION_0.M03_AXI --xp misc:map_connect=add.kernel.gaussian_s4_1.M_AXI_GMEM.core.OCL_REGION_0.M00_AXI -o $(BINARY_CONTAINERS) $(SIFT_OBJS)
			-@$(RMDIR) .Xil

emconf: 
		@echo "*******************************************"
		@echo "   BUILDING EMULATION CONFIGURE FILE...    "
		@echo "*******************************************"
		$(EMCONFIGUTIL) --platform /opt/kcu1500_custom_reference_design/xilinx_kcu1500_4ddr-xpr_4_0/xilinx_kcu1500_4ddr-xpr_4_0.xpfm --nd $(NUMBER_OF_DEVICES)
		-@$(RMDIR) TempConfig .Xil

all: exe xclbin emconf

clean: 
		-$(RM) $(HOST_EXE) $(HOST_OBJECTS)
		-$(RMDIR) $(HOST_BUILD_DIRS)

cleanall:
		-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES) $(EMCONFIG_FILE) $(HOST_EXE) $(HOST_OBJECTS)
		-$(RMDIR) $(FPGA_BUILD_DIRS) $(HOST_BUILD_DIRS)
		-$(RMDIR) _xocc*
		-$(RMDIR) .Xil

.DEFAULT_GOAL := all


