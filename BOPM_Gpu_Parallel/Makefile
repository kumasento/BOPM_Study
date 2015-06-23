
CXX=g++
CXXFLAGS=-O2 -std=c++0x -I$(OCLINCLUDE) -DPROGDEBUG -g

OCLSDK=/opt/AMDAPP
OCLINCLUDE=$(OCLSDK)/include
OCLLIB=$(OCLSDK)/lib/x86_64
LIBS=-lOpenCL

bopm: OpenCL_BOPM_Main.cpp opencl.o
	$(CXX) $(CXXFLAGS) -c opencl.cpp
	$(CXX) $(CXXFLAGS) -o bopm OpenCL_BOPM_Main.cpp opencl.o $(LIBS)

clean: 
	rm -f *~ *.o 
