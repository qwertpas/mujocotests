# This Makefile assumes that GLFW is installed via Homebrew.
# If your setup is different, you will need to set GLFWROOT manually.

# This Makefile also assumes that MuJoCo.app is present in /Applications.

GLFWROOT?=$(shell brew --prefix)
MUJOCOPATH?=/Applications/MuJoCo.app/Contents/Frameworks

CFLAGS=-O2 -F$(MUJOCOPATH) -I$(GLFWROOT)/include -pthread
CXXFLAGS=$(CFLAGS) -std=c++17 -stdlib=libc++
ALLFLAGS=$(CXXFLAGS) -L$(GLFWROOT)/lib -Wl,-rpath,$(MUJOCOPATH)

.PHONY: all
all:

	clang++ $(ALLFLAGS)    satyrr.cpp     -framework mujoco -lglfw -o satyrr
	
# clang++ $(ALLFLAGS)    basic.cc      -framework mujoco -lglfw -o basic
# clang++ $(ALLFLAGS)    record.cc     -framework mujoco -lglfw -o record