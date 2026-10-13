# MFEM configuration
MFEM_DIR       ?= /home/tucker/software/glvis_mfem/mfem
MFEM_BUILD_DIR ?= $(MFEM_DIR)/parallel_mumps_build

CONFIG_MK    := $(MFEM_BUILD_DIR)/config/config.mk
MFEM_LIB_FILE := $(MFEM_BUILD_DIR)/libmfem.a

-include $(CONFIG_MK)

MFEM_CXX ?= mpicxx

# ContinuationSolvers sources
SRC         := ../continuationsolvers
PROB_DIR    := $(SRC)/problems
SOLVER_DIR  := $(SRC)/solvers
TESTLIB_DIR := lib

FRAMEWORK_SRC_FILES := \
	$(SRC)/utilities.cpp \
	$(wildcard $(PROB_DIR)/*.cpp) \
	$(wildcard $(SOLVER_DIR)/*.cpp)

FRAMEWORK_OBJS := $(FRAMEWORK_SRC_FILES:.cpp=.o)

PROGRAMS := \
	TestProblem1 \
	TestProblem1a \
	TestProblem1b \
	TestProblem1c \
	TestProblem1d \
	TestProblem2 \
	TestProblem3 \
	TestProblem4 \
	TestProblem4a \
	TestProblem5 \
	TestObstacleProblem \
	TestSphericalObstacleProblem \
	ObstacleDesignProblem \
	elastic_body_mesh_example \
	elastic_body_elasticity_example

TESTLIB := $(TESTLIB_DIR)/diablolib.a

PROGRAM_OBJS := $(PROGRAMS:%=%.o)
ALL_OBJS     := $(FRAMEWORK_OBJS) $(PROGRAM_OBJS) LibTestProblem.o
DEP_FILES    := $(ALL_OBJS:.o=.d)

.PHONY: all clean

all: $(PROGRAMS) LibTestProblem

# Compile local and framework sources with automatic header dependencies.
%.o: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -MMD -MP -c -o $@ $<

# All ordinary test programs use the same link rule.
$(PROGRAMS): %: %.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $< $(FRAMEWORK_OBJS) \
		-o $@ $(MFEM_LIBS)

# Special external test library.
$(TESTLIB): $(TESTLIB_DIR)/diablo.c $(TESTLIB_DIR)/diablo.h
	$(MAKE) -C $(TESTLIB_DIR) diablolib.a

LibTestProblem: LibTestProblem.o $(FRAMEWORK_OBJS) \
                $(MFEM_LIB_FILE) $(TESTLIB)
	$(MFEM_CXX) $(MFEM_FLAGS) LibTestProblem.o $(FRAMEWORK_OBJS) \
		-o $@ $(TESTLIB) $(MFEM_LIBS)

# Give a clear error if MFEM has not been built.
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	$(RM) $(ALL_OBJS) $(DEP_FILES) $(PROGRAMS) LibTestProblem
	$(MAKE) -C $(TESTLIB_DIR) clean

-include $(DEP_FILES)
