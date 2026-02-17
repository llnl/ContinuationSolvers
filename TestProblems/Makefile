# Use the MFEM build directory
MFEM_DIR ?= /home/tucker/software/glvis_mfem/mfem
MFEM_BUILD_DIR ?= $(MFEM_DIR)/parallel_mumps_build


SRC=../continuationsolvers



CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = $(MFEM_BUILD_DIR)/libmfem.a
#MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MFEM_CXX=mpicxx
MFEM_FLAGS1=$(MFEM_FLAGS)



# Remove built-in rule
%: %.cpp
%: %.c
%.o: %.cpp


PROB_DIR=$(SRC)/problems
SOLVER_DIR=$(SRC)/solvers
TESTLIB_DIR=lib

FRAMEWORK_SRC_FILES = $(SRC)/utilities.cpp $(wildcard $(PROB_DIR)/*.cpp) $(wildcard $(SOLVER_DIR)/*.cpp)
FRAMEWORK_OBJS = $(FRAMEWORK_SRC_FILES:.cpp=.o)


%.o: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

$(SRC)/%.o: $(SRC)/%.cpp $(wildcard $(SRC)/%.hpp) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c -o $@ $< 

$(PROB_DIR)/%.o: $(PROB_DIR)/%.cpp $(wildcard $(PROB_DIR)/%.hpp) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c -o $@ $< 

$(SOLVER_DIR)/%.o: $(SOLVER_DIR)/%.cpp $(wildcard $(SOLVER_DIR)/%.hpp) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c -o $@ $< 


TestProblem1: TestProblem1.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem1.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem1a: TestProblem1a.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem1a.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem1b: TestProblem1b.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem1b.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem1c: TestProblem1c.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem1c.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)


TestProblem2: TestProblem2.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem2.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem3: TestProblem3.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem3.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem4: TestProblem4.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem4.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem4a: TestProblem4a.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem4a.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestProblem5: TestProblem5.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestProblem5.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

TestObstacleProblem: TestObstacleProblem.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TestObstacleProblem.o $(FRAMEWORK_OBJS) -o $@ $(MFEM_LIBS)

LibTestProblem: LibTestProblem.o $(FRAMEWORK_OBJS) $(MFEM_LIB_FILE)
	cd $(TESTLIB_DIR); make diablolib.a;
	$(MFEM_CXX) $(MFEM_FLAGS) LibTestProblem.o $(FRAMEWORK_OBJS) -o $@ lib/diablolib.a $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)


.PHONY: clean
clean:
	rm -f *.o $(PROB_DIR)/*.o $(SOLVER_DIR)/*.o $(TEST_LIB_DIR)/*.o $(TEST_LIB_DIR)/*.a TestProblem1 TestProblem1a TestProblem1b TestProblem1c TestProblem2 TestProblem3 TestProblem4 TestProblem4a TestProblem5 LibTestProblem
