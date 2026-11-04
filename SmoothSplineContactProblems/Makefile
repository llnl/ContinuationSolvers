MFEM_DIR ?= /home/tucker/software/glvis_mfem/mfem
MFEM_BUILD_DIR ?= $(MFEM_DIR)/parallel_mumps_build
CONFIG_MK := $(MFEM_BUILD_DIR)/config/config.mk
MFEM_LIB_FILE := $(MFEM_BUILD_DIR)/libmfem.a

-include $(CONFIG_MK)

MFEM_CXX ?= mpicxx

FRAMEWORK_DIR := ../continuationsolvers
PROBLEM_DIR := $(FRAMEWORK_DIR)/problems
SOLVER_DIR := $(FRAMEWORK_DIR)/solvers
SPLINE_DIR := Spline
DRIVER_DIR := SplineDrivers

FRAMEWORK_SOURCES := \
	$(FRAMEWORK_DIR)/utilities.cpp \
	$(wildcard $(PROBLEM_DIR)/*.cpp) \
	$(wildcard $(SOLVER_DIR)/*.cpp)
FRAMEWORK_OBJECTS := $(FRAMEWORK_SOURCES:.cpp=.o)

QUARTIC_OBJECT := $(SPLINE_DIR)/quartic_spline_1d.o
QUINTIC_OBJECT := $(SPLINE_DIR)/quintic_smoothing_spline_1d.o
CLOSEST_POINT_OBJECT := $(SPLINE_DIR)/spline_closest_point_1d.o
FIXED_CONTACT_OBJECT := $(DRIVER_DIR)/spline_contact_problem.o
PARAMETRIC_CONTACT_OBJECT := \
	$(DRIVER_DIR)/parametric_spline_contact_problem.o

PROGRAMS := \
	quartic_spline_example \
	quartic_spline_third_derivative_refinement \
	quintic_smoothing_spline_example \
	quintic_smoothing_spline_third_derivative_refinement \
	elastic_body_mesh_example \
	elastic_body_nocontact_example \
	spline_contact_example \
	spline_closest_point_derivative_check \
	spline_closest_point_parametric_derivative_check

all: $(PROGRAMS)

quartic_spline_example: \
	$(SPLINE_DIR)/quartic_spline_example.o $(QUARTIC_OBJECT) \
	$(MFEM_LIB_FILE)

quartic_spline_third_derivative_refinement: \
	$(SPLINE_DIR)/quartic_spline_third_derivative_refinement.o \
	$(QUARTIC_OBJECT) $(MFEM_LIB_FILE)

quintic_smoothing_spline_example: \
	$(SPLINE_DIR)/quintic_smoothing_spline_example.o $(QUINTIC_OBJECT) \
	$(MFEM_LIB_FILE)

quintic_smoothing_spline_third_derivative_refinement: \
	$(SPLINE_DIR)/quintic_smoothing_spline_third_derivative_refinement.o \
	$(QUINTIC_OBJECT) $(MFEM_LIB_FILE)

elastic_body_mesh_example: \
	$(DRIVER_DIR)/elastic_body_mesh_example.o $(MFEM_LIB_FILE)

elastic_body_nocontact_example: \
	$(DRIVER_DIR)/elastic_body_nocontact_example.o $(MFEM_LIB_FILE)

spline_contact_example: \
	$(DRIVER_DIR)/spline_contact_example.o \
	$(FIXED_CONTACT_OBJECT) $(CLOSEST_POINT_OBJECT) $(QUINTIC_OBJECT) \
	$(FRAMEWORK_OBJECTS) $(MFEM_LIB_FILE)

spline_closest_point_derivative_check: \
	$(DRIVER_DIR)/spline_closest_point_derivative_check.o \
	$(FIXED_CONTACT_OBJECT) $(CLOSEST_POINT_OBJECT) $(QUINTIC_OBJECT) \
	$(FRAMEWORK_OBJECTS) $(MFEM_LIB_FILE)

spline_closest_point_parametric_derivative_check: \
	$(DRIVER_DIR)/spline_closest_point_parametric_derivative_check.o \
	$(PARAMETRIC_CONTACT_OBJECT) $(CLOSEST_POINT_OBJECT) $(QUINTIC_OBJECT) \
	$(FRAMEWORK_OBJECTS) $(MFEM_LIB_FILE)

$(PROGRAMS):
	$(MFEM_CXX) $(MFEM_FLAGS) $(filter %.o,$^) -o $@ $(MFEM_LIBS)

%.o: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -MMD -MP -c -o $@ $<

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built at $(MFEM_BUILD_DIR))

DEPENDENCY_FILES := \
	$(FRAMEWORK_OBJECTS:.o=.d) \
	$(wildcard $(SPLINE_DIR)/*.d) \
	$(wildcard $(DRIVER_DIR)/*.d)
-include $(DEPENDENCY_FILES)

.PHONY: all clean
clean:
	rm -f $(PROGRAMS) \
		$(SPLINE_DIR)/*.o $(SPLINE_DIR)/*.d \
		$(DRIVER_DIR)/*.o $(DRIVER_DIR)/*.d \
		$(FRAMEWORK_OBJECTS) $(FRAMEWORK_OBJECTS:.o=.d)
