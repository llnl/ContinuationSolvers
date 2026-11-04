# Smooth spline contact problems

This directory contains the spline and smooth-contact work independently of
`TestProblems`.

- `Spline/` contains the parallel quartic and quintic smoothing splines and
  the closest-point projection implementation.
- `SplineDrivers/` contains the elastic-body, fixed-contact, parameterized
  contact, design-objective, and derivative-verification sources.
- `elastic_body.mesh` is the input mesh used by the contact drivers.

The sources use the framework in `../continuationsolvers` and an MFEM build
with MPI, Hypre, and MUMPS. By default the Makefile expects
`$(MFEM_DIR)/parallel_mumps_build`; override `MFEM_DIR` or `MFEM_BUILD_DIR`
when necessary.

Build every available driver with

```bash
make
```

or build an individual program, for example

```bash
make spline_contact_example
make spline_closest_point_parametric_derivative_check
```

Run the contact programs from this directory so the default mesh path finds
`elastic_body.mesh`:

```bash
mpirun -np 4 ./spline_contact_example
mpirun -np 4 ./spline_closest_point_parametric_derivative_check
```

`spline_contact_design_problem.*` is included as the upper-level MPEC
objective implementation. The repository snapshot supplied for this update
did not contain `spline_contact_design_example.cpp`, so the Makefile does not
advertise a non-existent design-driver target.
