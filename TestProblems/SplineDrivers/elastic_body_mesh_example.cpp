#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

enum BoundaryAttribute
{
   CONTACT = 1,
   RIGHT = 2,
   TOP = 3,
   LEFT = 4
};

/// Construct the domain
///
///   Omega = {(x,y) : -1 <= x <= 1, q(x) <= y <= 2.5},
///
/// where
///
///   q(x) = 1.5 - quadratic_depth * (1 - x^2).
///
/// Thus q(-1) = q(1) = 1.5 and q(0) = 1.5 - quadratic_depth.
/// The logical rectangular grid is mapped vertically between q(x) and the
/// horizontal top boundary.  Each mapped quadrilateral is divided into two
/// triangles.
mfem::Mesh MakeElasticBodyMesh(int horizontal_elements,
                               int vertical_elements,
                               mfem::real_t quadratic_depth)
{
   MFEM_VERIFY(horizontal_elements >= 2,
               "At least two horizontal elements are required.");
   MFEM_VERIFY(vertical_elements >= 1,
               "At least one vertical element is required.");
   MFEM_VERIFY(std::isfinite(quadratic_depth) && quadratic_depth > 0.0,
               "The quadratic depth must be finite and positive.");

   constexpr int dimension = 2;
   constexpr int element_attribute = 1;
   constexpr mfem::real_t side_bottom_y = 1.5;
   constexpr mfem::real_t top_y = 2.5;

   const int number_of_vertices =
      (horizontal_elements + 1) * (vertical_elements + 1);
   const int number_of_elements =
      2 * horizontal_elements * vertical_elements;
   const int number_of_boundary_elements =
      2 * horizontal_elements + 2 * vertical_elements;

   mfem::Mesh mesh(dimension, number_of_vertices, number_of_elements,
                   number_of_boundary_elements, dimension);

   std::vector<std::vector<int>> vertex(
      static_cast<std::size_t>(vertical_elements + 1),
      std::vector<int>(static_cast<std::size_t>(horizontal_elements + 1),
                       -1));

   for (int row = 0; row <= vertical_elements; ++row)
   {
      const mfem::real_t eta =
         static_cast<mfem::real_t>(row) / vertical_elements;
      for (int column = 0; column <= horizontal_elements; ++column)
      {
         const mfem::real_t x =
            -1.0 + 2.0 * static_cast<mfem::real_t>(column) /
                      horizontal_elements;
         const mfem::real_t bottom_y =
            side_bottom_y - quadratic_depth * (1.0 - x * x);
         const mfem::real_t y = bottom_y + eta * (top_y - bottom_y);

         vertex[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(column)] = mesh.AddVertex(x, y);
      }
   }

   for (int row = 0; row < vertical_elements; ++row)
   {
      for (int column = 0; column < horizontal_elements; ++column)
      {
         const int lower_left =
            vertex[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)];
         const int lower_right =
            vertex[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column + 1)];
         const int upper_right =
            vertex[static_cast<std::size_t>(row + 1)]
                  [static_cast<std::size_t>(column + 1)];
         const int upper_left =
            vertex[static_cast<std::size_t>(row + 1)]
                  [static_cast<std::size_t>(column)];

         mesh.AddTriangle(lower_left, lower_right, upper_right,
                          element_attribute);
         mesh.AddTriangle(lower_left, upper_right, upper_left,
                          element_attribute);
      }
   }

   // Traverse the boundary counterclockwise: contact from left to right,
   // right side upward, top from right to left, and left side downward.
   for (int column = 0; column < horizontal_elements; ++column)
   {
      mesh.AddBdrSegment(
         vertex[0][static_cast<std::size_t>(column)],
         vertex[0][static_cast<std::size_t>(column + 1)],
         CONTACT);
   }

   for (int row = 0; row < vertical_elements; ++row)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(horizontal_elements)],
         vertex[static_cast<std::size_t>(row + 1)]
               [static_cast<std::size_t>(horizontal_elements)],
         RIGHT);
   }

   for (int column = horizontal_elements; column > 0; --column)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(vertical_elements)]
               [static_cast<std::size_t>(column)],
         vertex[static_cast<std::size_t>(vertical_elements)]
               [static_cast<std::size_t>(column - 1)],
         TOP);
   }

   for (int row = vertical_elements; row > 0; --row)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(row)][0],
         vertex[static_cast<std::size_t>(row - 1)][0],
         LEFT);
   }

   mesh.FinalizeTriMesh(/*generate_edges=*/1,
                        /*refine=*/1,
                        /*fix_orientation=*/true);
   return mesh;
}

} // namespace

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int horizontal_elements = 32;
   int vertical_elements = 16;
   mfem::real_t quadratic_depth = 1.0;
   const char *serial_mesh_file = "elastic_body.mesh";
   const char *paraview_prefix = "ParaView";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&horizontal_elements, "-nx", "--horizontal-elements",
                  "Number of elements along the quadratic contact boundary.");
   args.AddOption(&vertical_elements, "-ny", "--vertical-elements",
                  "Number of mapped elements between the contact and top "
                  "boundaries.");
   args.AddOption(&quadratic_depth, "-d", "--quadratic-depth",
                  "Depth 1.5-q(0) of the convex quadratic boundary.");
   args.AddOption(&serial_mesh_file, "-m", "--mesh-file",
                  "Output MFEM mesh file.");
   args.AddOption(&paraview_prefix, "-p", "--paraview-prefix",
                  "Directory for parallel ParaView output.");
   args.Parse();

   if (!args.Good())
   {
      if (rank == 0) { args.PrintUsage(std::cout); }
      MPI_Finalize();
      return 1;
   }

   if (horizontal_elements < 2 || vertical_elements < 1 ||
       !std::isfinite(quadratic_depth) || quadratic_depth <= 0.0)
   {
      if (rank == 0)
      {
         std::cerr << "Expected nx >= 2, ny >= 1, and depth > 0.\n";
      }
      MPI_Finalize();
      return 2;
   }

   if (rank == 0) { args.PrintOptions(std::cout); }

   mfem::Mesh serial_mesh = MakeElasticBodyMesh(
      horizontal_elements, vertical_elements, quadratic_depth);

   if (rank == 0)
   {
      std::ofstream output(serial_mesh_file);
      MFEM_VERIFY(output.good(),
                  "Unable to open the serial mesh output file.");
      serial_mesh.Print(output);

      std::cout << "Serial mesh: " << serial_mesh.GetNV() << " vertices, "
                << serial_mesh.GetNE() << " triangles, "
                << serial_mesh.GetNBE() << " boundary segments.\n"
                << "Boundary attributes: contact=1, right=2, top=3, left=4.\n"
                << "Wrote " << serial_mesh_file << "\n";
   }

   mfem::ParMesh parallel_mesh(MPI_COMM_WORLD, serial_mesh);

   mfem::ParaViewDataCollection paraview("ElasticBodyMesh", &parallel_mesh);
   paraview.SetPrefixPath(paraview_prefix);
   paraview.SetCycle(0);
   paraview.SetTime(0.0);
   paraview.SetLevelsOfDetail(1);
   paraview.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview.SetHighOrderOutput(false);
   paraview.Save();

   if (rank == 0)
   {
       serial_mesh.PrintBdrVTU("ParaView/ElasticBodyMesh/elastic_body_boundary", mfem::VTKFormat::ASCII);    
       std::cout << "Wrote parallel ParaView collection under "
                << paraview_prefix << "/ElasticBodyMesh.\n";
   }

   MPI_Finalize();
   return 0;
}
