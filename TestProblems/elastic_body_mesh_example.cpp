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

/// Construct
///
///   ([-1,1] x [1.5,2.5]) union
///   {(x,y) : x^2 + (y-1.5)^2 <= 1 and y <= 1.5}.
///
/// The semicircle is divided into radial layers and angular sectors.  Its
/// diameter has 2*radial_layers segments.  The rectangle uses exactly the
/// same diameter vertices, so y=1.5 is a conforming interior interface and
/// not a mesh boundary.
mfem::Mesh MakeElasticBodyMesh(int radial_layers,
                               int angular_sectors,
                               int rectangle_vertical_layers)
{
   MFEM_VERIFY(radial_layers >= 1,
               "At least one semicircle radial layer is required.");
   MFEM_VERIFY(angular_sectors >= 2,
               "At least two semicircle angular sectors are required.");
   MFEM_VERIFY(rectangle_vertical_layers >= 1,
               "At least one rectangle vertical layer is required.");

   constexpr int dimension = 2;
   constexpr int element_attribute = 1;
   constexpr mfem::real_t center_y = 1.5;
   constexpr mfem::real_t rectangle_height = 1.0;
   const mfem::real_t pi = std::acos(mfem::real_t(-1.0));

   const int rectangle_horizontal_layers = 2 * radial_layers;

   // One center vertex; angular_sectors+1 vertices on every nonzero ring;
   // and rectangle rows above the shared diameter.
   const int number_of_vertices =
      1 + radial_layers * (angular_sectors + 1) +
      rectangle_vertical_layers * (rectangle_horizontal_layers + 1);

   // The innermost half-disk layer is a fan.  Every other half-disk cell
   // and every rectangle cell is split into two triangles.
   const int number_of_elements =
      angular_sectors +
      2 * (radial_layers - 1) * angular_sectors +
      2 * rectangle_vertical_layers * rectangle_horizontal_layers;

   const int number_of_boundary_elements =
      angular_sectors + 2 * rectangle_vertical_layers +
      rectangle_horizontal_layers;

   mfem::Mesh mesh(dimension, number_of_vertices, number_of_elements,
                   number_of_boundary_elements, dimension);

   const int center = mesh.AddVertex(0.0, center_y);

   // ring[j][k] has radius j/radial_layers and angle
   // pi + k*pi/angular_sectors.  k=0 and k=angular_sectors lie on the
   // diameter; j=radial_layers is the candidate contact arc.
   std::vector<std::vector<int>> ring(
      static_cast<std::size_t>(radial_layers + 1));
   ring[0].push_back(center);

   for (int j = 1; j <= radial_layers; ++j)
   {
      const mfem::real_t radius =
         static_cast<mfem::real_t>(j) / radial_layers;
      ring[static_cast<std::size_t>(j)].resize(
         static_cast<std::size_t>(angular_sectors + 1));

      for (int k = 0; k <= angular_sectors; ++k)
      {
         mfem::real_t x = 0.0;
         mfem::real_t y = center_y;
         if (k == 0)
         {
            x = -radius;
         }
         else if (k == angular_sectors)
         {
            x = radius;
         }
         else
         {
            const mfem::real_t angle =
               pi + pi * static_cast<mfem::real_t>(k) / angular_sectors;
            x = radius * std::cos(angle);
            y = center_y + radius * std::sin(angle);
         }

         ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
            mesh.AddVertex(x, y);
      }
   }

   // Triangular fan next to the center.
   for (int k = 0; k < angular_sectors; ++k)
   {
      mesh.AddTriangle(center,
                       ring[1][static_cast<std::size_t>(k)],
                       ring[1][static_cast<std::size_t>(k + 1)],
                       element_attribute);
   }

   // Triangulate the cells between successive half-rings.
   for (int j = 1; j < radial_layers; ++j)
   {
      for (int k = 0; k < angular_sectors; ++k)
      {
         const int inner_left =
            ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
         const int outer_left =
            ring[static_cast<std::size_t>(j + 1)]
                [static_cast<std::size_t>(k)];
         const int outer_right =
            ring[static_cast<std::size_t>(j + 1)]
                [static_cast<std::size_t>(k + 1)];
         const int inner_right =
            ring[static_cast<std::size_t>(j)]
                [static_cast<std::size_t>(k + 1)];

         mesh.AddTriangle(inner_left, outer_left, outer_right,
                          element_attribute);
         mesh.AddTriangle(inner_left, outer_right, inner_right,
                          element_attribute);
      }
   }

   // rectangle[row][column], with row zero equal to the already-created
   // diameter vertices.  Columns run from x=-1 to x=1.
   std::vector<std::vector<int>> rectangle(
      static_cast<std::size_t>(rectangle_vertical_layers + 1),
      std::vector<int>(
         static_cast<std::size_t>(rectangle_horizontal_layers + 1), -1));

   for (int column = 0; column <= rectangle_horizontal_layers; ++column)
   {
      if (column < radial_layers)
      {
         const int j = radial_layers - column;
         rectangle[0][static_cast<std::size_t>(column)] =
            ring[static_cast<std::size_t>(j)][0];
      }
      else if (column == radial_layers)
      {
         rectangle[0][static_cast<std::size_t>(column)] = center;
      }
      else
      {
         const int j = column - radial_layers;
         rectangle[0][static_cast<std::size_t>(column)] =
            ring[static_cast<std::size_t>(j)]
                [static_cast<std::size_t>(angular_sectors)];
      }
   }

   for (int row = 1; row <= rectangle_vertical_layers; ++row)
   {
      const mfem::real_t y =
         center_y + rectangle_height * static_cast<mfem::real_t>(row) /
                       rectangle_vertical_layers;
      for (int column = 0; column <= rectangle_horizontal_layers; ++column)
      {
         const mfem::real_t x =
            -1.0 + 2.0 * static_cast<mfem::real_t>(column) /
                      rectangle_horizontal_layers;
         rectangle[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)] = mesh.AddVertex(x, y);
      }
   }

   for (int row = 0; row < rectangle_vertical_layers; ++row)
   {
      for (int column = 0; column < rectangle_horizontal_layers; ++column)
      {
         const int lower_left =
            rectangle[static_cast<std::size_t>(row)]
                     [static_cast<std::size_t>(column)];
         const int lower_right =
            rectangle[static_cast<std::size_t>(row)]
                     [static_cast<std::size_t>(column + 1)];
         const int upper_right =
            rectangle[static_cast<std::size_t>(row + 1)]
                     [static_cast<std::size_t>(column + 1)];
         const int upper_left =
            rectangle[static_cast<std::size_t>(row + 1)]
                     [static_cast<std::size_t>(column)];

         mesh.AddTriangle(lower_left, lower_right, upper_right,
                          element_attribute);
         mesh.AddTriangle(lower_left, upper_right, upper_left,
                          element_attribute);
      }
   }

   // Boundary orientation is counterclockwise around the body.  The
   // diameter y=1.5 is intentionally omitted because it is interior.
   for (int k = 0; k < angular_sectors; ++k)
   {
      mesh.AddBdrSegment(
         ring[static_cast<std::size_t>(radial_layers)]
             [static_cast<std::size_t>(k)],
         ring[static_cast<std::size_t>(radial_layers)]
             [static_cast<std::size_t>(k + 1)],
         CONTACT);
   }

   for (int row = 0; row < rectangle_vertical_layers; ++row)
   {
      mesh.AddBdrSegment(
         rectangle[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(rectangle_horizontal_layers)],
         rectangle[static_cast<std::size_t>(row + 1)]
                  [static_cast<std::size_t>(rectangle_horizontal_layers)],
         RIGHT);
   }

   for (int column = rectangle_horizontal_layers; column > 0; --column)
   {
      mesh.AddBdrSegment(
         rectangle[static_cast<std::size_t>(rectangle_vertical_layers)]
                  [static_cast<std::size_t>(column)],
         rectangle[static_cast<std::size_t>(rectangle_vertical_layers)]
                  [static_cast<std::size_t>(column - 1)],
         TOP);
   }

   for (int row = rectangle_vertical_layers; row > 0; --row)
   {
      mesh.AddBdrSegment(
         rectangle[static_cast<std::size_t>(row)][0],
         rectangle[static_cast<std::size_t>(row - 1)][0],
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

   int radial_layers = 8;
   int angular_sectors = 32;
   int rectangle_vertical_layers = 8;
   const char *serial_mesh_file = "elastic_body.mesh";
   const char *paraview_prefix = "ParaView";

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&radial_layers, "-nr", "--radial-layers",
                  "Number of radial layers in the lower semicircle.");
   args.AddOption(&angular_sectors, "-na", "--angular-sectors",
                  "Number of angular sectors along the contact arc.");
   args.AddOption(&rectangle_vertical_layers, "-ny", "--vertical-layers",
                  "Number of vertical layers in the rectangle.");
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

   if (radial_layers < 1 || angular_sectors < 2 ||
       rectangle_vertical_layers < 1)
   {
      if (rank == 0)
      {
         std::cerr << "Expected nr >= 1, na >= 2, and ny >= 1.\n";
      }
      MPI_Finalize();
      return 2;
   }

   if (rank == 0) { args.PrintOptions(std::cout); }

   mfem::Mesh serial_mesh = MakeElasticBodyMesh(
      radial_layers, angular_sectors, rectangle_vertical_layers);

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
