/* Config parameters. */
#include "../config.h"

/* This object's header. */
#include "darkmatter_write_grids.h"

/* Local includes. */
#include <mpi.h>
#include <hdf5_hl.h>
//#include <hdf5.h>
#include "error.h"
#include "threadpool.h"

#define DS_NAME_SIZE 8

// TODO(smutch): Make this available from `mesh_gravity.c`
/**
 * @brief Returns 1D index of a 3D NxNxN array using row-major style.
 *
 * Wraps around in the corresponding dimension if any of the 3 indices is >= N
 * or < 0.
 *
 * @param i Index along x.
 * @param j Index along y.
 * @param k Index along z.
 * @param N Size of the array along one axis.
 */
__attribute__((always_inline)) INLINE static unsigned long long row_major_id_periodic(int i,
                                                                       int j,
                                                                       int k,
                                                                       int N) {
  unsigned long long ii = (unsigned long long)((i + N) % N);
  unsigned long long jj = (unsigned long long)((j + N) % N);
  unsigned long long kk = (unsigned long long)((k + N) % N);
  unsigned long long NN = (unsigned long long)N;
  return (ii * NN * NN + jj * NN + kk);
}

// TODO(smutch): Make this available from `mesh_gravity.c`
/**
 * @brief Interpolate a value to a mesh using CIC.
 *
 * @param mesh The mesh to write to
 * @param N The side-length of the mesh
 * @param i The index of the cell along x
 * @param j The index of the cell along y
 * @param k The index of the cell along z
 * @param tx First CIC coefficient along x
 * @param ty First CIC coefficient along y
 * @param tz First CIC coefficient along z
 * @param dx Second CIC coefficient along x
 * @param dy Second CIC coefficient along y
 * @param dz Second CIC coefficient along z
 * @param value The value to interpolate.
 */
__attribute__((always_inline)) INLINE static void CIC_set(
    float* mesh, int N, int i, int j, int k,
    float tx, float ty, float tz,
    float dx, float dy, float dz, float value) {

  /* Classic CIC interpolation */
  atomic_add_f(&mesh[row_major_id_periodic(i + 0, j + 0, k + 0, N)],
               value * tx * ty * tz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 0, j + 0, k + 1, N)],
               value * tx * ty * dz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 0, j + 1, k + 0, N)],
               value * tx * dy * tz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 0, j + 1, k + 1, N)],
               value * tx * dy * dz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 1, j + 0, k + 0, N)],
               value * dx * ty * tz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 1, j + 0, k + 1, N)],
               value * dx * ty * dz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 1, j + 1, k + 0, N)],
               value * dx * dy * tz);
  atomic_add_f(&mesh[row_major_id_periodic(i + 1, j + 1, k + 1, N)],
               value * dx * dy * dz);
}

/**
 * @brief Assigns a given #gpart property to a grid using the CIC method.
 *
 * @param gp The #gpart.
 * @param prop_offset The byte offset of the #gpart struct property to be gridded.
 * @param grid The grid.
 * @param dim the size of the grid along each axis.
 * @param fac The width of the grid cells in each dimension.
 * @param box_size The dimensions of the simulation box.
 */
__attribute__((always_inline)) INLINE static void part_to_grid_CIC(
    const struct gpart* gp, ptrdiff_t prop_offset,
    float* grid, const int dim,
    const double fac[3], const double box_size[3]) {

  /* Box wrap the multipole's position */
  const double pos_x = box_wrap(gp->x[0], 0., box_size[0]);
  const double pos_y = box_wrap(gp->x[1], 0., box_size[1]);
  const double pos_z = box_wrap(gp->x[2], 0., box_size[2]);

  /* Workout the CIC coefficients */
  int i = (int)(fac[0] * pos_x);
  if (i >= dim) i = dim - 1;
  const double dx = fac[0] * pos_x - i;
  const double tx = 1. - dx;

  int j = (int)(fac[1] * pos_y);
  if (j >= dim) j = dim - 1;
  const double dy = fac[1] * pos_y - j;
  const double ty = 1. - dy;

  int k = (int)(fac[2] * pos_z);
  if (k >= dim) k = dim - 1;
  const double dz = fac[2] * pos_z - k;
  const double tz = 1. - dz;

#ifdef SWIFT_DEBUG_CHECKS
  if (i < 0 || i >= dim) error("Invalid gpart position in x");
  if (j < 0 || j >= dim) error("Invalid gpart position in y");
  if (k < 0 || k >= dim) error("Invalid gpart position in z");
#endif

  const float val = (prop_offset < 0) ? 1.0 : *(float*)((char *)gp + prop_offset);

  /* CIC ! */
  CIC_set(grid, dim, i, j, k, tx, ty, tz, dx, dy, dz, val);
}

/**
 * @brief Shared information about the grid to be used by all the threads in the
 * pool.
 */
struct gridding_extra_data {
  double cell_size[3];
  double box_size[3];
  ptrdiff_t prop_offset;
  float* grid;
  int grid_dim;
  unsigned long long n_grid_points;
};

/**
 * @brief Threadpool mapper function for the grid CIC assignment.
 *
 * @param gparts_v The #gpart array recast as a void pointer
 * @param N The number of #gparts
 * @param extra_data_v Extra data to be passed to the gridding
 */
static void construct_grid_CIC_mapper(void* restrict gparts_v, int N,
                                      void* restrict extra_data_v) {

  const struct gridding_extra_data extra_data =
      *((const struct gridding_extra_data*)extra_data_v);
  const struct gpart* gparts = (struct gpart*)gparts_v;

  const double* cell_size = extra_data.cell_size;
  const double* box_size = extra_data.box_size;
  const double fac[3] = {1.0 / cell_size[0], 1.0 / cell_size[1],
                         1.0 / cell_size[2]};
  float* grid = extra_data.grid;
  const int grid_dim = extra_data.grid_dim;
  const ptrdiff_t prop_offset = extra_data.prop_offset;

  for (int ii = 0; ii < N; ++ii) {
    const struct gpart* gp = &(gparts[ii]);
    // Assumption here is that all particles have the same mass.
    part_to_grid_CIC(gp, prop_offset, grid, grid_dim, fac, box_size);
  }
}

/**
 * @brief Find the grid index for a #gpart position.
 *
 * @param gp The #gpart.
 * @param cell_size The size of each cell.
 * @param grid_dim The dimensionality of the grid.
 * @param n_grid_points The total number of points (cells) in the grid (used for checking).
 */
__attribute__((always_inline)) INLINE static unsigned long long part_to_grid_index(
    const struct gpart* gp, const double cell_size[3], const double grid_dim,
    const unsigned long long n_grid_points) {
  int coord[3] = {-1, -1, -1};

  for (int jj = 0; jj < 3; ++jj) {
    coord[jj] = (int)round(gp->x[jj] / cell_size[jj]);
  }

  unsigned long long index = row_major_id_periodic(coord[0], coord[1], coord[2], grid_dim);
  assert((index < n_grid_points));
  return index;
}

/**
 * @brief Threadpool mapper function for the grid NGP assignment.
 *
 * @param gparts_v The #gpart array recast as a void pointer
 * @param N The number of #gpart
 * @param extra_data_v The #gridding_extra_data data to be passed to the gridding recast as a void pointer
 */
static void construct_grid_NGP_mapper(void* restrict gparts_v, int N,
                                      void* restrict extra_data_v) {

  const struct gridding_extra_data extra_data =
      *((const struct gridding_extra_data*)extra_data_v);
  const struct gpart* gparts = (struct gpart*)gparts_v;

  const double* cell_size = extra_data.cell_size;
  float* grid = extra_data.grid;
  const int grid_dim = extra_data.grid_dim;
  const unsigned long long n_grid_points = extra_data.n_grid_points;
  const ptrdiff_t prop_offset = extra_data.prop_offset;

  for (int ii = 0; ii < N; ++ii) {
    const struct gpart* gp = &(gparts[ii]);
    unsigned long long index = part_to_grid_index(gp, cell_size, grid_dim, n_grid_points);
    const float val = (prop_offset < 0) ? 1.0 : *(float*)((char *)gp + prop_offset);
    atomic_add_f(&(grid[index]), val);
  }
}

/**
 * @brief Construct and write dark matter density and velocity grids.
 *
 * @param e The #engine.
 * @param Npart The number of #gpart
 * @param h_file The output HDF5 file handle.
 * @param internal_units The #unit_system used internally.
 * @param snapshot_units The #unit_system used in the snapshots.
 */
void darkmatter_write_grids(struct engine* e, const size_t Npart,
                            const hid_t h_file,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            const int grid_dim,
                            const char grid_method[PARSER_MAX_LINE_SIZE]
                        ) {

  struct gpart* gparts = e->s->gparts;
  const unsigned long long gd = (unsigned long long)grid_dim;
  const unsigned long long n_grid_points = gd * gd * gd;
  const double* box_size = e->s->dim;
  char dataset_name[DS_NAME_SIZE] = "";

  double cell_size[3] = {0, 0, 0};
  for (int ii = 0; ii < 3; ++ii) {
    cell_size[ii] = box_size[ii] / (double)grid_dim;
  }

  /* array to be used for all grids */
  float* grid = NULL;
  if (swift_memalign("writegrid", (void**)&grid, IO_BUFFER_ALIGNMENT,
                     n_grid_points * sizeof(float)) != 0) {
    error("Failed to allocate output DM grids! Requesting %d grid_dim giving %llu and %f GB of memory", grid_dim, n_grid_points, n_grid_points*sizeof(float)/1024.0/1024./1024.);
  }
  memset(grid, 0, n_grid_points * sizeof(float));

  /* Array to be used to store particle counts at all grid points. */
  float* point_counts = NULL;
  if (swift_memalign("countgrid", (void**)&point_counts, IO_BUFFER_ALIGNMENT,
                     n_grid_points * sizeof(float)) != 0) {
    error("Failed to allocate point counts! Requesting %d grid_dim giving %llu and %f GB of memory", grid_dim, n_grid_points, n_grid_points*sizeof(float)/1024.0/1024./1024.);
  }
  memset(point_counts, 0, n_grid_points * sizeof(float));

  /* Calculate information for the write that is not dependent on the property
     being written. */

  const char group_name[] = {"/PartType1/Grids"};
  hid_t h_grp = H5Gcreate(h_file, group_name, H5P_DEFAULT, H5P_DEFAULT,
                          H5P_DEFAULT);
  if (h_grp < 0) error("Error while creating dark matter grids group.");

  /* attach an attribute with the gridding type */
//  H5LTset_attribute_string(h_file, group_name, "gridding_method", e->snapshot_grid_method);

  int i_rank = 0, n_ranks = 1;
#ifdef WITH_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
#endif

  /* split the write into slabs on the x axis */
  int local_slab_size = grid_dim / n_ranks;
  int local_offset = local_slab_size * i_rank;
  if (i_rank == n_ranks - 1) {
    local_slab_size = grid_dim - local_offset;
  }

  /* create hdf5 properties, selections, etc. that will be used for all grid
   * writes */
  hsize_t dims[3] = {grid_dim, grid_dim, grid_dim};
  hid_t fspace_id = H5Screate_simple(3, dims, NULL);

  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 3, (hsize_t[3]){1, grid_dim, grid_dim});

  /* Uncomment this line to enable compression. */
  // H5Pset_deflate(dcpl_id, 6);

  hid_t plist_id;
#if defined(WITH_MPI) && defined(HAVE_PARALLEL_HDF5)
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
#else
  plist_id = H5P_DEFAULT;
#endif

  hsize_t start[3] = {local_offset, 0, 0};
  hsize_t count[3] = {local_slab_size, grid_dim, grid_dim};
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

  hsize_t mem_dims[3] = {local_slab_size, grid_dim, grid_dim};
  hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

  /* extra data for threadpool_map calls */
  struct gridding_extra_data extra_data;
  memcpy(extra_data.cell_size, cell_size, sizeof(double) * 3);
  memcpy(extra_data.box_size, box_size, sizeof(double) * 3);
  extra_data.grid_dim = grid_dim;
  extra_data.n_grid_points = n_grid_points;

  /* NOTE THE ORDER IS IMPORTANT HERE.  DENSITY must come first so that we have
   * point_counts available to do the averaging of the velocities. */
  enum grid_types { DENSITY, VELOCITY_X, VELOCITY_Y, VELOCITY_Z };

  void (*construct_grid_mapper)(void* restrict, int, void* restrict) = &construct_grid_NGP_mapper;
  if (strncmp(grid_method, "CIC", 3) == 0) {
    construct_grid_mapper = &construct_grid_CIC_mapper;
  } else if (strncmp(grid_method, "NGP", 3) != 0) {
    message(
        "WARNING: Unknown snapshot gridding method"
        "Falling back to NGP.");
  }

  for (enum grid_types grid_type = DENSITY; grid_type <= VELOCITY_Z;
       ++grid_type) {
    /* Loop through all particles and assign to the grid. */
    switch (grid_type) {
      case DENSITY:
        extra_data.grid = point_counts;
        extra_data.prop_offset = -1;
        break;

      case VELOCITY_X:
        extra_data.grid = grid;
        extra_data.prop_offset = offsetof(struct gpart, v_full[0]);
        break;

      case VELOCITY_Y:
        extra_data.grid = grid;
        extra_data.prop_offset = offsetof(struct gpart, v_full[1]);
        break;

      case VELOCITY_Z:
        extra_data.grid = grid;
        extra_data.prop_offset = offsetof(struct gpart, v_full[2]);
        break;
    }

    threadpool_map((struct threadpool*)&e->threadpool,
        construct_grid_mapper, gparts, Npart,
        sizeof(struct gpart), 0, (void*)&extra_data);

    /* Do any necessary conversions */
    switch (grid_type) {
      double n_to_density;
      double unit_conv_factor;
      case DENSITY:
#ifdef WITH_MPI
        /* reduce the grid */
        if (n_grid_points > INT_MAX) {
           // loop over as necessary
           int nloops = (int)(ceil((double)(n_grid_points)/(double)(INT_MAX)));
           unsigned long long offset = 0;
           int ncount = INT_MAX;
           for (int iloop = 0; iloop < nloops; iloop++){
              MPI_Allreduce(MPI_IN_PLACE, &point_counts[offset], ncount, MPI_FLOAT,
                            MPI_SUM, MPI_COMM_WORLD);
              offset += ncount;
              if (iloop == nloops -2) ncount = n_grid_points - offset;
              else ncount = INT_MAX;
           }
        }
        else {
          MPI_Allreduce(MPI_IN_PLACE, point_counts, n_grid_points, MPI_FLOAT,
                        MPI_SUM, MPI_COMM_WORLD);
        }
#endif

        /* convert n_particles to density */
        unit_conv_factor = units_conversion_factor(
            internal_units, snapshot_units, UNIT_CONV_DENSITY);
        n_to_density = gparts[0].mass * unit_conv_factor /
                       (cell_size[0] * cell_size[1] * cell_size[2]);
        for (unsigned long long ii = 0; ii < n_grid_points; ++ii) {
          grid[ii] = n_to_density * point_counts[ii];
        }
        break;

      case VELOCITY_X:
      case VELOCITY_Y:
      case VELOCITY_Z:
#ifdef WITH_MPI
        /* reduce the grid */
        if (n_grid_points > INT_MAX) {
           // loop over as necessary
           int nloops = (int)(ceil((double)(n_grid_points)/(double)(INT_MAX)));
           unsigned long long offset = 0;
           int ncount = INT_MAX;
           for (int iloop = 0; iloop < nloops; iloop++){
              MPI_Allreduce(MPI_IN_PLACE, &grid[offset], ncount, MPI_FLOAT,
                            MPI_SUM, MPI_COMM_WORLD);
              offset += ncount;
              if (iloop == nloops - 2) ncount = n_grid_points - offset;
              else ncount = INT_MAX;
           }
        }
        else {
          MPI_Allreduce(MPI_IN_PLACE, grid, n_grid_points, MPI_FLOAT,
                        MPI_SUM, MPI_COMM_WORLD);
        }
#endif

        /* take the mean */
        unit_conv_factor = units_conversion_factor(
            internal_units, snapshot_units, UNIT_CONV_VELOCITY);
        for (unsigned long long ii = 0; ii < n_grid_points; ++ii) {
            if (point_counts[ii] > 0.0) {
                grid[ii] *= unit_conv_factor / point_counts[ii];
            }
        }
        break;
    }

    switch (grid_type) {
      case DENSITY:
        snprintf(dataset_name, DS_NAME_SIZE, "Density");
        break;

      case VELOCITY_X:
        snprintf(dataset_name, DS_NAME_SIZE, "Vx");
        break;

      case VELOCITY_Y:
        snprintf(dataset_name, DS_NAME_SIZE, "Vy");
        break;

      case VELOCITY_Z:
        snprintf(dataset_name, DS_NAME_SIZE, "Vz");
        break;
    }

    /* actually do the write finally! */
    hid_t dset_id = H5Dcreate(h_grp, dataset_name, H5T_NATIVE_FLOAT, fspace_id,
                              H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
     unsigned long long local_grid_offset = row_major_id_periodic(local_offset, 0, 0, grid_dim);
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id, fspace_id, plist_id,
             &grid[local_grid_offset]);
    H5Dclose(dset_id);

    /* reset the grid if necessary */
    if (grid_type != VELOCITY_Z) {
      memset(grid, 0, n_grid_points * sizeof(float));
    }
  }

  H5Sclose(memspace_id);
  H5Pclose(plist_id);
  H5Pclose(dcpl_id);
  H5Sclose(fspace_id);
  H5Gclose(h_grp);

  /* free the grids */
  if (point_counts) swift_free("countgrid", point_counts);
  if (grid) swift_free("writegrid", grid);
}

#if defined(HAVE_HDF5)

/**
 * @brief Prepares a file for a parallel write.
 *
 * @param e The #engine.
 * @param baseName The base name of the snapshots.
 * @param N_total The total number of particles of each type to write.
 * @param internal_units The #unit_system used internally.
 * @param snapshot_units The #unit_system used in the snapshots.
 */
void prepare_density_grids_file(struct engine* e, const char* baseName, long long N_total[6],
                  const struct unit_system* internal_units,
                  const struct unit_system* snapshot_units,
                  const int output_count,
                  const bool iproducexmf
              ) {

  FILE* xmfFile = 0;
  int numFiles = 1;

  /* First time, we need to create the XMF file */
  if (iproducexmf) {
      if (output_count == 0) xmf_create_file(baseName);
      /* Prepare the XMF file for the new entry */
      xmfFile = xmf_prepare_file(baseName);
  }

  /* HDF5 File name */
  char fileName[FILENAME_BUFFER_SIZE];
  snprintf(fileName, FILENAME_BUFFER_SIZE, "%s_%04i.hdf5", baseName,
             output_count);

  /* Open HDF5 file with the chosen parameters */
  hid_t h_file = H5Fcreate(fileName, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (h_file < 0) error("Error while opening file '%s'.", fileName);

  if (iproducexmf) {
    /* Write the part of the XMF file corresponding to this
    * specific output */
    xmf_write_outputheader(xmfFile, fileName, e->time);
  }

  /* Open header to write simulation properties */
  /* message("Writing file header..."); */
  hid_t h_grp =
      H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (h_grp < 0) error("Error while creating file header\n");

  /* Convert basic output information to snapshot units */
  const double factor_time =
      units_conversion_factor(internal_units, snapshot_units, UNIT_CONV_TIME);
  const double factor_length =
      units_conversion_factor(internal_units, snapshot_units, UNIT_CONV_LENGTH);
  const double dblTime = e->time * factor_time;
  const double dim[3] = {e->s->dim[0] * factor_length,
                         e->s->dim[1] * factor_length,
                         e->s->dim[2] * factor_length};

  /* Print the relevant information and print status */
  io_write_attribute(h_grp, "BoxSize", DOUBLE, dim, 3);
  io_write_attribute(h_grp, "Time", DOUBLE, &dblTime, 1);
  const int dimension = (int)hydro_dimension;
  io_write_attribute(h_grp, "Dimension", INT, &dimension, 1);
  io_write_attribute(h_grp, "Redshift", DOUBLE, &e->cosmology->z, 1);
  io_write_attribute(h_grp, "Scale-factor", DOUBLE, &e->cosmology->a, 1);
  io_write_attribute_s(h_grp, "Code", "SWIFT");
  time_t tm = time(NULL);
  io_write_attribute_s(h_grp, "Snapshot date", ctime(&tm));
  io_write_attribute_s(h_grp, "RunName", e->run_name);

  /* GADGET-2 legacy values */
  /* Number of particles of each type */
  unsigned int numParticles[swift_type_count] = {0};
  unsigned int numParticlesHighWord[swift_type_count] = {0};
  for (int ptype = 0; ptype < swift_type_count; ++ptype) {
    numParticles[ptype] = (unsigned int)N_total[ptype];
    numParticlesHighWord[ptype] = (unsigned int)(N_total[ptype] >> 32);
  }
  io_write_attribute(h_grp, "NumPart_ThisFile", LONGLONG, N_total,
                     swift_type_count);
  io_write_attribute(h_grp, "NumPart_Total", UINT, numParticles,
                     swift_type_count);
  io_write_attribute(h_grp, "NumPart_Total_HighWord", UINT,
                     numParticlesHighWord, swift_type_count);
  double MassTable[6] = {0., 0., 0., 0., 0., 0.};
  io_write_attribute(h_grp, "MassTable", DOUBLE, MassTable, swift_type_count);
  unsigned int flagEntropy[swift_type_count] = {0};
  //flagEntropy[0] = writeEntropyFlag();
  io_write_attribute(h_grp, "Flag_Entropy_ICs", UINT, flagEntropy,
                     swift_type_count);
  io_write_attribute(h_grp, "NumFilesPerSnapshot", INT, &numFiles, 1);

  /* Close header */
  H5Gclose(h_grp);

  /* Print the code version */
  io_write_code_description(h_file);

  /* Print the run's policy */
  io_write_engine_policy(h_file, e);

  /* Print the cosmological parameters */
  h_grp =
      H5Gcreate(h_file, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (h_grp < 0) error("Error while creating cosmology group");
  if (e->policy & engine_policy_cosmology)
    io_write_attribute_i(h_grp, "Cosmological run", 1);
  else
    io_write_attribute_i(h_grp, "Cosmological run", 0);
  cosmology_write_model(h_grp, e->cosmology);
  H5Gclose(h_grp);

  /* Print the runtime parameters */
  h_grp =
      H5Gcreate(h_file, "/Parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (h_grp < 0) error("Error while creating parameters group");
  parser_write_params_to_hdf5(e->parameter_file, h_grp, 1);
  H5Gclose(h_grp);

  /* Print the runtime unused parameters */
  h_grp = H5Gcreate(h_file, "/UnusedParameters", H5P_DEFAULT, H5P_DEFAULT,
                    H5P_DEFAULT);
  if (h_grp < 0) error("Error while creating parameters group");
  parser_write_params_to_hdf5(e->parameter_file, h_grp, 0);
  H5Gclose(h_grp);

  /* Print the system of Units used in the spashot */
  io_write_unit_system(h_file, snapshot_units, "Units");

  /* Print the system of Units used internally */
  io_write_unit_system(h_file, internal_units, "InternalCodeUnits");

  /* Loop over all particle types */
  for (int ptype = 0; ptype < swift_type_count; ptype++) {

    /* Don't do anything if no particle of this kind */
    if (N_total[ptype] == 0) continue;

    if (iproducexmf) {
      /* Add the global information for that particle type to
       * the XMF meta-file */
      xmf_write_groupheader(xmfFile, fileName, /*distributed=*/0, N_total[ptype],
                          (enum part_type)ptype);
    }

    /* Create the particle group in the file */
    char partTypeGroupName[PARTICLE_GROUP_BUFFER_SIZE];
    snprintf(partTypeGroupName, PARTICLE_GROUP_BUFFER_SIZE, "/PartType%d",
             ptype);
    h_grp = H5Gcreate(h_file, partTypeGroupName, H5P_DEFAULT, H5P_DEFAULT,
                      H5P_DEFAULT);
    if (h_grp < 0)
      error("Error while opening particle group %s.", partTypeGroupName);

    /* Close particle group */
    H5Gclose(h_grp);

    if (iproducexmf){
      /* Close this particle group in the XMF file as well */
      xmf_write_groupfooter(xmfFile, (enum part_type)ptype);
    }
  }

  if (iproducexmf) {
      /* Write LXMF file descriptor */
      xmf_write_outputfooter(xmfFile, output_count, e->time);
  }

  /* Close the file for now */
  H5Fclose(h_file);
}

#if defined(WITH_MPI) && defined(HAVE_PARALLEL_HDF5)
/**
 * @brief Write dark matter density and velocity grids.
 * as a separate file.
 *
 * @param e The #engine.
 * @param baseName file name
 * @param internal_units The #unit_system used internally.
 * @param snapshot_units The #unit_system used in the snapshots.
 */
void write_grids_parallel(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units,
                           int mpi_rank, int mpi_size, MPI_Comm comm,
                           MPI_Info info) {
  const struct gpart* gparts = e->s->gparts;
  //const struct part* parts = e->s->parts;
  //const struct xpart* xparts = e->s->xparts;
  //const struct spart* sparts = e->s->sparts;
  //const struct bpart* bparts = e->s->bparts;
  //struct swift_params* params = e->parameter_file;
  //const int with_cosmology = e->policy & engine_policy_cosmology;
  //const int with_cooling = e->policy & engine_policy_cooling;
  //const int with_temperature = e->policy & engine_policy_temperature;
  //const int with_fof = e->policy & engine_policy_fof;
  const int with_DM_background = e->s->with_DM_background;

  /* Number of particles currently in the arrays */
  const size_t Ntot = e->s->nr_gparts;
  //const size_t Ngas = e->s->nr_parts;
  //const size_t Nstars = e->s->nr_sparts;
  //const size_t Nblackholes = e->s->nr_bparts;
  //const size_t Nbaryons = Ngas + Nstars + Nblackholes;
  //const size_t Ndm = Ntot > 0 ? Ntot - Nbaryons : 0;

  size_t Ndm_background = 0;
  if (with_DM_background) {
    Ndm_background = io_count_dm_background_gparts(gparts, Ntot);
  }

  /* Number of particles that we will write */
  const size_t Ntot_written =
      e->s->nr_gparts - e->s->nr_inhibited_gparts - e->s->nr_extra_gparts;
  const size_t Ngas_written =
      e->s->nr_parts - e->s->nr_inhibited_parts - e->s->nr_extra_parts;
  const size_t Nstars_written =
      e->s->nr_sparts - e->s->nr_inhibited_sparts - e->s->nr_extra_sparts;
  const size_t Nblackholes_written =
      e->s->nr_bparts - e->s->nr_inhibited_bparts - e->s->nr_extra_bparts;
  const size_t Nbaryons_written =
      Ngas_written + Nstars_written + Nblackholes_written;
  const size_t Ndm_written =
      Ntot_written > 0 ? Ntot_written - Nbaryons_written - Ndm_background : 0;

  /* Compute offset in the file and total number of particles */
  size_t N[swift_type_count] = {Ngas_written,   Ndm_written,
                                Ndm_background, 0,
                                Nstars_written, Nblackholes_written};
  long long N_total[swift_type_count] = {0};
  long long offset[swift_type_count] = {0};
  MPI_Exscan(N, offset, swift_type_count, MPI_LONG_LONG_INT, MPI_SUM, comm);
  for (int ptype = 0; ptype < swift_type_count; ++ptype)
    N_total[ptype] = offset[ptype] + N[ptype];

  /* The last rank now has the correct N_total. Let's
   * broadcast from there */
  MPI_Bcast(N_total, 6, MPI_LONG_LONG_INT, mpi_size - 1, comm);

  /* Now everybody konws its offset and the total number of
   * particles of each type */

  /* Rank 0 prepares the file */
  if (mpi_rank == 0)
    prepare_density_grids_file(e, baseName, N_total,
        internal_units, snapshot_units,
        e->density_grids_output_count, true
    );

  MPI_Barrier(MPI_COMM_WORLD);

  /* HDF5 File name */
  char fileName[FILENAME_BUFFER_SIZE];
  snprintf(fileName, FILENAME_BUFFER_SIZE, "%s_%04i.hdf5", baseName,
             e->density_grids_output_count);

  /* Prepare some file-access properties */
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

  /* Set some MPI-IO parameters */
  // MPI_Info_set(info, "IBM_largeblock_io", "true");
  MPI_Info_set(info, "romio_cb_write", "enable");
  MPI_Info_set(info, "romio_ds_write", "disable");

  /* Activate parallel i/o */
  hid_t h_err = H5Pset_fapl_mpio(plist_id, comm, info);
  if (h_err < 0) error("Error setting parallel i/o");

  /* Align on 4k pages. */
  h_err = H5Pset_alignment(plist_id, 1024, 4096);
  if (h_err < 0) error("Error setting Hdf5 alignment");

  /* Disable meta-data cache eviction */
  H5AC_cache_config_t mdc_config;
  mdc_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
  h_err = H5Pget_mdc_config(plist_id, &mdc_config);
  if (h_err < 0) error("Error getting the MDC config");

  mdc_config.evictions_enabled = 0; /* false */
  mdc_config.incr_mode = H5C_incr__off;
  mdc_config.decr_mode = H5C_decr__off;
  mdc_config.flash_incr_mode = H5C_flash_incr__off;
  h_err = H5Pset_mdc_config(plist_id, &mdc_config);
  if (h_err < 0) error("Error setting the MDC config");

/* Use parallel meta-data writes */
#if H5_VERSION_GE(1, 10, 0)
  h_err = H5Pset_all_coll_metadata_ops(plist_id, 1);
  if (h_err < 0) error("Error setting collective meta-data on all ops");
    // h_err = H5Pset_coll_metadata_write(plist_id, 1);
    // if (h_err < 0) error("Error setting collective meta-data writes");
#endif

  /* Open HDF5 file with the chosen parameters */
  hid_t h_file = H5Fopen(fileName, H5F_ACC_RDWR, plist_id);
  if (h_file < 0) error("Error while opening file '%s'.", fileName);

  /* Tell the user if a conversion will be needed */
  if (e->verbose && mpi_rank == 0) {
    if (units_are_equal(snapshot_units, internal_units)) {

      message("Snapshot and internal units match. No conversion needed.");

    } else {

      message("Conversion needed from:");
      message("(Snapshot) Unit system: U_M =      %e g.",
              snapshot_units->UnitMass_in_cgs);
      message("(Snapshot) Unit system: U_L =      %e cm.",
              snapshot_units->UnitLength_in_cgs);
      message("(Snapshot) Unit system: U_t =      %e s.",
              snapshot_units->UnitTime_in_cgs);
      message("(Snapshot) Unit system: U_I =      %e A.",
              snapshot_units->UnitCurrent_in_cgs);
      message("(Snapshot) Unit system: U_T =      %e K.",
              snapshot_units->UnitTemperature_in_cgs);
      message("to:");
      message("(internal) Unit system: U_M = %e g.",
              internal_units->UnitMass_in_cgs);
      message("(internal) Unit system: U_L = %e cm.",
              internal_units->UnitLength_in_cgs);
      message("(internal) Unit system: U_t = %e s.",
              internal_units->UnitTime_in_cgs);
      message("(internal) Unit system: U_I = %e A.",
              internal_units->UnitCurrent_in_cgs);
      message("(internal) Unit system: U_T = %e K.",
              internal_units->UnitTemperature_in_cgs);
    }
  }

  // darkmatter_write_density_grids_outputs(e, Ndm_written, h_file, internal_units, snapshot_units);
  darkmatter_write_grids(e, Ndm_written, h_file,
      internal_units, snapshot_units,
      e->density_grids_grid_dim, e->density_grids_grid_method
  );

  /* Close particle group */
  //H5Gclose(h_grp);

  /* Close property descriptor */
  //H5Pclose(plist_id);

  /* Close file */
  H5Fclose(h_file);

  e->density_grids_output_count++;
}

/**
 * @brief Write dark matter density and velocity grids.
 * as a separate file.
 *
 * @param e The #engine.
 * @param baseName file name
 * @param internal_units The #unit_system used internally.
 * @param snapshot_units The #unit_system used in the snapshots.
 */
void write_stf_grids_parallel(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units,
                           int mpi_rank, int mpi_size, MPI_Comm comm,
                           MPI_Info info) {
  const struct gpart* gparts = e->s->gparts;
  const int with_DM_background = e->s->with_DM_background;

  /* Number of particles currently in the arrays */
  const size_t Ntot = e->s->nr_gparts;

  size_t Ndm_background = 0;
  if (with_DM_background) {
    Ndm_background = io_count_dm_background_gparts(gparts, Ntot);
  }

  /* Number of particles that we will write */
  const size_t Ntot_written =
      e->s->nr_gparts - e->s->nr_inhibited_gparts - e->s->nr_extra_gparts;
  const size_t Ngas_written =
      e->s->nr_parts - e->s->nr_inhibited_parts - e->s->nr_extra_parts;
  const size_t Nstars_written =
      e->s->nr_sparts - e->s->nr_inhibited_sparts - e->s->nr_extra_sparts;
  const size_t Nblackholes_written =
      e->s->nr_bparts - e->s->nr_inhibited_bparts - e->s->nr_extra_bparts;
  const size_t Nbaryons_written =
      Ngas_written + Nstars_written + Nblackholes_written;
  const size_t Ndm_written =
      Ntot_written > 0 ? Ntot_written - Nbaryons_written - Ndm_background : 0;

  /* Compute offset in the file and total number of particles */
  size_t N[swift_type_count] = {Ngas_written,   Ndm_written,
                                Ndm_background, 0,
                                Nstars_written, Nblackholes_written};
  long long N_total[swift_type_count] = {0};
  long long offset[swift_type_count] = {0};
  MPI_Exscan(N, offset, swift_type_count, MPI_LONG_LONG_INT, MPI_SUM, comm);
  for (int ptype = 0; ptype < swift_type_count; ++ptype)
    N_total[ptype] = offset[ptype] + N[ptype];

  /* The last rank now has the correct N_total. Let's
   * broadcast from there */
  MPI_Bcast(N_total, 6, MPI_LONG_LONG_INT, mpi_size - 1, comm);

  /* Now everybody konws its offset and the total number of
   * particles of each type */

  /* Rank 0 prepares the file */
  if (mpi_rank == 0)
    prepare_density_grids_file(e, baseName, N_total,
        internal_units, snapshot_units,
        e->stf_output_count-1, false);

  MPI_Barrier(MPI_COMM_WORLD);

  /* HDF5 File name */
  char fileName[FILENAME_BUFFER_SIZE];
  snprintf(fileName, FILENAME_BUFFER_SIZE, "%s_%04i.hdf5", baseName,
             e->stf_output_count-1);

  /* Prepare some file-access properties */
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

  /* Set some MPI-IO parameters */
  // MPI_Info_set(info, "IBM_largeblock_io", "true");
  MPI_Info_set(info, "romio_cb_write", "enable");
  MPI_Info_set(info, "romio_ds_write", "disable");

  /* Activate parallel i/o */
  hid_t h_err = H5Pset_fapl_mpio(plist_id, comm, info);
  if (h_err < 0) error("Error setting parallel i/o");

  /* Align on 4k pages. */
  h_err = H5Pset_alignment(plist_id, 1024, 4096);
  if (h_err < 0) error("Error setting Hdf5 alignment");

  /* Disable meta-data cache eviction */
  H5AC_cache_config_t mdc_config;
  mdc_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
  h_err = H5Pget_mdc_config(plist_id, &mdc_config);
  if (h_err < 0) error("Error getting the MDC config");

  mdc_config.evictions_enabled = 0; /* false */
  mdc_config.incr_mode = H5C_incr__off;
  mdc_config.decr_mode = H5C_decr__off;
  mdc_config.flash_incr_mode = H5C_flash_incr__off;
  h_err = H5Pset_mdc_config(plist_id, &mdc_config);
  if (h_err < 0) error("Error setting the MDC config");

/* Use parallel meta-data writes */
#if H5_VERSION_GE(1, 10, 0)
  h_err = H5Pset_all_coll_metadata_ops(plist_id, 1);
  if (h_err < 0) error("Error setting collective meta-data on all ops");
    // h_err = H5Pset_coll_metadata_write(plist_id, 1);
    // if (h_err < 0) error("Error setting collective meta-data writes");
#endif

  /* Open HDF5 file with the chosen parameters */
  hid_t h_file = H5Fopen(fileName, H5F_ACC_RDWR, plist_id);
  if (h_file < 0) error("Error while opening file '%s'.", fileName);

  darkmatter_write_grids(e, Ndm_written, h_file,
      internal_units, snapshot_units,
      e->stf_density_grids_grid_dim, e->stf_density_grids_grid_method
  );

  /* Close file */
  H5Fclose(h_file);

}

#elif defined(WITH_MPI) && !defined(HAVE_PARALLEL_HDF5)
void write_grids_serial(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units,
                           int mpi_rank, int mpi_size, MPI_Comm comm,
                           MPI_Info info) {
}
void write_stf_grids_serial(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units,
                           int mpi_rank, int mpi_size, MPI_Comm comm,
                           MPI_Info info) {
}
#endif

void write_grids_single(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units) {
}
void write_stf_grids_single(struct engine* e, const char* baseName,
                           const struct unit_system* internal_units,
                           const struct unit_system* snapshot_units) {
}
#endif
