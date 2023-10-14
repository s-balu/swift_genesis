#ifndef SWIFT_DARKMATTER_WRITE_GRIDS_H
#define SWIFT_DARKMATTER_WRITE_GRIDS_H

/* Config parameters. */
#include "../config.h"
#include "engine.h"
#include "units.h"
#include "common_io.h"
#include "xmf.h"
#include <stdbool.h> 

void prepare_density_grids_file(struct engine* e, const char* baseName, long long N_total[6],
                  const struct unit_system* internal_units,
                  const struct unit_system* snapshot_units,
                  const int output_count,
                  const bool iproducexmf
              );
// void prepare_stf_density_grids_file(struct engine* e, const char* baseName, long long N_total[6],
//                 const struct unit_system* internal_units,
//                 const struct unit_system* snapshot_units);

void darkmatter_write_grids(struct engine* e, const size_t Npart,
                            const hid_t h_file,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            const int grid_dim,
                            const char grid_method[PARSER_MAX_LINE_SIZE]
                        );
// void darkmatter_write_grids(struct engine* e, const size_t Npart,
//                             const hid_t h_file,
//                             const struct unit_system* internal_units,
//                             const struct unit_system* snapshot_units);
// void darkmatter_write_density_grids_outputs(struct engine* e, const size_t Npart,
//                             const hid_t h_file,
//                             const struct unit_system* internal_units,
//                             const struct unit_system* snapshot_units);
// void darkmatter_stf_write_density_grids_outputs(struct engine* e, const size_t Npart,
//                             const hid_t h_file,
//                             const struct unit_system* internal_units,
//                             const struct unit_system* snapshot_units);

#if defined(HAVE_HDF5) && defined(WITH_MPI) && defined(HAVE_PARALLEL_HDF5)
void write_grids_parallel(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            int mpi_rank, int mpi_size, MPI_Comm comm,
                            MPI_Info info);
void write_stf_grids_parallel(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            int mpi_rank, int mpi_size, MPI_Comm comm,
                            MPI_Info info);
#elif defined(HAVE_HDF5) && defined(WITH_MPI)
void write_grids_serial(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            int mpi_rank, int mpi_size, MPI_Comm comm,
                            MPI_Info info);
void write_stf_grids_serial(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units,
                            int mpi_rank, int mpi_size, MPI_Comm comm,
                            MPI_Info info);
#endif
void write_grids_single(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units);
void write_stf_grids_single(struct engine* e, const char* baseName,
                            const struct unit_system* internal_units,
                            const struct unit_system* snapshot_units);


#endif /* SWIFT_DARKMATTER_WRITE_GRIDS_H */
