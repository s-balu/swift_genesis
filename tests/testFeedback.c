/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (C) 2015 Matthieu Schaller (matthieu.schaller@durham.ac.uk).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "swift.h"

int main(int argc, char *argv[]) {
  /* Declare relevant structs */
  struct swift_params *params = malloc(sizeof(struct swift_params));
  struct unit_system us;
  struct chemistry_global_data chem_data;
  struct part p;
  struct xpart xp;
  struct spart sp;
  struct phys_const phys_const;
  struct cosmology cosmo;
  struct hydro_props hydro_properties;
  struct stars_props stars_properties;
  char *parametersFileName = "./testFeedback.yml";

  /* Read the parameter file */
  if (params == NULL) error("Error allocating memory for the parameter file.");
  message("Reading runtime parameters from file '%s'", parametersFileName);
  parser_read_file(parametersFileName, params);

  /* Init units */
  units_init_from_params(&us, params, "InternalUnitSystem");
  phys_const_init(&us, params, &phys_const);

  /* Init chemistry */
  chemistry_init(params, &us, &phys_const, &chem_data);
  chemistry_first_init_part(&phys_const, &us, &cosmo, &chem_data, &p, &xp);
  chemistry_print(&chem_data);

  /* Init cosmology */
  cosmology_init(params, &us, &phys_const, &cosmo);
  cosmology_print(&cosmo);

  /* Init hydro properties */
  hydro_props_init(&hydro_properties, &phys_const, &us, params);

  /* Init star properties */
  stars_props_init(&stars_properties, &phys_const, &us, params, &hydro_properties, &cosmo);

  /* Read yield tables */
  stars_evolve_init(params, &stars_properties);

  /* Init spart */
  stars_first_init_spart(&sp);

  sp.mass_init = 4.706273e-5;

  for (int i = 0; i < chemistry_element_count; i++) sp.metals_released[i] = 0.f;
  sp.chemistry_data.metal_mass_fraction_from_AGB = 0.f;
  sp.to_distribute.mass = 0.f;

  FILE *AGB_output;
  char fname[25] = "test_feedback_AGB.txt";
  if (!(AGB_output = fopen(fname, "w"))) {
    error("error in opening file '%s'\n", fname);
  }
  fprintf(AGB_output,
          "# time[Gyr] | total mass | metal mass: total | H | He | C | N  | O  "
          "| Ne | Mg | Si | Fe | per solar mass\n");

  float Gyr_to_s = 3.154e16;
  float dt = 0.1 * Gyr_to_s / units_cgs_conversion_factor(&us,UNIT_CONV_TIME);
  float max_age = 13.f * Gyr_to_s / units_cgs_conversion_factor(&us,UNIT_CONV_TIME);
  for (float age = 0; age <= max_age; age += dt) {
    compute_stellar_evolution(&stars_properties, &sp, &us, age, dt);
    float age_Gyr = age * units_cgs_conversion_factor(&us,UNIT_CONV_TIME) / Gyr_to_s;
    fprintf(AGB_output, "%f %e %e ", age_Gyr, sp.to_distribute.mass / sp.mass_init, sp.chemistry_data.metal_mass_fraction_from_AGB / sp.mass_init);
    for (int i = 0; i < chemistry_element_count; i++)
      fprintf(AGB_output, "%e ", sp.metals_released[i]);
    fprintf(AGB_output, "\n");
  }
  return 0;
}
