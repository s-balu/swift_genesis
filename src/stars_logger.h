/*******************************************************************************
 * This file is part of SWIFT.
 * Coypright (c) 2020 Loic Hausammann (loic.hausammann@epfl.ch)
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
#ifndef SWIFT_STARS_LOGGER_H
#define SWIFT_STARS_LOGGER_H

/* Include config */
#include "../config.h"

/* Local includes */
#include "./const.h"
#include "align.h"
#include "logger.h"
#include "part_type.h"
#include "timeline.h"

/* Load the correct star type */
#if defined(FEEDBACK_CONST)
#error TODO
#elif defined(STARS_NONE)
#include "./stars/Default/stars_logger.h"
#elif defined(STARS_EAGLE)
#error TODO
#elif defined(STARS_GEAR)
#error TODO
#else
#error "Invalid choice of star model"
#endif

#endif /* SWIFT_STARS_LOGGER_H */
