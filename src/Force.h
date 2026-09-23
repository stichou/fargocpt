#pragma once

#include "types.h"
#include "data.h"

Pair ComputeDiskOnPlanetAccel(t_data &data, const unsigned nb);
double compute_smoothing(t_data &data, const int n_radial,
			 const int n_azimuthal, const unsigned nb);

/**
 * Norm of the body force per unit length (in the dx/s_dist direction),
 * shared between CalculateAccelOnGas and ComputeDiskOnPlanetAccel.
 * mass is the mass of the body exerting the force; the sign of the
 * resulting force has to be applied by the caller.
 */
double compute_body_force_norm(t_data &data, const int n_radial,
			       const int n_azimuthal, const unsigned nb,
			       const double mass, const double dist_2,
			       const double s_dist);
double compute_smoothing_iso_planet(const double Rp);

