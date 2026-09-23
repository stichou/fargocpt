/**
	\file Force.cpp

	Contains the function used to evaluate the %force due to disk, and the
function that writes the 'tqwk' log files. Although the planet mass is given as
an argument to ComputeForce(), this mass is used only to specify the distance
cutoff in the case of the Hill sphere avoidance. The force returned is a
specific force. It has therefore the dimension of an acceleration (LT^-2).
**/

#include <cmath>
#include <float.h>
#include <stdio.h>

#include "Force.h"
#include "constants.h"
#include "global.h"
#include "parameters.h"

/**
	Computes the acceleration due to the disk on an object at position (x,y)
*/
Pair ComputeDiskOnPlanetAccel(t_data &data, const unsigned int nb)
{

	t_planetary_system & psys = data.get_planetary_system();
	t_planet & planet = psys.get_planet(nb);
	const double x = planet.get_x();
	const double y = planet.get_y();
	const double a = planet.get_r();

	const double klahr_smoothing_constant = planet.get_cubic_smoothing_factor();


    Pair acceleration;
    double localaccel[4] = {0., 0., 0., 0.};
    double globalaccel[4] = {0., 0., 0., 0.};
    double axi, ayi, axo, ayo;

    const unsigned int ns = data[t_data::SIGMA].Nsec;
	const auto & sigma = data[t_data::SIGMA];
	const auto & sigma1d = data[t_data::SIGMA_1D];
    const double *cell_center_x = CellCenterX->Field;
    const double *cell_center_y = CellCenterY->Field;
    axi = ayi = axo = ayo = 0.0;

	#pragma omp parallel for collapse(2) reduction(+ : axi, ayi, axo, ayo)
    for (unsigned int n_rad = radial_first_active; n_rad < radial_active_size;
	 ++n_rad) {
	for (unsigned int n_az = 0; n_az < ns; ++n_az) {
	    // calculate smoothing length if dependend on radius
	    // i.e. for thickness smoothing with scale height at cell location
		double smooth = compute_smoothing(data, n_rad, n_az, nb);


		// Phi = GMm / r_sm
		// r_sm = sqrt(r**2 + (eps * H)**2)

	    const int cell_id = n_az + n_rad * ns;
	    const double xc = cell_center_x[cell_id];
	    const double yc = cell_center_y[cell_id];

		double cell_sigma = sigma(n_rad, n_az);
		if (parameters::correct_disk_selfgravity) {
			cell_sigma -= sigma1d(n_rad);
		}
	    const double cellmass = Surf[n_rad] * cell_sigma;

	    const double dx = xc - x;
	    const double dy = yc - y;
	    const double dist_2 = std::pow(dx, 2) + std::pow(dy, 2);
		const double dist_sm_2 = dist_2 + std::pow(smooth, 2);
		const double dist_sm = std::sqrt(dist_sm_2);
		const double s_dist = std::sqrt(dist_2);

		// Choose between smoothing and bessel methods for force
		// calculation, mirroring CalculateAccelOnGas
		const double force_norm =
		    compute_body_force_norm(data, n_rad, n_az, nb, cellmass,
					     dist_2, s_dist);

	    // just to be consistent with the force the gas feels from the
	    // planets
	    double smooth_factor_klahr = 1.0;
	    if (klahr_smoothing_constant > 0.0) {
		/// scale height is reduced by the planets and can cause the
		/// epsilon smoothing be not sufficient for numerical stability.
		/// Thus we add the gravitational potential smoothing proposed
		/// by Klahr & Kley 2005; but the derivative of it, since we
		/// apply it directly on the force
			const double l1 = planet.get_dimensionless_roche_radius() *
				planet.get_distance_to_primary();
			const double r_sm = l1 * klahr_smoothing_constant;

			if (dist_sm < r_sm) {
			smooth_factor_klahr =
				-(3.0 * std::pow(dist_sm / r_sm, 4.0) -
				  4.0 * std::pow(dist_sm / r_sm, 3.0));
		    }
		}

	    if (Rmed[n_rad] < a) {
		axi += dx / s_dist * force_norm * smooth_factor_klahr;
		ayi += dy / s_dist * force_norm * smooth_factor_klahr;
	    } else {
		axo += dx / s_dist * force_norm * smooth_factor_klahr;
		ayo += dy / s_dist * force_norm * smooth_factor_klahr;
	    }
	}
    }

    localaccel[0] = axi;
    localaccel[1] = ayi;
    localaccel[2] = axo;
    localaccel[3] = ayo;
    MPI_Allreduce(&localaccel, &globalaccel, 4, MPI_DOUBLE, MPI_SUM,
		  MPI_COMM_WORLD);

    acceleration.x = globalaccel[0] + globalaccel[2];
    acceleration.y = globalaccel[1] + globalaccel[3];

    return acceleration;
}

inline static double compute_smoothing_scaleheight(t_data &data, const int n_radial,
			 const int n_azimuthal)
{
    const double scale_height =
	data[t_data::SCALE_HEIGHT](n_radial, n_azimuthal);
    const double smooth = parameters::thickness_smoothing * scale_height;
    return smooth;
}

inline static double compute_smoothing_iso_planet(t_data &data, const double nb)
{
	t_planetary_system & psys = data.get_planetary_system();
	t_planet & planet = psys.get_planet(nb);
	const double a = planet.get_r();
	const double h0 = parameters::aspectratio_ref;
	const double beta = parameters::flaring_index;
    const double scale_height = h0*std::pow(a, 1+beta);
    const double smooth = parameters::thickness_smoothing * scale_height;
    return smooth;
}

/**
   Computes the norm of the body force per unit length, shared between
   CalculateAccelOnGas and ComputeDiskOnPlanetAccel. The direction of
   the force (dx/s_dist, dy/s_dist) and its sign have to be applied by
   the caller.
 */
double compute_body_force_norm(t_data &data, const int n_radial,
			       const int n_azimuthal, const unsigned nb,
			       const double mass, const double dist_2,
			       const double s_dist)
{
	double force_norm = 0.0;
	if (s_dist == 0.0) {
		return 0.0;
	}

	switch (parameters::body_force_method) {
	case parameters::body_force_smoothing: {
		const double smooth =
		    compute_smoothing(data, n_radial, n_azimuthal, nb);
		const double dist_2_sm = dist_2 + std::pow(smooth, 2);
		const double dist_sm = std::sqrt(dist_2_sm);
		const double inv_dist_3_sm = 1.0 / (dist_sm * dist_2_sm);
		force_norm = constants::G * mass * s_dist * inv_dist_3_sm;
		break;
	}
	case parameters::body_force_bessel_exact: {
		const double scale_height =
		    data[t_data::SCALE_HEIGHT](n_radial, n_azimuthal);
		const double X_aux =
		    dist_2 / (4.0 * scale_height * scale_height); // c^2

		double bessel_term;
		if (X_aux > 60.0) {
			// Taylor expansion at infinity to avoid exp overflow
			// (See Sect. 5.3 of Rendon Restrepo et al. 2025)
			bessel_term =
			    X_aux * 0.5 * std::pow(M_PI / 2., 0.5) *
			    (std::pow(X_aux, -1.5) - 3. / 8. * std::pow(X_aux, -2.5) +
			     45. / 128. * std::pow(X_aux, -3.5));
		} else {
			bessel_term = X_aux * std::exp(X_aux) *
			    (std::cyl_bessel_kl(1., X_aux) -
			     std::cyl_bessel_kl(0., X_aux));
		}
		force_norm = constants::G * mass / s_dist / scale_height *
			     std::pow(2.0 / M_PI, 0.5) * bessel_term;
		break;
	}
	case parameters::body_force_bessel_approx: {
		/* Approximation of the Bessel function formulation
		 * The use of Bessel functions is expensive. Therefore we
		 * use an approximation with a space varying smoothing
		 * length.
		 */
		const double scale_height =
		    data[t_data::SCALE_HEIGHT](n_radial, n_azimuthal);
		const double d_norm = s_dist / scale_height;
		const double beta_sg = 0.06427627;
		const double q_sg = 1.14735482;
		const double eps_p0 = std::pow(M_PI / 2.0, 1 / 6.0);

		const double svsl = (1 -
				     std::exp(-eps_p0 *
					      std::pow(d_norm, 2.0 / 3.0) -
					      beta_sg *
						  std::pow(d_norm, q_sg)));
		force_norm = constants::G * mass / s_dist / scale_height *
		     //std::pow(2.0 / M_PI, 0.5) *
		     //std::pow(M_PI / 2.0, 0.5) *
		     std::pow(d_norm, 2.0) *
		     std::pow(std::pow(d_norm, 2.0) +
				  std::pow(svsl, 2.0),
			      -1.5);
		break;
	}
	}
	return force_norm;
}

double compute_smoothing(t_data &data, const int n_radial,
			 const int n_azimuthal, const unsigned int nb)
{
	double rv;
	if (parameters::compatibility_no_star_smoothing && nb == 0) {
		return 0;
	}

	if (parameters::compatibility_smoothing_planetloc) {
		rv = compute_smoothing_iso_planet(data, nb);
	} else {
		rv = compute_smoothing_scaleheight(data, n_radial, n_azimuthal);
	}
	return rv;
}
