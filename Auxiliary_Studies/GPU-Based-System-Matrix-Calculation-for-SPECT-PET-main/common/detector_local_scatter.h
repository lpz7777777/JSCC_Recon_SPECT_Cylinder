#pragma once

#include <cmath>

#include "energy_window.h"
#include "../physics_data/nist_xcom_materials_1_1000keV.h"

struct LocalSecondInteractionPartition
{
    double escape;
    double photoelectric;
    double compton;
};

struct DetectorLocalScatterResponse
{
    double recoil_windowed;
    double self_photoelectric_windowed;
    double escape_probability;
    double second_photoelectric_probability;
    double second_compton_probability;
};

inline double local_scatter_xcom_interpolate(
    const float* table,
    int material_id,
    double energy_keV)
{
    if (material_id < 0) return 0.0;
    if (energy_keV <= kXcomEnergyMinKeV)
        return table[material_id * kXcomEnergyCount];
    if (energy_keV >= kXcomEnergyMaxKeV)
        return table[(material_id + 1) * kXcomEnergyCount - 1];

    const int lower_energy = static_cast<int>(std::floor(energy_keV));
    const double fraction = energy_keV - lower_energy;
    const int lower_index = material_id * kXcomEnergyCount
        + lower_energy - kXcomEnergyMinKeV;
    return table[lower_index]
        + fraction * (table[lower_index + 1] - table[lower_index]);
}

inline LocalSecondInteractionPartition local_second_interaction_partition(
    double mu_photoelectric,
    double mu_compton,
    double path_mm)
{
    LocalSecondInteractionPartition result = {1.0, 0.0, 0.0};
    const double mu_total = mu_photoelectric + mu_compton;
    if (!(mu_total > 0.0) || !(path_mm > 0.0)) return result;

    result.escape = std::exp(-mu_total * path_mm);
    const double interaction = 1.0 - result.escape;
    result.photoelectric = interaction * mu_photoelectric / mu_total;
    result.compton = interaction * mu_compton / mu_total;
    return result;
}

inline double detector_center_exit_distance(
    double direction_x,
    double direction_y,
    double direction_z,
    double width_mm,
    double thickness_mm,
    double height_mm)
{
    const double half_x = 0.5 * width_mm;
    const double half_y = 0.5 * thickness_mm;
    const double half_z = 0.5 * height_mm;
    double distance = INFINITY;
    const double epsilon = 1e-14;
    if (std::fabs(direction_x) > epsilon)
        distance = std::fmin(distance, half_x / std::fabs(direction_x));
    if (std::fabs(direction_y) > epsilon)
        distance = std::fmin(distance, half_y / std::fabs(direction_y));
    if (std::fabs(direction_z) > epsilon)
        distance = std::fmin(distance, half_z / std::fabs(direction_z));
    return std::isfinite(distance) ? distance : 0.0;
}

inline double local_scatter_klein_nishina_weight(double cosine_theta, double energy_keV)
{
    const double alpha = energy_keV / 511.0;
    const double factor1 = alpha * (1.0 - cosine_theta);
    const double factor2 = 1.0 + cosine_theta * cosine_theta;
    return factor2 / ((1.0 + factor1) * (1.0 + factor1))
        * (1.0 + factor1 * factor1 / (factor2 * (1.0 + factor1)));
}

inline double local_scatter_gaussian_acceptance(
    double mean_energy_keV,
    double relative_fwhm_at_source,
    double source_energy_keV,
    double lower_window_keV,
    double upper_window_keV)
{
    if (!(mean_energy_keV > 0.0)) return 0.0;
    const double relative_fwhm = relative_fwhm_at_source
        * std::sqrt(source_energy_keV / mean_energy_keV);
    return gaussian_energy_window_acceptance(
        static_cast<float>(mean_energy_keV),
        static_cast<float>(relative_fwhm),
        static_cast<float>(lower_window_keV),
        static_cast<float>(upper_window_keV));
}

inline DetectorLocalScatterResponse integrate_detector_local_scatter_response(
    double incoming_x,
    double incoming_y,
    double incoming_z,
    double width_mm,
    double thickness_mm,
    double height_mm,
    int material_id,
    double source_energy_keV,
    double relative_fwhm_at_source,
    double lower_window_keV,
    double upper_window_keV,
    int cosine_samples,
    int azimuth_samples)
{
    DetectorLocalScatterResponse result = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (material_id < 0 || !(source_energy_keV > 0.0)
        || cosine_samples < 1 || azimuth_samples < 1)
        return result;

    const double norm = std::sqrt(incoming_x * incoming_x
        + incoming_y * incoming_y + incoming_z * incoming_z);
    if (!(norm > 0.0)) return result;
    incoming_x /= norm;
    incoming_y /= norm;
    incoming_z /= norm;

    double basis1_x;
    double basis1_y;
    double basis1_z;
    if (std::fabs(incoming_z) < 0.9)
    {
        basis1_x = incoming_y;
        basis1_y = -incoming_x;
        basis1_z = 0.0;
    }
    else
    {
        basis1_x = 0.0;
        basis1_y = incoming_z;
        basis1_z = -incoming_y;
    }
    const double basis1_norm = std::sqrt(basis1_x * basis1_x
        + basis1_y * basis1_y + basis1_z * basis1_z);
    basis1_x /= basis1_norm;
    basis1_y /= basis1_norm;
    basis1_z /= basis1_norm;
    const double basis2_x = incoming_y * basis1_z - incoming_z * basis1_y;
    const double basis2_y = incoming_z * basis1_x - incoming_x * basis1_z;
    const double basis2_z = incoming_x * basis1_y - incoming_y * basis1_x;

    const double full_energy_acceptance = local_scatter_gaussian_acceptance(
        source_energy_keV, relative_fwhm_at_source, source_energy_keV,
        lower_window_keV, upper_window_keV);
    const double pi = 3.14159265358979323846;
    double weight_sum = 0.0;

    for (int cosine_index = 0; cosine_index < cosine_samples; ++cosine_index)
    {
        const double cosine_theta = -1.0
            + 2.0 * (cosine_index + 0.5) / cosine_samples;
        const double sine_theta = std::sqrt(std::fmax(0.0,
            1.0 - cosine_theta * cosine_theta));
        const double scattered_energy = source_energy_keV
            / (1.0 + source_energy_keV / 511.0 * (1.0 - cosine_theta));
        const double recoil_energy = source_energy_keV - scattered_energy;
        const double angular_weight = local_scatter_klein_nishina_weight(
            cosine_theta, source_energy_keV);
        const double recoil_acceptance = local_scatter_gaussian_acceptance(
            recoil_energy, relative_fwhm_at_source, source_energy_keV,
            lower_window_keV, upper_window_keV);
        const double mu_photoelectric = local_scatter_xcom_interpolate(
            kXcomMuPhotoelectric, material_id, scattered_energy);
        const double mu_compton = local_scatter_xcom_interpolate(
            kXcomMuCompton, material_id, scattered_energy);

        for (int azimuth_index = 0; azimuth_index < azimuth_samples; ++azimuth_index)
        {
            const double azimuth = 2.0 * pi * (azimuth_index + 0.5)
                / azimuth_samples;
            const double transverse1 = sine_theta * std::cos(azimuth);
            const double transverse2 = sine_theta * std::sin(azimuth);
            const double direction_x = cosine_theta * incoming_x
                + transverse1 * basis1_x + transverse2 * basis2_x;
            const double direction_y = cosine_theta * incoming_y
                + transverse1 * basis1_y + transverse2 * basis2_y;
            const double direction_z = cosine_theta * incoming_z
                + transverse1 * basis1_z + transverse2 * basis2_z;
            const double path = detector_center_exit_distance(
                direction_x, direction_y, direction_z,
                width_mm, thickness_mm, height_mm);
            const LocalSecondInteractionPartition partition
                = local_second_interaction_partition(
                    mu_photoelectric, mu_compton, path);

            result.recoil_windowed += angular_weight
                * partition.escape * recoil_acceptance;
            result.self_photoelectric_windowed += angular_weight
                * partition.photoelectric * full_energy_acceptance;
            result.escape_probability += angular_weight * partition.escape;
            result.second_photoelectric_probability += angular_weight
                * partition.photoelectric;
            result.second_compton_probability += angular_weight
                * partition.compton;
            weight_sum += angular_weight;
        }
    }

    if (weight_sum > 0.0)
    {
        result.recoil_windowed /= weight_sum;
        result.self_photoelectric_windowed /= weight_sum;
        result.escape_probability /= weight_sum;
        result.second_photoelectric_probability /= weight_sum;
        result.second_compton_probability /= weight_sum;
    }
    return result;
}
