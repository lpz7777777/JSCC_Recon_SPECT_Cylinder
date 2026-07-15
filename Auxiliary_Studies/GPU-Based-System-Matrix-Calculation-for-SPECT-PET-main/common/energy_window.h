#pragma once

#include <cmath>

inline void resolve_energy_window(
    const float* physics,
    float relative_fwhm,
    float* lower_keV,
    float* upper_keV)
{
    const float source_energy_keV = physics[7];
    if (static_cast<int>(std::floor(physics[4] + 0.5f)) > 0)
    {
        *lower_keV = physics[5];
        *upper_keV = physics[6];
    }
    else
    {
        *lower_keV = (1.0f - relative_fwhm / 2.0f) * source_energy_keV;
        *upper_keV = (1.0f + relative_fwhm / 2.0f) * source_energy_keV;
    }
}

inline float gaussian_energy_window_acceptance(
    float mean_energy_keV,
    float relative_fwhm,
    float lower_keV,
    float upper_keV)
{
    if (!(upper_keV > lower_keV) || !(mean_energy_keV > 0.0f))
        return 0.0f;

    if (!(relative_fwhm > 0.0f))
        return (mean_energy_keV >= lower_keV && mean_energy_keV <= upper_keV) ? 1.0f : 0.0f;

    const float sigma_keV = relative_fwhm * mean_energy_keV / 2.35482f;
    const float denominator = sigma_keV * std::sqrt(2.0f);
    const float z_lower = (lower_keV - mean_energy_keV) / denominator;
    const float z_upper = (upper_keV - mean_energy_keV) / denominator;
    float probability = 0.5f * (std::erf(z_upper) - std::erf(z_lower));
    if (probability < 0.0f) probability = 0.0f;
    if (probability > 1.0f) probability = 1.0f;
    return probability;
}

inline float photopeak_energy_window_acceptance(
    const float* physics,
    const float* detector_record)
{
    float lower_keV = 0.0f;
    float upper_keV = 0.0f;
    const float relative_fwhm = detector_record[9];
    resolve_energy_window(physics, relative_fwhm, &lower_keV, &upper_keV);
    return gaussian_energy_window_acceptance(
        physics[7], relative_fwhm, lower_keV, upper_keV);
}
