#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "../common/detector_local_scatter.h"
#include "../common/pe_v4_reference.h"

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void requireClose(
    double actual,
    double expected,
    double absolute_tolerance,
    double relative_tolerance,
    const char* message)
{
    const double error = std::fabs(actual - expected);
    const double allowed = absolute_tolerance
        + relative_tolerance * std::fmax(std::fabs(expected), 1e-30);
    if (error > allowed)
    {
        std::cerr << message << ": actual=" << actual
            << " expected=" << expected << " error=" << error
            << " allowed=" << allowed << std::endl;
        throw std::runtime_error(message);
    }
}

double centeredRectangleSolidAngle(double half_width, double half_height, double distance)
{
    return 4.0 * std::atan(
        half_width * half_height
        / (distance * std::sqrt(distance * distance
            + half_width * half_width + half_height * half_height)));
}
}

int main()
{
    const double pi = 3.14159265358979323846;

    requireClose(detector_segment_box_chord(
        -3.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        4.0, 6.0, 8.0), 4.0, 1e-14, 0.0,
        "axis-aligned segment chord");
    requireClose(detector_segment_box_chord(
        -3.0, -3.0, 0.0, 3.0, 3.0, 0.0,
        4.0, 4.0, 8.0), 4.0 * std::sqrt(2.0), 1e-13, 0.0,
        "diagonal segment chord");
    requireClose(detector_segment_box_chord(
        -3.0, 5.0, 0.0, 3.0, 5.0, 0.0,
        4.0, 4.0, 8.0), 0.0, 1e-14, 0.0,
        "missing segment chord");

    // Opaque centered box: only the front face is visible, so the integral
    // tends to the exact rectangular solid angle divided by 4*pi.
    const double width = 3.0;
    const double thickness = 3.0;
    const double height = 3.0;
    const double front_distance = 100.0;
    const PEV4ReferenceResult opaque = integrate_pe_v4_point_source_reference(
        0.0, -front_distance - thickness / 2.0, 0.0,
        width, thickness, height,
        100.0, 0.0, 64, 8);
    const double exact_solid_angle_fraction = centeredRectangleSolidAngle(
        width / 2.0, height / 2.0, front_distance) / (4.0 * pi);
    requireClose(opaque.photoelectric_probability,
        exact_solid_angle_fraction, 1e-12, 2e-6,
        "opaque-box rectangular solid angle");
    requireClose(opaque.first_interaction_probability,
        opaque.photoelectric_probability, 1e-15, 1e-12,
        "opaque all-photoelectric partition");
    const PEV4ReferenceResult opaque_halton = integrate_pe_v4_point_source_reference(
        0.0, -front_distance - thickness / 2.0, 0.0,
        width, thickness, height,
        100.0, 0.0, 256, 8, true);
    requireClose(opaque_halton.photoelectric_probability,
        exact_solid_angle_fraction, 1e-12, 2e-4,
        "Halton opaque-box rectangular solid angle");

    // Production uses N*N samples. The symmetric Halton construction must
    // preserve both in-face reflection symmetries even at the default N=16.
    const PEV4ReferenceResult halton_positive = integrate_pe_v4_point_source_reference(
        1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 16, 8, true);
    const PEV4ReferenceResult halton_reflect_x = integrate_pe_v4_point_source_reference(
        -1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 16, 8, true);
    const PEV4ReferenceResult halton_reflect_z = integrate_pe_v4_point_source_reference(
        1.1, -25.0, -0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 16, 8, true);
    requireClose(halton_positive.photoelectric_probability,
        halton_reflect_x.photoelectric_probability, 1e-18, 2e-13,
        "symmetric Halton x reflection");
    requireClose(halton_positive.photoelectric_probability,
        halton_reflect_z.photoelectric_probability, 1e-18, 2e-13,
        "symmetric Halton z reflection");

    // PE and Compton first-interaction branches must preserve their cross-section ratio.
    const PEV4ReferenceResult mixed = integrate_pe_v4_point_source_reference(
        0.4, -80.0, -0.7,
        4.0, 10.0, 6.0,
        0.3, 0.2, 32, 16);
    require(mixed.first_interaction_probability > 0.0,
        "mixed reference produced no interactions");
    requireClose(mixed.first_interaction_probability,
        mixed.photoelectric_probability + mixed.compton_probability,
        2e-16, 2e-13, "first-interaction probability closure");
    requireClose(mixed.photoelectric_probability
        / mixed.first_interaction_probability,
        0.6, 1e-14, 1e-13, "photoelectric branch ratio");
    requireClose(mixed.compton_probability
        / mixed.first_interaction_probability,
        0.4, 1e-14, 1e-13, "Compton branch ratio");

    // A normal parallel ray has one 10 mm chord. Its conditional depth mean
    // has a closed form for a truncated exponential.
    double parallel_weight = 0.0;
    double parallel_depth = 0.0;
    const double mu_total = 0.5;
    for_each_parallel_box_first_interaction_state(
        0.0, 1.0, 0.0,
        4.0, 10.0, 6.0,
        0.3, 0.2, 1, 8192,
        [&](const FirstInteractionState& state)
        {
            parallel_weight += state.interaction_weight;
            parallel_depth += state.depth_mm * state.interaction_weight;
        });
    parallel_depth /= parallel_weight;
    requireClose(parallel_depth,
        first_interaction_truncated_exponential_mean_depth(mu_total, 10.0),
        2e-7, 0.0, "truncated-exponential mean depth");

    // Face quadrature should converge monotonically for a finite attenuation case.
    const PEV4ReferenceResult face16 = integrate_pe_v4_point_source_reference(
        1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 16, 8);
    const PEV4ReferenceResult face32 = integrate_pe_v4_point_source_reference(
        1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 32, 8);
    const PEV4ReferenceResult face64 = integrate_pe_v4_point_source_reference(
        1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 64, 8);
    const PEV4ReferenceResult face128 = integrate_pe_v4_point_source_reference(
        1.1, -25.0, 0.8,
        3.0, 3.0, 3.0,
        0.11634, 0.06702, 128, 8);
    const double error16 = std::fabs(face16.photoelectric_probability
        - face64.photoelectric_probability);
    const double error32 = std::fabs(face32.photoelectric_probability
        - face64.photoelectric_probability);
    const double error64 = std::fabs(face64.photoelectric_probability
        - face128.photoelectric_probability);
    std::cout << "surface_values face16=" << face16.photoelectric_probability
        << " states16=" << face16.state_count
        << " face32=" << face32.photoelectric_probability
        << " states32=" << face32.state_count
        << " face64=" << face64.photoelectric_probability
        << " states64=" << face64.state_count
        << " face128=" << face128.photoelectric_probability
        << " states128=" << face128.state_count << std::endl;
    require(error32 < error16, "surface quadrature did not converge");
    require(error64 < error32, "fine surface quadrature did not converge");
    requireClose(face64.photoelectric_probability,
        face128.photoelectric_probability, 1e-12, 5e-3,
        "64x64 to 128x128 surface convergence");

    // The production detector-local calculation now consumes the same shared
    // first-interaction states. Its physical partition must still close.
    const double energy = 440.0;
    const double resolution = 0.13 * std::sqrt(511.0 / energy);
    const DetectorLocalScatterResponse local
        = integrate_detector_local_scatter_response(
            0.12, 0.98, 0.15,
            3.0, 3.0, 3.0, kMaterialGAGG,
            energy, resolution,
            (1.0 - resolution / 2.0) * energy,
            (1.0 + resolution / 2.0) * energy,
            64, 64, 4);
    requireClose(local.escape_probability
        + local.second_photoelectric_probability
        + local.second_compton_probability,
        1.0, 2e-12, 0.0, "first-interaction-state detector-local partition");

    std::cout << "opaque_solid_angle_relative_error="
        << std::fabs(opaque.photoelectric_probability
            / exact_solid_angle_fraction - 1.0) << std::endl;
    std::cout << "mixed_closure_error="
        << mixed.first_interaction_probability
            - mixed.photoelectric_probability - mixed.compton_probability
        << std::endl;
    std::cout << "depth_mean_error=" << std::fabs(parallel_depth
        - first_interaction_truncated_exponential_mean_depth(mu_total, 10.0))
        << std::endl;
    std::cout << "surface_16_error=" << error16
        << " surface_32_error=" << error32
        << " surface_64_error=" << error64 << std::endl;
    std::cout << "PASS pe_v4_reference_test" << std::endl;
    return EXIT_SUCCESS;
}
