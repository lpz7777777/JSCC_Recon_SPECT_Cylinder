#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "../common/detector_local_scatter.h"

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static void requireClose(double actual, double expected, double tolerance, const char* message)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        std::cerr << std::setprecision(17) << message << ": actual=" << actual
            << " expected=" << expected << " tolerance=" << tolerance << std::endl;
        throw std::runtime_error(message);
    }
}

static void validateResponse(
    const DetectorLocalScatterResponse& response,
    const char* label)
{
    require(response.recoil_windowed >= 0.0, "negative recoil response");
    require(response.self_photoelectric_windowed >= 0.0,
        "negative self-photoelectric response");
    require(response.escape_probability >= 0.0, "negative escape probability");
    require(response.second_photoelectric_probability >= 0.0,
        "negative second-photoelectric probability");
    require(response.second_compton_probability >= 0.0,
        "negative second-Compton probability");
    require(response.recoil_windowed <= response.escape_probability + 1e-12,
        "windowed recoil exceeds escape probability");
    require(response.self_photoelectric_windowed
        <= response.second_photoelectric_probability + 1e-12,
        "windowed self-photoelectric exceeds physical probability");
    const double closure = response.escape_probability
        + response.second_photoelectric_probability
        + response.second_compton_probability;
    if (std::fabs(closure - 1.0) > 2e-12)
    {
        std::cerr << std::setprecision(17) << label
            << " probability closure=" << closure
            << " error=" << closure - 1.0 << std::endl;
        throw std::runtime_error("local probability partition does not sum to one");
    }
}

int main()
{
    // Hand-checkable example: mu_total=0.5/mm and L=4 mm.
    // P_exit=e^-2; the interaction branch is split 60% PE and 40% Compton.
    const LocalSecondInteractionPartition simple
        = local_second_interaction_partition(0.3, 0.2, 4.0);
    const double expected_escape = std::exp(-2.0);
    requireClose(simple.escape, expected_escape, 1e-15, "simple escape probability");
    requireClose(simple.photoelectric, (1.0 - expected_escape) * 0.6, 1e-15,
        "simple photoelectric probability");
    requireClose(simple.compton, (1.0 - expected_escape) * 0.4, 1e-15,
        "simple Compton probability");
    requireClose(simple.escape + simple.photoelectric + simple.compton, 1.0,
        1e-15, "simple probability conservation");

    requireClose(detector_center_exit_distance(1, 0, 0, 4, 10, 6), 2.0, 1e-15,
        "box +X exit distance");
    requireClose(detector_center_exit_distance(0, -1, 0, 4, 10, 6), 5.0, 1e-15,
        "box -Y exit distance");
    requireClose(detector_center_exit_distance(0, 0, 1, 4, 10, 6), 3.0, 1e-15,
        "box +Z exit distance");
    requireClose(detector_box_exit_distance(
        1, -4, 0, 0, 1, 0, 4, 10, 6), 9.0, 1e-15,
        "off-center box forward exit distance");
    requireClose(detector_box_exit_distance(
        1, -4, 0, 0, -1, 0, 4, 10, 6), 1.0, 1e-15,
        "off-center box entry depth");

    const double energy = 440.0;
    const double resolution = 0.13 * std::sqrt(511.0 / energy);
    const double lower = (1.0 - resolution / 2.0) * energy;
    const double upper = (1.0 + resolution / 2.0) * energy;
    const DetectorLocalScatterResponse coarse
        = integrate_detector_local_scatter_response(
            0.15, 0.98, 0.12, 4.0, 10.0, 4.0, kMaterialNaI,
            energy, resolution, lower, upper, 48, 48);
    const DetectorLocalScatterResponse medium
        = integrate_detector_local_scatter_response(
            0.15, 0.98, 0.12, 4.0, 10.0, 4.0, kMaterialNaI,
            energy, resolution, lower, upper, 96, 96);
    const DetectorLocalScatterResponse fine
        = integrate_detector_local_scatter_response(
            0.15, 0.98, 0.12, 4.0, 10.0, 4.0, kMaterialNaI,
            energy, resolution, lower, upper, 192, 192);
    validateResponse(coarse, "NaI coarse");
    validateResponse(medium, "NaI medium");
    validateResponse(fine, "NaI fine");

    requireClose(medium.recoil_windowed, fine.recoil_windowed, 2e-4,
        "recoil angular convergence");
    requireClose(medium.self_photoelectric_windowed,
        fine.self_photoelectric_windowed, 2e-4,
        "self-photoelectric angular convergence");
    requireClose(medium.escape_probability, fine.escape_probability, 2e-4,
        "escape angular convergence");

    // Dedicated 440 -> 218 keV-window check.  The recoil continuum overlaps
    // this window, whereas a 440 keV full-energy pulse is many sigma away.
    const double cross_lower = 196.30538;
    const double cross_upper = 239.69462;
    const DetectorLocalScatterResponse cross_window
        = integrate_detector_local_scatter_response(
            0.15, 0.98, 0.12, 4.0, 10.0, 4.0, kMaterialNaI,
            energy, resolution, cross_lower, cross_upper, 192, 192);
    validateResponse(cross_window, "NaI cross window");
    require(cross_window.recoil_windowed > 1e-4,
        "440 keV recoil continuum must overlap the 218 keV window");
    require(cross_window.self_photoelectric_windowed < 1e-12,
        "440 keV full-energy pulse must not enter the 218 keV window appreciably");
    requireClose(cross_window.escape_probability, fine.escape_probability, 2e-12,
        "physical escape probability must not depend on the selected energy window");
    requireClose(cross_window.second_photoelectric_probability,
        fine.second_photoelectric_probability, 2e-12,
        "second-PE probability must not depend on the selected energy window");

    const double full_energy_cross_acceptance = local_scatter_gaussian_acceptance(
        energy, resolution, energy, cross_lower, cross_upper);
    requireClose(cross_window.self_photoelectric_windowed,
        cross_window.second_photoelectric_probability * full_energy_cross_acceptance,
        2e-12, "self Compton+PE response must use the total deposited energy");
    const double full_energy_direct_acceptance = local_scatter_gaussian_acceptance(
        energy, resolution, energy, lower, upper);
    requireClose(fine.self_photoelectric_windowed,
        fine.second_photoelectric_probability * full_energy_direct_acceptance,
        2e-12, "direct-window self Compton+PE probability");

    const DetectorLocalScatterResponse position_medium
        = integrate_detector_local_scatter_response(
            0.12, 0.98, 0.15, 3.0, 3.0, 3.0, kMaterialGAGG,
            energy, resolution, lower, upper, 64, 64, 4);
    const DetectorLocalScatterResponse position_fine
        = integrate_detector_local_scatter_response(
            0.12, 0.98, 0.15, 3.0, 3.0, 3.0, kMaterialGAGG,
            energy, resolution, lower, upper, 64, 64, 6);
    validateResponse(position_medium, "GAGG position medium");
    validateResponse(position_fine, "GAGG position fine");
    requireClose(position_medium.self_photoelectric_windowed,
        position_fine.self_photoelectric_windowed, 2e-3,
        "first-interaction position convergence");
    requireClose(position_medium.escape_probability,
        position_fine.escape_probability, 2e-3,
        "position-integrated escape convergence");

    const DetectorLocalScatterResponse thin
        = integrate_detector_local_scatter_response(
            0, 1, 0, 1e-6, 1e-6, 1e-6, kMaterialNaI,
            energy, resolution, 0.0, energy + 1.0, 96, 96);
    validateResponse(thin, "NaI thin");
    require(thin.escape_probability > 0.999999,
        "vanishing crystal must approach unit escape probability");
    require(thin.second_photoelectric_probability < 1e-6,
        "vanishing crystal must approach zero self absorption");

    const DetectorLocalScatterResponse thick
        = integrate_detector_local_scatter_response(
            0, 1, 0, 1e6, 1e6, 1e6, kMaterialNaI,
            energy, resolution, 0.0, energy + 1.0, 96, 96);
    validateResponse(thick, "NaI thick");
    require(thick.second_photoelectric_probability
        > thin.second_photoelectric_probability,
        "thick crystal must absorb more scattered photons than a thin crystal");

    std::cout << "simple_exact escape=" << simple.escape
        << " second_pe=" << simple.photoelectric
        << " second_compton=" << simple.compton << std::endl;
    std::cout << "NaI_440_medium recoil_windowed=" << medium.recoil_windowed
        << " self_pe_windowed=" << medium.self_photoelectric_windowed
        << " escape=" << medium.escape_probability
        << " second_pe=" << medium.second_photoelectric_probability
        << " second_compton=" << medium.second_compton_probability << std::endl;
    std::cout << "convergence medium_to_fine recoil="
        << std::fabs(medium.recoil_windowed - fine.recoil_windowed)
        << " self_pe="
        << std::fabs(medium.self_photoelectric_windowed
            - fine.self_photoelectric_windowed)
        << " escape="
        << std::fabs(medium.escape_probability - fine.escape_probability)
        << std::endl;
    std::cout << "GAGG_3mm_position_convergence self_pe="
        << std::fabs(position_medium.self_photoelectric_windowed
            - position_fine.self_photoelectric_windowed)
        << " escape="
        << std::fabs(position_medium.escape_probability
            - position_fine.escape_probability)
        << " fine_second_pe=" << position_fine.second_photoelectric_probability
        << std::endl;
    std::cout << "NaI_440_to_218_window recoil_windowed="
        << cross_window.recoil_windowed
        << " self_pe_windowed=" << cross_window.self_photoelectric_windowed
        << " full_energy_tail_acceptance=" << full_energy_cross_acceptance
        << std::endl;
    std::cout << "PASS detector_local_scatter_test" << std::endl;
    return EXIT_SUCCESS;
}
