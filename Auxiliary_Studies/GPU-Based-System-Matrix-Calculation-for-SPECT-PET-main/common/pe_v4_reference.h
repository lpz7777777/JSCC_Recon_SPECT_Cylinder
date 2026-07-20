#pragma once

#include <cmath>
#include <cstddef>

#include "first_interaction.h"

struct PEV4ReferenceResult
{
    double attenuated_solid_angle_fraction;
    double first_interaction_probability;
    double photoelectric_probability;
    double compton_probability;
    double mean_depth_mm;
    double mean_position_x;
    double mean_position_y;
    double mean_position_z;
    double photoelectric_probability_by_entry_face[6];
    std::size_t state_count;
};

struct PEV4UnitSurvival
{
    double operator()(const FirstInteractionState&) const
    {
        return 1.0;
    }
};

template <typename SurvivalEvaluator>
inline PEV4ReferenceResult integrate_pe_v4_point_source_reference(
    double source_x,
    double source_y,
    double source_z,
    double width_mm,
    double thickness_mm,
    double height_mm,
    double mu_photoelectric_per_mm,
    double mu_compton_per_mm,
    int face_samples_per_axis,
    int depth_samples,
    SurvivalEvaluator survival_evaluator,
    bool use_halton_surface_rule = false)
{
    PEV4ReferenceResult result = {};
    double depth_numerator = 0.0;
    double position_x_numerator = 0.0;
    double position_y_numerator = 0.0;
    double position_z_numerator = 0.0;

    for_each_point_source_box_first_interaction_state(
        source_x, source_y, source_z,
        width_mm, thickness_mm, height_mm,
        mu_photoelectric_per_mm, mu_compton_per_mm,
        face_samples_per_axis, depth_samples,
        [&](const FirstInteractionState& state)
        {
            double survival = survival_evaluator(state);
            if (!std::isfinite(survival) || survival <= 0.0) return;
            if (survival > 1.0) survival = 1.0;
            const double surface_contribution = state.surface_weight
                * state.conditional_depth_weight * survival;
            const double interaction_contribution = state.interaction_weight * survival;
            result.attenuated_solid_angle_fraction += surface_contribution;
            result.first_interaction_probability += interaction_contribution;
            result.photoelectric_probability += state.photoelectric_weight * survival;
            result.compton_probability += state.compton_weight * survival;
            const int face_index = state.entry_face_axis * 2
                + (state.entry_face_sign > 0.0 ? 1 : 0);
            if (face_index >= 0 && face_index < 6)
                result.photoelectric_probability_by_entry_face[face_index]
                    += state.photoelectric_weight * survival;
            depth_numerator += state.depth_mm * interaction_contribution;
            position_x_numerator += state.position_x * interaction_contribution;
            position_y_numerator += state.position_y * interaction_contribution;
            position_z_numerator += state.position_z * interaction_contribution;
            ++result.state_count;
        }, use_halton_surface_rule);

    if (result.first_interaction_probability > 0.0)
    {
        const double inverse = 1.0 / result.first_interaction_probability;
        result.mean_depth_mm = depth_numerator * inverse;
        result.mean_position_x = position_x_numerator * inverse;
        result.mean_position_y = position_y_numerator * inverse;
        result.mean_position_z = position_z_numerator * inverse;
    }
    return result;
}

inline PEV4ReferenceResult integrate_pe_v4_point_source_reference(
    double source_x,
    double source_y,
    double source_z,
    double width_mm,
    double thickness_mm,
    double height_mm,
    double mu_photoelectric_per_mm,
    double mu_compton_per_mm,
    int face_samples_per_axis,
    int depth_samples,
    bool use_halton_surface_rule = false)
{
    return integrate_pe_v4_point_source_reference(
        source_x, source_y, source_z,
        width_mm, thickness_mm, height_mm,
        mu_photoelectric_per_mm, mu_compton_per_mm,
        face_samples_per_axis, depth_samples, PEV4UnitSurvival(),
        use_halton_surface_rule);
}
