#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

// One quadrature state for the first interaction inside a rectangular detector.
// Coordinates and directions are detector-local. Weights are per incident
// primary before any attenuation outside the target detector is applied.
struct FirstInteractionState
{
    int entry_face_axis;
    double entry_face_sign;
    double entry_x;
    double entry_y;
    double entry_z;
    double position_x;
    double position_y;
    double position_z;
    double incoming_x;
    double incoming_y;
    double incoming_z;
    double chord_mm;
    double depth_mm;
    double surface_weight;
    double conditional_depth_weight;
    double first_interaction_probability;
    double interaction_weight;
    double photoelectric_weight;
    double compton_weight;
};

inline bool normalize_first_interaction_direction(
    double* direction_x,
    double* direction_y,
    double* direction_z)
{
    const double norm = std::sqrt(*direction_x * *direction_x
        + *direction_y * *direction_y + *direction_z * *direction_z);
    if (!(norm > 0.0) || !std::isfinite(norm)) return false;
    *direction_x /= norm;
    *direction_y /= norm;
    *direction_z /= norm;
    return true;
}

inline double detector_box_exit_distance(
    double position_x,
    double position_y,
    double position_z,
    double direction_x,
    double direction_y,
    double direction_z,
    double width_mm,
    double thickness_mm,
    double height_mm)
{
    const double half_extent[3] = {
        0.5 * width_mm,
        0.5 * thickness_mm,
        0.5 * height_mm
    };
    const double position[3] = {position_x, position_y, position_z};
    const double direction[3] = {direction_x, direction_y, direction_z};
    double distance = std::numeric_limits<double>::infinity();
    const double epsilon = 1e-12;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (std::fabs(position[axis]) > half_extent[axis] + epsilon)
            return 0.0;
        if (direction[axis] > epsilon)
            distance = std::fmin(distance,
                (half_extent[axis] - position[axis]) / direction[axis]);
        else if (direction[axis] < -epsilon)
            distance = std::fmin(distance,
                (-half_extent[axis] - position[axis]) / direction[axis]);
    }
    if (!std::isfinite(distance) || distance < -epsilon) return 0.0;
    return std::fmax(0.0, distance);
}

inline double detector_center_exit_distance(
    double direction_x,
    double direction_y,
    double direction_z,
    double width_mm,
    double thickness_mm,
    double height_mm)
{
    return detector_box_exit_distance(
        0.0, 0.0, 0.0,
        direction_x, direction_y, direction_z,
        width_mm, thickness_mm, height_mm);
}

inline double detector_segment_box_chord(
    double start_x,
    double start_y,
    double start_z,
    double end_x,
    double end_y,
    double end_z,
    double width_mm,
    double thickness_mm,
    double height_mm)
{
    const double start[3] = {start_x, start_y, start_z};
    const double delta[3] = {
        end_x - start_x,
        end_y - start_y,
        end_z - start_z
    };
    const double half_extent[3] = {
        0.5 * width_mm,
        0.5 * thickness_mm,
        0.5 * height_mm
    };
    const double segment_length = std::sqrt(delta[0] * delta[0]
        + delta[1] * delta[1] + delta[2] * delta[2]);
    if (!(segment_length > 0.0)) return 0.0;

    double lower = 0.0;
    double upper = 1.0;
    const double epsilon = 1e-14;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (std::fabs(delta[axis]) <= epsilon)
        {
            if (start[axis] < -half_extent[axis]
                || start[axis] > half_extent[axis])
                return 0.0;
            continue;
        }
        double first = (-half_extent[axis] - start[axis]) / delta[axis];
        double second = (half_extent[axis] - start[axis]) / delta[axis];
        if (first > second)
        {
            const double temporary = first;
            first = second;
            second = temporary;
        }
        lower = std::fmax(lower, first);
        upper = std::fmin(upper, second);
        if (lower >= upper) return 0.0;
    }
    return std::fmax(0.0, upper - lower) * segment_length;
}

inline double first_interaction_truncated_exponential_mean_depth(
    double mu_total_per_mm,
    double chord_mm)
{
    if (!(mu_total_per_mm > 0.0) || !(chord_mm > 0.0)) return 0.0;
    const double attenuation = std::exp(-mu_total_per_mm * chord_mm);
    const double interaction = 1.0 - attenuation;
    if (!(interaction > 0.0)) return 0.0;
    return 1.0 / mu_total_per_mm - chord_mm * attenuation / interaction;
}

template <typename Visitor>
inline void emit_first_interaction_depth_states(
    const double entry[3],
    const double incoming[3],
    int entry_face_axis,
    double entry_face_sign,
    double chord_mm,
    double surface_weight,
    double mu_photoelectric_per_mm,
    double mu_compton_per_mm,
    int depth_samples,
    Visitor visitor)
{
    const double mu_total = mu_photoelectric_per_mm + mu_compton_per_mm;
    if (!(mu_total > 0.0) || !(chord_mm > 0.0)
        || !(surface_weight > 0.0) || depth_samples < 1)
        return;

    const double first_probability = -std::expm1(-mu_total * chord_mm);
    if (!(first_probability > 0.0)) return;
    const double conditional_weight = 1.0 / depth_samples;
    const double interaction_weight = surface_weight
        * first_probability * conditional_weight;

    for (int depth_index = 0; depth_index < depth_samples; ++depth_index)
    {
        const double quantile = (depth_index + 0.5) * conditional_weight;
        const double survival = std::fmax(
            std::numeric_limits<double>::min(), 1.0 - quantile * first_probability);
        const double depth = std::fmin(chord_mm, -std::log(survival) / mu_total);

        FirstInteractionState state;
        state.entry_face_axis = entry_face_axis;
        state.entry_face_sign = entry_face_sign;
        state.entry_x = entry[0];
        state.entry_y = entry[1];
        state.entry_z = entry[2];
        state.position_x = entry[0] + incoming[0] * depth;
        state.position_y = entry[1] + incoming[1] * depth;
        state.position_z = entry[2] + incoming[2] * depth;
        state.incoming_x = incoming[0];
        state.incoming_y = incoming[1];
        state.incoming_z = incoming[2];
        state.chord_mm = chord_mm;
        state.depth_mm = depth;
        state.surface_weight = surface_weight;
        state.conditional_depth_weight = conditional_weight;
        state.first_interaction_probability = first_probability;
        state.interaction_weight = interaction_weight;
        state.photoelectric_weight = interaction_weight
            * mu_photoelectric_per_mm / mu_total;
        state.compton_weight = interaction_weight
            * mu_compton_per_mm / mu_total;
        visitor(state);
    }
}

// Parallel-beam states are used by the detector-local response lookup. The
// surface weight is projected area; consumers normalize by the accumulated
// first-interaction weight, so absolute beam fluence cancels.
template <typename Visitor>
inline void for_each_parallel_box_first_interaction_state(
    double incoming_x,
    double incoming_y,
    double incoming_z,
    double width_mm,
    double thickness_mm,
    double height_mm,
    double mu_photoelectric_per_mm,
    double mu_compton_per_mm,
    int face_samples_per_axis,
    int depth_samples,
    Visitor visitor)
{
    if (!(width_mm > 0.0) || !(thickness_mm > 0.0) || !(height_mm > 0.0)
        || face_samples_per_axis < 1 || depth_samples < 1
        || !normalize_first_interaction_direction(
            &incoming_x, &incoming_y, &incoming_z))
        return;

    const double half_extent[3] = {
        0.5 * width_mm,
        0.5 * thickness_mm,
        0.5 * height_mm
    };
    const double incoming[3] = {incoming_x, incoming_y, incoming_z};
    const double axis_length[3] = {width_mm, thickness_mm, height_mm};
    const double direction_epsilon = 1e-14;

    for (int normal_axis = 0; normal_axis < 3; ++normal_axis)
    {
        if (std::fabs(incoming[normal_axis]) <= direction_epsilon) continue;
        const int first_axis = (normal_axis + 1) % 3;
        const int second_axis = (normal_axis + 2) % 3;
        const double entry_sign = incoming[normal_axis] > 0.0 ? -1.0 : 1.0;
        const double projected_cell_area = std::fabs(incoming[normal_axis])
            * axis_length[first_axis] * axis_length[second_axis]
            / (face_samples_per_axis * face_samples_per_axis);

        for (int first_index = 0; first_index < face_samples_per_axis; ++first_index)
        {
            for (int second_index = 0; second_index < face_samples_per_axis;
                ++second_index)
            {
                double entry[3] = {0.0, 0.0, 0.0};
                entry[normal_axis] = entry_sign * half_extent[normal_axis];
                entry[first_axis] = axis_length[first_axis]
                    * ((first_index + 0.5) / face_samples_per_axis - 0.5);
                entry[second_axis] = axis_length[second_axis]
                    * ((second_index + 0.5) / face_samples_per_axis - 0.5);
                const double chord = detector_box_exit_distance(
                    entry[0], entry[1], entry[2],
                    incoming[0], incoming[1], incoming[2],
                    width_mm, thickness_mm, height_mm);
                emit_first_interaction_depth_states(
                    entry, incoming, normal_axis, entry_sign,
                    chord, projected_cell_area,
                    mu_photoelectric_per_mm, mu_compton_per_mm,
                    depth_samples, visitor);
            }
        }
    }
}

inline double first_interaction_radical_inverse(unsigned long long index, unsigned int base)
{
    double inverse_base = 1.0 / base;
    double factor = inverse_base;
    double value = 0.0;
    while (index > 0)
    {
        value += factor * (index % base);
        index /= base;
        factor *= inverse_base;
    }
    return value;
}

// Finite-distance point-source states are the PE v4 reference model. Surface
// weight is dOmega/(4*pi), so summing photoelectric_weight gives the absolute
// first-photoelectric probability for an unattenuated isotropic primary.
// Composite two-point Gauss is the smooth-integrand rule. The nested Halton
// rule avoids phase locking when upstream crystal shadows create discontinuities.
template <typename Visitor>
inline void for_each_point_source_box_first_interaction_state(
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
    Visitor visitor,
    bool use_halton_surface_rule = false)
{
    if (!(width_mm > 0.0) || !(thickness_mm > 0.0) || !(height_mm > 0.0)
        || face_samples_per_axis < 1 || depth_samples < 1)
        return;

    const double pi = 3.14159265358979323846;
    const double half_extent[3] = {
        0.5 * width_mm,
        0.5 * thickness_mm,
        0.5 * height_mm
    };
    const double axis_length[3] = {width_mm, thickness_mm, height_mm};
    const double source[3] = {source_x, source_y, source_z};

    for (int normal_axis = 0; normal_axis < 3; ++normal_axis)
    {
        double face_sign = 0.0;
        if (source[normal_axis] < -half_extent[normal_axis]) face_sign = -1.0;
        else if (source[normal_axis] > half_extent[normal_axis]) face_sign = 1.0;
        else continue;

        const int first_axis = (normal_axis + 1) % 3;
        const int second_axis = (normal_axis + 2) % 3;
        const double face_area = axis_length[first_axis] * axis_length[second_axis];
        const double cell_area = face_area
            / (face_samples_per_axis * face_samples_per_axis);
        const double sample_area = 0.25 * cell_area;
        const double gauss_offset = 0.5 / std::sqrt(3.0);
        const double gauss_position[2] = {
            0.5 - gauss_offset,
            0.5 + gauss_offset
        };

        const auto emit_surface_sample = [&](
            double first_fraction,
            double second_fraction,
            double area_weight)
        {
            double entry[3] = {0.0, 0.0, 0.0};
            entry[normal_axis] = face_sign * half_extent[normal_axis];
            entry[first_axis] = axis_length[first_axis] * (first_fraction - 0.5);
            entry[second_axis] = axis_length[second_axis] * (second_fraction - 0.5);

            double incoming[3] = {
                entry[0] - source[0],
                entry[1] - source[1],
                entry[2] - source[2]
            };
            const double distance_squared = incoming[0] * incoming[0]
                + incoming[1] * incoming[1] + incoming[2] * incoming[2];
            if (!(distance_squared > 0.0)
                || !normalize_first_interaction_direction(
                    &incoming[0], &incoming[1], &incoming[2]))
                return;
            const double projected_cosine = -face_sign * incoming[normal_axis];
            if (!(projected_cosine > 0.0)) return;

            const double surface_weight = projected_cosine * area_weight
                / (4.0 * pi * distance_squared);
            const double chord = detector_box_exit_distance(
                entry[0], entry[1], entry[2],
                incoming[0], incoming[1], incoming[2],
                width_mm, thickness_mm, height_mm);
            emit_first_interaction_depth_states(
                entry, incoming, normal_axis, face_sign,
                chord, surface_weight,
                mu_photoelectric_per_mm, mu_compton_per_mm,
                depth_samples, visitor);
        };

        if (use_halton_surface_rule)
        {
            const unsigned long long sample_count
                = static_cast<unsigned long long>(face_samples_per_axis)
                * face_samples_per_axis;
            const double halton_area = face_area / sample_count;
            const unsigned long long symmetric_groups = sample_count / 4;
            for (unsigned long long group = 1; group <= symmetric_groups; ++group)
            {
                const double u = first_interaction_radical_inverse(group, 2);
                const double v = first_interaction_radical_inverse(group, 3);
                emit_surface_sample(u, v, halton_area);
                emit_surface_sample(1.0 - u, v, halton_area);
                emit_surface_sample(u, 1.0 - v, halton_area);
                emit_surface_sample(1.0 - u, 1.0 - v, halton_area);
            }
            if (sample_count % 4 == 1)
                emit_surface_sample(0.5, 0.5, halton_area);
        }
        else
        {
            for (int first_index = 0; first_index < face_samples_per_axis; ++first_index)
            {
                for (int second_index = 0; second_index < face_samples_per_axis;
                    ++second_index)
                {
                    for (int first_gauss = 0; first_gauss < 2; ++first_gauss)
                    {
                        for (int second_gauss = 0; second_gauss < 2; ++second_gauss)
                        {
                            emit_surface_sample(
                                (first_index + gauss_position[first_gauss])
                                    / face_samples_per_axis,
                                (second_index + gauss_position[second_gauss])
                                    / face_samples_per_axis,
                                sample_area);
                        }
                    }
                }
            }
        }
    }
}
