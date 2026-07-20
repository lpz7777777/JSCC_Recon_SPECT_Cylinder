function [volume_full, metadata] = calculate_polar_cell_volumes(coor_polar, z_axis)
%CALCULATE_POLAR_CELL_VOLUMES Physical volume represented by each polar sample.
% Radial and axial boundaries are adjacent-sample midpoints. Every sample on
% one ring owns an equal angular sector. Output ordering matches
% coor_polar_full: all transverse points at z1, then z2, and so on.

    arguments
        coor_polar (:, 2) double
        z_axis (:, 1) double
    end

    radius = round(hypot(coor_polar(:, 1), coor_polar(:, 2)), 8);
    radii = unique(radius, 'sorted');
    z_values = unique(z_axis(:), 'sorted');
    if numel(radii) < 2 || numel(z_values) < 2
        error('calculate_polar_cell_volumes:InsufficientGrid', ...
            'At least two radii and two z layers are required.');
    end

    radial_edges = midpoint_edges(radii, radii(1) == 0, true);
    z_edges = midpoint_edges(z_values, false, false);
    points_per_ring = zeros(numel(radii), 1);
    area_per_point = zeros(numel(radii), 1);
    for idx = 1:numel(radii)
        points_per_ring(idx) = sum(radius == radii(idx));
        area_per_point(idx) = pi * ...
            (radial_edges(idx + 1)^2 - radial_edges(idx)^2) / ...
            points_per_ring(idx);
    end

    [found, radial_index] = ismember(radius, radii);
    if ~all(found)
        error('calculate_polar_cell_volumes:RadiusMapping', ...
            'Failed to map every sample to a radius ring.');
    end
    thickness = diff(z_edges);
    volume_grid = area_per_point(radial_index) * thickness.';
    volume_full = volume_grid(:);
    if any(~isfinite(volume_full)) || any(volume_full <= 0)
        error('calculate_polar_cell_volumes:InvalidVolume', ...
            'Polar-cell volume must be finite and positive.');
    end

    analytic_volume = pi * ...
        (radial_edges(end)^2 - radial_edges(1)^2) * ...
        (z_edges(end) - z_edges(1));
    represented_volume = sum(volume_full);
    relative_error = represented_volume / analytic_volume - 1;
    if abs(relative_error) > 1e-12
        error('calculate_polar_cell_volumes:Closure', ...
            'Polar-cell volume closure error is %.6g.', relative_error);
    end

    metadata = struct();
    metadata.method = ...
        'midpoint radial/axial bounds and equal angular sectors per ring';
    metadata.units = 'mm3';
    metadata.coordinate_count = numel(volume_full);
    metadata.radii_mm = radii.';
    metadata.points_per_ring = points_per_ring.';
    metadata.z_values_mm = z_values.';
    metadata.radial_inner_domain_mm = radial_edges(1);
    metadata.radial_outer_domain_mm = radial_edges(end);
    metadata.axial_lower_domain_mm = z_edges(1);
    metadata.axial_upper_domain_mm = z_edges(end);
    metadata.minimum_mm3 = min(volume_full);
    metadata.maximum_mm3 = max(volume_full);
    metadata.mean_mm3 = mean(volume_full);
    metadata.median_mm3 = median(volume_full);
    metadata.sum_mm3 = represented_volume;
    metadata.analytic_domain_volume_mm3 = analytic_volume;
    metadata.relative_volume_closure_error = relative_error;
end


function edges = midpoint_edges(values, force_first_to_zero, clamp_nonnegative)
    values = values(:);
    edges = zeros(numel(values) + 1, 1);
    edges(2:end-1) = (values(1:end-1) + values(2:end)) / 2;
    edges(1) = values(1) - (values(2) - values(1)) / 2;
    edges(end) = values(end) + (values(end) - values(end-1)) / 2;
    if force_first_to_zero
        edges(1) = 0;
    elseif clamp_nonnegative
        edges(1) = max(edges(1), 0);
    end
end
