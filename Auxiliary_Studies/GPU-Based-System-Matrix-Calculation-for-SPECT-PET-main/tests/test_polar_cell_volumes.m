function tests = test_polar_cell_volumes
%TEST_POLAR_CELL_VOLUMES Regression tests for the density-basis polar grid.
    tests = functiontests(localfunctions);
end


function testCenterInclusiveProductionGrid(testCase)
    [coor, z_axis] = production_grid(true);
    [volume, metadata] = calculate_polar_cell_volumes(coor, z_axis);
    verifyEqual(testCase, numel(volume), 25620);
    verifyEqual(testCase, min(volume), 33.929200658769766, 'AbsTol', 1e-11);
    verifyEqual(testCase, max(volume), 212.05750411731103, 'AbsTol', 1e-11);
    verifyLessThanOrEqual(testCase, ...
        abs(metadata.relative_volume_closure_error), 1e-12);
    verifyEqual(testCase, metadata.points_per_ring, ...
        [1, 20*ones(1, 6), 40*ones(1, 6), ...
         60*ones(1, 6), 80*ones(1, 7)]);
end


function testRotationPreservesVolume(testCase)
    [coor, z_axis, rotmat] = production_grid(true);
    volume = calculate_polar_cell_volumes(coor, z_axis);
    verifyEqual(testCase, volume(rotmat), repmat(volume, 1, 20), ...
        'AbsTol', 1e-12);
end


function [coor, z_axis, rotmat] = production_grid(include_center)
    radii = 6:6:150;
    theta_counts = repelem([20, 40, 60, 80], [6, 6, 6, 7]);
    if include_center
        coor = [0, 0];
        rotmat = ones(1, 20);
    else
        coor = zeros(0, 2);
        rotmat = zeros(0, 20);
    end
    for ring_idx = 1:numel(radii)
        count = theta_counts(ring_idx);
        theta = (0:count-1).' * 360 / count;
        first = size(coor, 1) + 1;
        coor = [coor; radii(ring_idx) * cosd(theta), ...
            radii(ring_idx) * sind(theta)]; %#ok<AGROW>
        ring_indices = (first:first+count-1).';
        ring_rotation = zeros(count, 20);
        interval = count / 20;
        for view = 1:20
            ring_rotation(:, view) = ring_indices( ...
                mod((view-1)*interval + (0:count-1), count) + 1);
        end
        rotmat = [rotmat; ring_rotation]; %#ok<AGROW>
    end
    z_axis = (-28.5:3:28.5).';
    points_per_layer = size(coor, 1);
    rotmat_layer = rotmat;
    layer_cells = arrayfun(@(layer) ...
        rotmat_layer + (layer-1)*points_per_layer, 1:numel(z_axis), ...
        'UniformOutput', false);
    rotmat = vertcat(layer_cells{:});
end
