PixelNumX = 100;
PixelNumY = 100;
PixelNumZ = 20;

axis_font_size = 21;
title_font_size = 24;

%% Select reconstruction folder
folderPath = uigetdir("./");
if isequal(folderPath, 0)
    return
end
Path = fullfile(folderPath, "Cartesian");

%% Select ground truth file
[gt_file, gt_path] = uigetfile( ...
    {"*.raw", "Raw Image Files (*.raw)"}, ...
    "Select Ground Truth Image", ...
    "./");
if isequal(gt_file, 0)
    return
end
gt_full_path = fullfile(gt_path, gt_file);

%% Parse iteration file
iterInfo = parse_iter_file(Path, "Image_*_Iter_*");
iter_max = iterInfo.iterMax;
iter_interval = iterInfo.iterInterval;
save_count = iterInfo.saveCount;
iterations = iter_interval:iter_interval:iter_max;

%% Read ground truth
img_gt = read_float32_tensor(gt_full_path, [PixelNumX * PixelNumY * PixelNumZ, 1]);
img_gt_3d = reshape(img_gt, [PixelNumX, PixelNumY, PixelNumZ]);

% Ground truth mean for normalization
gt_mean = mean(img_gt(img_gt > 0));

%% Read iteration images
img_iter = read_float32_tensor( ...
    fullfile(Path, iterInfo.fileName), ...
    [PixelNumX * PixelNumY * PixelNumZ, save_count]);

%% Compute SSIM for each iteration
ssim_values = zeros(1, save_count);

fprintf("Computing SSIM for %d iterations...\n", save_count);
for i = 1:save_count
    img_recon = img_iter(:, i);
    
    % Normalize: scale reconstruction so that its mean matches ground truth mean
    recon_mean = mean(img_recon(img_recon > 0));
    if recon_mean > 0
        img_recon = img_recon * (gt_mean / recon_mean);
    end
    
    img_recon_3d = reshape(img_recon, [PixelNumX, PixelNumY, PixelNumZ]);
    
    % Compute 3D SSIM
    ssim_values(i) = ssim(img_recon_3d, img_gt_3d);
    
    if mod(i, 10) == 0 || i == save_count
        fprintf("  Iteration %d/%d: SSIM = %.6f\n", iterations(i), iter_max, ssim_values(i));
    end
end

%% Find best iteration
[best_ssim, best_idx] = max(ssim_values);
best_iter = iterations(best_idx);
fprintf("\nBest SSIM = %.6f at Iteration %d\n", best_ssim, best_iter);

%% Plot SSIM vs Iteration
f = figure;
f.Position = [100, 100, 600, 450];
plot(iterations, ssim_values, "LineWidth", 2.5, "Color", [0.2, 0.4, 0.8], ...
    "Marker", "o", "MarkerSize", 6, "MarkerFaceColor", [0.2, 0.4, 0.8]);
hold on;

% Mark best iteration
plot(best_iter, best_ssim, "r*", "MarkerSize", 18, "LineWidth", 2);
text(best_iter, best_ssim + 0.005 * (max(ssim_values) - min(ssim_values)), ...
    sprintf("Best: %d", best_iter), ...
    "HorizontalAlignment", "center", "FontSize", 14, "Color", "r");

xlim([0 iter_max]);
xlabel("Iteration");
ylabel("SSIM");
title("SSIM vs Iteration");
grid on;
style_axes(gca, axis_font_size, title_font_size);

saveas(f, fullfile(folderPath, "ssim_vs_iteration.png"));
savefig(f, fullfile(folderPath, "ssim_vs_iteration.fig"));
fprintf("Figure saved to %s\n", fullfile(folderPath, "ssim_vs_iteration.png"));

%% Visual Comparison at Optimal Iteration
img_best = reshape(img_iter(:, best_idx), [PixelNumX, PixelNumY, PixelNumZ]);
% Normalize best image
recon_mean_best = mean(img_best(img_best > 0));
if recon_mean_best > 0
    img_best = img_best * (gt_mean / recon_mean_best);
end
z_slice = floor(PixelNumZ / 2);

colorMap = flipud(gray(1024));
f_best = figure;
f_best.Position = [200, 200, 900, 350];

subplot(1, 3, 1);
imagesc(img_gt_3d(:, :, z_slice));
colormap(colorMap); axis image; colorbar;
title("Ground Truth");
xlabel("X"); ylabel("Y");

subplot(1, 3, 2);
imagesc(img_best(:, :, z_slice));
colormap(colorMap); axis image; colorbar;
title(sprintf("Best Recon (Iter %d)", best_iter));
xlabel("X"); ylabel("Y");

subplot(1, 3, 3);
diff_img = abs(img_best(:, :, z_slice) - img_gt_3d(:, :, z_slice));
imagesc(diff_img);
colormap(hot(1024)); axis image; colorbar;
title("|Difference|");
xlabel("X"); ylabel("Y");

saveas(f_best, fullfile(folderPath, "best_iter_comparison.png"));
savefig(f_best, fullfile(folderPath, "best_iter_comparison.fig"));
fprintf("Comparison figure saved to %s\n", fullfile(folderPath, "best_iter_comparison.png"));


%% Local functions

function iterInfo = parse_iter_file(folderPath, pattern)
    matches = dir(fullfile(folderPath, pattern));
    if isempty(matches)
        error("Cannot find %s under %s.", pattern, folderPath);
    end
    if numel(matches) > 1
        warning("Found multiple files for %s. Using %s.", pattern, matches(1).name);
    end
    
    tokens = regexp(matches(1).name, ".*_Iter_(\d+)_(\d+)$", "tokens", "once");
    if isempty(tokens)
        error("Failed to parse iteration info from %s.", matches(1).name);
    end
    
    iterInfo.fileName = matches(1).name;
    iterInfo.iterMax = str2double(tokens{1});
    iterInfo.saveCount = str2double(tokens{2});
    iterInfo.iterInterval = round(iterInfo.iterMax / iterInfo.saveCount);
end


function tensor = read_float32_tensor(filePath, tensorShape)
    fid = fopen(filePath, "r");
    if fid < 0
        error("Failed to open %s.", filePath);
    end
    cleanupObj = onCleanup(@() fclose(fid));
    raw = fread(fid, "float32");
    expectedNumel = prod(tensorShape);
    if numel(raw) ~= expectedNumel
        error("Unexpected element count in %s: expected %d, got %d.", filePath, expectedNumel, numel(raw));
    end
    tensor = reshape(raw, tensorShape);
end


function style_axes(ax, axisFontSize, titleFontSize)
    ax.Box = "on";
    ax.LineWidth = 1.5;
    ax.FontSize = axisFontSize;
    ax.TitleFontSizeMultiplier = 1.0;
    ax.Title.FontSize = titleFontSize;
    ax.XLabel.FontSize = axisFontSize;
    ax.YLabel.FontSize = axisFontSize;
end