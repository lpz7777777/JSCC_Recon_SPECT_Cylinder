data = readmatrix("line_profile_511keV.csv");


data(:, 2:4) = data(:, 2:4) ./ max(data(:, 2:4));
data(:, 1) = data(:, 1) - 150;

%%
f = figure;
f.Position = [50, 50, 400, 300];
hold on

plot(data(:, 1), data(:, 2), "Color", "k", "LineWidth", 1.5);
plot(data(:, 1), data(:, 3), "Color", [0.48, 0.69, 0.40], "LineWidth", 1.5);
plot(data(:, 1), data(:, 4), "Color", [0.77, 0.33, 0.16], "LineWidth", 1.5);
xlabel("x");
ylabel("Relevant Voxel Value");
xlim([-110, 110]);
ax = gca;
ax.Box = "on";

% lgd = legend("Ground Truth", "JSCC", "Parallel-hole");
% lgd.Box = "off";