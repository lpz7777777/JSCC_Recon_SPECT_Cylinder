interp_factor = 2;
PixelNumX = 100;
PixelNumY = 100;
PixelNumZ = 20;
PixelNumX_new = 2^interp_factor * (PixelNumX-1) + 1;
PixelNumY_new = 2^interp_factor * (PixelNumY-1) + 1;
PixelNumZ_new = 2^interp_factor * (PixelNumZ-1) + 1;
% PixelLengthX = 3/2^interp_factor;
% PixelLengthY = 3/2^interp_factor;
% PixelLengthZ = 3/2^interp_factor;

PixelLengthX = 3;
PixelLengthY = 3;

back_rod_r = 120;
rod_r = 8:2:18;
r_tocenter = 60;

C = 6;

folderPath = uigetdir("./Figure/");
if folderPath ~= 0
    slashPositions = strfind(folderPath, '\');
    if ~isempty(slashPositions)
        Name = folderPath(slashPositions(end) + 1 : end);
    else
        Name = folderPath;
    end
end
% Name = "ContrastPhantom_70_5e9_1_ER0.08_OSEM4_SDU5726210_DDU433766";
Path = sprintf("./Figure/%s/Cartesian/", Name);

iter_max = 20000;
iter_interval = 100;

iterations = iter_interval:iter_interval:iter_max;

fid = fopen(sprintf("%sImage_SC_Iter_%d_%d", Path, iter_max, floor(iter_max/iter_interval)), "r");
img_sc_iter = reshape(fread(fid, "float32"), PixelNumX*PixelNumY*PixelNumZ, []);
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCSD_Iter_%d_%d", Path, iter_max, floor(iter_max/iter_interval)), "r");
img_jsccsd_iter = reshape(fread(fid, "float32"), PixelNumX*PixelNumY*PixelNumZ, []);
fclose(fid);


% img_sc_iter_new = zeros(PixelNumX_new, PixelNumY_new, PixelNumZ_new, size(img_sc_iter, 2));
% img_jsccsd_iter_new = zeros(PixelNumX_new, PixelNumY_new, PixelNumZ_new, size(img_sc_iter, 2));
% 
% for i = 1 : size(img_sc_iter, 2)
%     img_sc_iter_new(:, :, :, i) = interp3(reshape(img_sc_iter(:, i), PixelNumX, PixelNumY, PixelNumZ), interp_factor);
%     img_jsccsd_iter_new(:, :, :, i) = interp3(reshape(img_jsccsd_iter(:, i), PixelNumX, PixelNumY, PixelNumZ), interp_factor);
% end
% 
% img_sc_iter = reshape(img_sc_iter_new, [], size(img_sc_iter, 2));
% img_jsccsd_iter = reshape(img_jsccsd_iter_new, [], size(img_jsccsd_iter, 2));

mean_sc = zeros(6, size(img_sc_iter, 2));
mean_jsccsd = mean_sc;
std_sc = zeros(6, size(img_sc_iter, 2));
std_jsccsd = mean_sc;

% PixelNumX = PixelNumX_new;
% PixelNumY = PixelNumY_new;
% PixelNumZ = PixelNumZ_new;

mean_sc_b = zeros(1, size(img_sc_iter, 2));
mean_jsccsd_b = mean_sc_b;

start_id_z = 6;
end_id_z = 15;

Mask_b = zeros(PixelNumX, PixelNumY, PixelNumZ);
for i = 1 : PixelNumX
    for j = 1 : PixelNumY
        X = (i - 1/2 - PixelNumX/2) * PixelLengthX;
        Y = (j - 1/2 - PixelNumY/2) * PixelLengthY;
        
        if X^2 + Y^2 <= (back_rod_r*0.9)^2
            Mask_b(i, j, start_id_z:end_id_z) = 1;
        end
    end
end


for Id_R = 1 : 6
    R = rod_r(Id_R);
    Mask = zeros(PixelNumX, PixelNumY, PixelNumZ);
    Mask_tmp = zeros(PixelNumX, PixelNumY, PixelNumZ);
    
    theta_tmp = (Id_R-1) * pi/3;
    x_tmp = r_tocenter * cos(theta_tmp);
    y_tmp = r_tocenter * sin(theta_tmp);

    for i = 1 : PixelNumX
        for j = 1 : PixelNumY
            X = (i - 1/2 - PixelNumX/2) * PixelLengthX;
            Y = (j - 1/2 - PixelNumY/2) * PixelLengthY;
            
            if (X-x_tmp)^2 + (Y-y_tmp)^2 <= (0.9*R)^2
                Mask(i, j, start_id_z:end_id_z) = 1;
            end

            if (X-x_tmp)^2 + (Y-y_tmp)^2 <= (1.1*R)^2
                Mask_tmp(i, j, start_id_z:end_id_z) = 1;
            end
        end
    end

    Mask_b = Mask_b - Mask_tmp;

    Mask = logical(reshape(Mask, [], 1));
    
    img_sc_iter_masked = img_sc_iter(Mask, :);
    img_jsccsd_iter_masked = img_jsccsd_iter(Mask, :);

    mean_sc(Id_R, :) = mean(img_sc_iter_masked, 1);
    mean_jsccsd(Id_R, :) = mean(img_jsccsd_iter_masked, 1);
    std_sc(Id_R, :) = std(img_sc_iter_masked, 1);
    std_jsccsd(Id_R, :) = std(img_jsccsd_iter_masked, 1);
end

Mask_b = logical(reshape(Mask_b, [], 1));

img_sc_iter_masked = img_sc_iter(Mask_b, :);
img_jsccsd_iter_masked = img_jsccsd_iter(Mask_b, :);

mean_sc_b = mean(img_sc_iter_masked, 1);
mean_jsccsd_b = mean(img_jsccsd_iter_masked, 1);
std_sc_b = std(img_sc_iter_masked, 1);
std_jsccsd_b = std(img_jsccsd_iter_masked, 1);

%
crc_sc = (mean_sc - mean_sc_b) ./ mean_sc_b / (C - 1);
cnr_sc = (mean_sc - mean_sc_b) ./ std_sc_b;

crc_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ mean_jsccsd_b / (C - 1);
cnr_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ std_jsccsd_b;

f = figure;
t = tiledlayout(2, 2);
f.Position = [50, 50, 1000, 800];
t.TileSpacing = "tight";
t.Padding  = "tight";
Color = parula;

nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    plot(iterations, crc_sc(i, :), "Color", Color_tmp);
end
legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=20mm", "D=37mm", "Location", "southeast");
ylim([0 1]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("SC CRC");

nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    plot(iterations, crc_jsccsd(i, :), "Color", Color_tmp);
end
legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=20mm", "D=37mm", "Location", "southeast");
ylim([0 1]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("JSCC CRC");

nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    plot(iterations, cnr_sc(i, :), "Color", Color_tmp);
end
legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=20mm", "D=37mm");
ylim([0 6]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("SC CNR");

nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    plot(iterations, cnr_jsccsd(i, :), "Color", Color_tmp);
end
legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=20mm", "D=37mm");
% lgd.NumColumns = 2;
% legend('boxoff')
ylim([0 6]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("JSCC CNR");

% nexttile
% plot(iterations, cat(1, std_sc, std_sc_b).', "-o", MarkerSize=2);
% legend("R=6mm", "R=8mm", "R=10mm", "R=12mm", "R=14mm", "R=16mm", "background");
% xlabel("Iteration");
% ylabel("CNR");
% title("SC std");
% 
% nexttile
% plot(iterations, cat(1, std_jsccsd, std_jsccsd_b).', "-o", MarkerSize=2);
% legend("R=6mm", "R=8mm", "R=10mm", "R=12mm", "R=14mm", "R=16mm", "background");
% xlabel("Iteration");
% ylabel("CNR");
% title("JSCC std");


% f = figure;
% plot(iterations, mean_sc);
% hold on
% plot(iterations, mean_jsccsd);
%%
% 假設前面的數據計算部分保持不變
crc_sc = (mean_sc - mean_sc_b) ./ mean_sc_b / (C - 1);
cnr_sc = (mean_sc - mean_sc_b) ./ std_sc_b;

crc_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ mean_jsccsd_b / (C - 1);
cnr_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ std_jsccsd_b;

f = figure;
t = tiledlayout(2, 2);
f.Position = [50, 50, 1000, 800];
t.TileSpacing = "tight";
t.Padding = "tight";
Color = turbo;

% 定義 6 種不同的 Marker 形狀，以便區分不同的 D 值
MarkerList = {'.', '.', '.', '.', '.', '.'}; 
% 計算 Marker 的索引位置：從第 1 個點開始，每隔 100 個點畫一個
% 注意：這裡假設 iterations 的長度與數據列長度一致
idx_markers = 80:80:length(iterations); 

% Plot 1: SC CRC
nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);
    
    % plot(iterations, crc_sc(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...           % 設定標記形狀
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);         % 設定僅在特定索引處顯示標記

    plot(iterations, crc_sc(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...           % 設定標記形狀
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);         % 設定僅在特定索引處顯示標記
end
legend("D=16mm", "D=20mm", "D=24mm", "D=28mm", "D=32mm", "D=36mm");
grid on
grid minor
ylim([0 1]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("SC CRC");

% Plot 2: JSCC CRC
nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    % plot(iterations, crc_jsccsd(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, crc_jsccsd(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
legend("D=16mm", "D=20mm", "D=24mm", "D=28mm", "D=32mm", "D=36mm");
grid on
grid minor
ylim([0 1]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("JSCC CRC");

% Plot 3: SC CNR
nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    % plot(iterations, cnr_sc(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_sc(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
legend("D=16mm", "D=20mm", "D=24mm", "D=28mm", "D=32mm", "D=36mm");
ylim([0 9]);
grid on
grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("SC CNR");

% Plot 4: JSCC CNR
nexttile
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    % plot(iterations, cnr_jsccsd(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_jsccsd(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
legend("D=16mm", "D=20mm", "D=24mm", "D=28mm", "D=32mm", "D=36mm");
% lgd.NumColumns = 2;
% legend('boxoff')
ylim([0 9]);
grid on
grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("JSCC CNR");

%%
f = figure;
f.Position = [100, 100, 450, 300];
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    % plot(iterations, cnr_sc(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_sc(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
% legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=28mm", "D=37mm");
legend("D=5mm", "D=6.5mm", "D=8.5mm", "D=11mm", "D=14mm", "D=18.5mm");
ylim([0 10]);
grid on
grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
% title("SC CNR");

%%
f = figure;
f.Position = [100, 100, 450, 300];
hold on
for i = 1 : 6
    Color_tmp = Color(round((i-1)/6*256)+1, :);

    % plot(iterations, cnr_jsccsd(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_jsccsd(i, :), ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
% legend("D=10mm", "D=13mm", "D=17mm", "D=22mm", "D=28mm", "D=37mm");
legend("D=5mm", "D=6.5mm", "D=8.5mm", "D=11mm", "D=14mm", "D=18.5mm");
% lgd.NumColumns = 2;
% legend('boxoff')
ylim([0 10]);
grid on
grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
% title("JSCC CNR");