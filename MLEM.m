clear
parallel.gpu.enableCUDAForwardCompatibility(true);
fid = fopen("SysMat_polar", "r");
SysMat = fread(fid, "float32");
fclose(fid);
C_ij_SC = reshape(SysMat, [], 1160*20);

%%
a_tmp = C_ij_SC * ones(1160*20, 1) * 3.64e11/(1160*20);
a = readmatrix("./CntStat_Point_Phantom_3D_511_3.64e11.csv").';
c = 1.25 * a ./ a_tmp;
figure;plot(c);
C_ij_SC = C_ij_SC .* c;
b = sum(C_ij_SC, 1);
fid = fopen("Sensi_s", "w");
fwrite(fid, b, "float32");
fclose(fid);

fid = fopen("SysMat", "w");
fwrite(fid, C_ij_SC, "float32");
fclose(fid);

%%

Name1 = "HotPoints_1e9";
% Name2 = "Array_10mm_1e8";
% Name3 = "25_15_1e6";
% Name1 = "15_-15_1e6";

% CntStat_1 = load(sprintf("CntStat_%s.mat", Name1)).P.';
CntStat_1 = readmatrix("CntStat.csv");
% CntStat_2 = readmatrix(sprintf(".\\CntStat\\CntStat_%s.csv", Name2));
% CntStat_3 = readmatrix(sprintf(".\\CntStat\\CntStat_%s.csv", Name3));
% CntStat_4 = readmatrix(sprintf(".\\CntStat\\CntStat_%s.csv", Name4));
% CntStat = CntStat_1 + CntStat_2;
CntStat = CntStat_1;
% CntStat = CntStat(6:10, :);
CntStat = sum(CntStat, 1);

Flag = (CntStat == 0);
C_ij_SC(Flag, :) = [];
CntStat(Flag) = [];

Iteration = 500000;
ScRepeaNum = 1;
SubSetNumSC = 1;
ScinNum = length(CntStat);
SubSetSize_SC = ScinNum / SubSetNumSC;

Id_SC = zeros(SubSetNumSC, 2);
Id_SC(:, 1) = round(1 : SubSetSize_SC : ((SubSetNumSC-1)*SubSetSize_SC + 1));
Id_SC(1:(SubSetNumSC-1), 2) = Id_SC(2:SubSetNumSC, 1)-1;
Id_SC(SubSetNumSC, 2) = ScinNum;
Sum_C_ij_SC = sum(C_ij_SC, 1);

PixelNumU = 100;
PixelNumV = 100;
PixelLengthU = 5;
PixelLengthV = 5;


% Image_SC = ones(Iteration+1, PixelNumU*PixelNumV);            % 自准直成像

Image = ones(1, PixelNumU*PixelNumV);

Image = gpuArray(Image);
% Image_SC = gpuArray(Image_SC);
CntStat = gpuArray(CntStat);
Sum_C_ij_SC = gpuArray(Sum_C_ij_SC);
C_ij_SC = gpuArray(C_ij_SC);

tic
for i = 1 : Iteration
    Image = Image .* ((CntStat ./ (C_ij_SC * Image.').') * C_ij_SC) ./ Sum_C_ij_SC;

    % Image_SC(i+1, :) = Image_SC(i, :);
    % 
    % for j = 1 : SubSetNumSC
    %     Range_SC = Id_SC(j, 1) : Id_SC(j, 2);
    %     Image_SC(i+1, :) = Image_SC(i+1, :) .* ((CntStat(Range_SC) ./ ((C_ij_SC(Range_SC, :) * (Image_SC(i+1, :).')).') * C_ij_SC(Range_SC, :)) ./ Sum_C_ij_SC);
    % end
end
toc
%% figure
color = 1 - gray(256);
Cut_Range = 5;
range_U = (1 + Cut_Range) : (PixelNumU - Cut_Range);
range_V = (1 + Cut_Range) : (PixelNumV - Cut_Range);
Min_X = (1/2 + Cut_Range - PixelNumU/2) * PixelLengthU;
Max_X = -Min_X;
Min_Y = Min_X;
Max_Y = Max_X;
a = reshape(Image, PixelNumU, PixelNumV);
a = a(range_U, range_V);
figure
imagesc([Min_X Max_X], [Min_Y Max_Y],  a);
axis square
% axis off
colormap(color);
colorbar;
title(sprintf("Iteration: %d   TotalCounts: %d", Iteration, sum(CntStat)));