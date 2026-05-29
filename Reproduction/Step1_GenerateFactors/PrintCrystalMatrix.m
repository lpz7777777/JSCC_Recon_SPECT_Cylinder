%% PrintCrystalMatrix.m
% =========================================================================
% 功能：将 CrystalMatrix 从 .mat 二进制格式导出为 .txt 文本格式
%       方便查看和验证晶体矩阵的内容
% =========================================================================
%
% 输入文件（需在 Data/ 目录中）：
%   - CrystalMatrix_20250307_JSCCGC_32x64x4.mat
%     包含变量 CrystalMatrix（32×64×4 的三维数组）
%     值含义：0=空, 1=前3层NaI晶体, 2=后1层CdZnTe晶体
%
% 输出文件：
%   - CrystalMatrix.txt（制表符分隔的文本文件）
%
% 使用方法：
%   1. 确保 Data/ 目录中有 CrystalMatrix_20250307_JSCCGC_32x64x4.mat
%   2. 在 MATLAB 中运行本脚本
%   3. 检查生成的 CrystalMatrix.txt

%% 加载晶体矩阵
% 从 .mat 文件加载 CrystalMatrix 变量
% 命名规则：20250307 = 日期，JSCCGC = JSCC Geant4 Compton
% 32x64x4 = X方向32个单元 × Y方向64个单元 × Z方向4层
load('Data/CrystalMatrix_20250307_JSCCGC_32x64x4.mat');

%% 导出为文本文件
% reshape 将三维矩阵展平为行向量
% writematrix 写入文本文件，使用制表符作为分隔符
writematrix(reshape(CrystalMatrix, 1, []), 'Data/CrystalMatrix.txt', 'Delimiter', 'tab');

%% 显示基本信息
fprintf('CrystalMatrix 尺寸: [%d, %d, %d]\n', size(CrystalMatrix, 1), size(CrystalMatrix, 2), size(CrystalMatrix, 3));
fprintf('总单元数: %d\n', numel(CrystalMatrix));
fprintf('=0 (空): %d\n', sum(CrystalMatrix(:) == 0));
fprintf('=1 (NaI前3层): %d\n', sum(CrystalMatrix(:) == 1));
fprintf('=2 (CdZnTe后1层): %d\n', sum(CrystalMatrix(:) == 2));
fprintf('已导出到 Data/CrystalMatrix.txt\n');