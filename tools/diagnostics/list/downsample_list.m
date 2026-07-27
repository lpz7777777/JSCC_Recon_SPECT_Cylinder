folderPath = uigetdir("./List/", "选择要缩减的 List 文件夹");
if isequal(folderPath, 0)
    return;
end

prompt = {sprintf("输入缩减比例 (当前行数 × 该比例，范围 (0, 1])："), ...
          sprintf("输出文件夹名称（留空则自动添加 _ds 后缀）：")};
dlgtitle = "List 下采样参数";
dims = [1, 50];
definput = {"0.5", ""};
answer = inputdlg(prompt, dlgtitle, dims, definput);
if isempty(answer)
    return;
end

ds = str2double(answer{1});
if isnan(ds) || ds <= 0 || ds > 1
    error("缩减比例必须在 (0, 1] 范围内，当前输入: %s。", answer{1});
end

[~, srcName] = fileparts(folderPath);
if isempty(strtrim(answer{2}))
    outputPath = fullfile(fileparts(folderPath), sprintf("%s_ds%g", srcName, ds));
else
    outputPath = fullfile(fileparts(folderPath), strtrim(answer{2}));
end

if exist(outputPath, "dir")
    warning("输出文件夹已存在，文件将被覆盖: %s", outputPath);
else
    mkdir(outputPath);
end

csvFiles = dir(fullfile(folderPath, "*.csv"));
if isempty(csvFiles)
    error("所选文件夹中没有找到 CSV 文件: %s", folderPath);
end

rng(20260416);

totalOriginal = 0;
totalSampled = 0;

for fileIdx = 1:numel(csvFiles)
    srcFile = fullfile(folderPath, csvFiles(fileIdx).name);
    dstFile = fullfile(outputPath, csvFiles(fileIdx).name);

    data = readmatrix(srcFile, "Delimiter", ",");
    if isempty(data)
        writematrix(data, dstFile, "Delimiter", ",");
        fprintf("  %s: 空文件，跳过\n", csvFiles(fileIdx).name);
        continue;
    end

    numRows = size(data, 1);
    numKeep = max(1, round(numRows * ds));
    if numKeep >= numRows
        writematrix(data, dstFile, "Delimiter", ",");
        fprintf("  %s: %d 行 (ds=1，全部保留)\n", csvFiles(fileIdx).name, numRows);
        totalOriginal = totalOriginal + numRows;
        totalSampled = totalSampled + numRows;
        continue;
    end

    selectedIdx = sort(randperm(numRows, numKeep));
    dataSampled = data(selectedIdx, :);
    writematrix(dataSampled, dstFile, "Delimiter", ",");

    totalOriginal = totalOriginal + numRows;
    totalSampled = totalSampled + numKeep;
    fprintf("  %s: %d -> %d 行 (%.1f%%)\n", csvFiles(fileIdx).name, numRows, numKeep, 100 * numKeep / numRows);
end

fprintf("\n===== 完成 =====\n");
fprintf("源文件夹:   %s (%d 行)\n", folderPath, totalOriginal);
fprintf("输出文件夹: %s (%d 行, 比例 %.4f)\n", outputPath, totalSampled, totalSampled / totalOriginal);