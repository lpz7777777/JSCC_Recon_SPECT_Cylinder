iter_max = 100;
system("chmod -R a+wrx *");

if exist("./1/CntStat_440.csv", "file")==2
    CntStat_440 = readmatrix("1/CntStat_440.csv");
    CntStat_218 = readmatrix("1/CntStat_218.csv");
    CntStat218_From218 = readmatrix("1/CntStat218_From218.csv");
    CntStat218_From440 = readmatrix("1/CntStat218_From440.csv");
    CntStat440_From440 = readmatrix("1/CntStat440_From440.csv");
    PrimaryCount218 = readmatrix("1/PrimaryCount218.csv");
    PrimaryCount440 = readmatrix("1/PrimaryCount440.csv");
    PrimaryCountOther = readmatrix("1/PrimaryCountOther.csv");
    List = readmatrix("1/List.csv");
end

for iter = 2 : iter_max
    if exist(sprintf("./%d/CntStat_440.csv", iter), "file")==2
        CntStat_tmp = readmatrix(sprintf("%d/CntStat_440.csv", iter));
        CntStat_440 = CntStat_440 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/CntStat_218.csv", iter));
        CntStat_218 = CntStat_218 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/CntStat218_From218.csv", iter));
        CntStat218_From218 = CntStat218_From218 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/CntStat218_From440.csv", iter));
        CntStat218_From440 = CntStat218_From440 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/CntStat440_From440.csv", iter));
        CntStat440_From440 = CntStat440_From440 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/PrimaryCount218.csv", iter));
        PrimaryCount218 = PrimaryCount218 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/PrimaryCount440.csv", iter));
        PrimaryCount440 = PrimaryCount440 + CntStat_tmp;

        CntStat_tmp = readmatrix(sprintf("%d/PrimaryCountOther.csv", iter));
        PrimaryCountOther = PrimaryCountOther + CntStat_tmp;

        List_tmp = readmatrix(sprintf("%d/List.csv", iter));
        List = cat(1, List, List_tmp);
    end
end

filename = mfilename("fullpath");
[path, ~, ~] = fileparts(filename);
foderparts = strsplit(path, filesep);
fodername = foderparts{end};

writematrix(CntStat_440, "CntStat_440.csv");
writematrix(CntStat_218, "CntStat_218.csv");
writematrix(CntStat218_From218, "CntStat218_From218.csv");
writematrix(CntStat218_From440, "CntStat218_From440.csv");
writematrix(CntStat440_From440, "CntStat440_From440.csv");
writematrix(PrimaryCount218, "PrimaryCount218.csv");
writematrix(PrimaryCount440, "PrimaryCount440.csv");
writematrix(PrimaryCountOther, "PrimaryCountOther.csv");
writematrix(List, "List_%s.csv");

