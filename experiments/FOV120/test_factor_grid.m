function test_factor_grid()
% Small analytic matrix catches grid, density weighting and detector translation.
    repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(fullfile(repo,'Auxiliary_Studies', ...
        'GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main','GenFactors'));
    root = fullfile(repo,'experiments','FOV120','generated','factor_grid_test');
    for nz = [20,40]
        folder = fullfile(root,sprintf('z%d',nz));
        if ~isfolder(folder); mkdir(folder); end
        det = zeros(4,12,'single');
        det(:,2) = [30;60;90;120]; det(:,12) = 1;
        fid=fopen(fullfile(folder,'Params_Detector.dat'),'wb');
        fwrite(fid,[single(4);reshape(det.',[],1)],'single'); fclose(fid);
        fid=fopen(fullfile(folder,'Params_Image.dat'),'wb');
        fwrite(fid,[51 51 nz 6 6 3 1 0 0 0 0 170],'single'); fclose(fid);
        z = ((0:nz-1)-(nz-1)/2)*3;
        [xx,yy,zz]=ndgrid(-150:6:150,-150:6:150,z);
        values=single(1+xx/1000+yy/2000+zz/3000);
        values=repmat(values,1,1,1,4);
        fid=fopen(fullfile(folder,'analytic.sysmat'),'wb'); fwrite(fid,values,'single'); fclose(fid);
        options=struct('include_center_point',true,'apply_polar_volume_weighting',true, ...
            'write_cartesian_tmp',false,'z_axis',z);
        out=fullfile(folder,'Factors');
        gen_factors(440,fullfile(folder,'analytic.sysmat'), ...
            fullfile(folder,'Params_Detector.dat'),out,'',struct('enabled',false),options);
        coor=readmatrix(fullfile(out,'coor_polar_full.csv'));
        assert(size(coor,1)==1281*nz);
        volume=readmatrix(fullfile(out,'polar_cell_volume_mm3.csv'));
        assert(abs(sum(volume)-pi*153^2*nz*3)<1e-6);
        detector=readmatrix(fullfile(out,'Detector.csv'));
        assert(isequal(detector(:,3),[200;230;260;290]));
        fid=fopen(fullfile(out,'SysMat_polar'),'rb');
        matrix=reshape(fread(fid,inf,'single=>single'),4,[]); fclose(fid);
        expected=(1+coor(:,1)/1000+coor(:,2)/2000+coor(:,3)/3000).*volume;
        assert(max(abs(double(matrix(1,:).')-expected)./expected)<1e-6);
        rot=readmatrix(fullfile(out,'RotMat_full.csv'));
        inv=readmatrix(fullfile(out,'RotMatInv_full.csv'));
        for v=1:20
            assert(isequal(rot(inv(:,v),v),(1:numel(volume)).'));
        end
    end
    fprintf('PASS: 20/40 layers, density integral, detector shift, rotations, analytic interpolation.\n');
end
