function [coeff, density_g_cm3] = material_db(material, energy_keV)
% MATERIAL_DB 衰减系数查表 [mu_total, mu_PE, mu_Compton]，单位 1/mm（线性衰减系数）。
%
% 约定（与现有 CUDA 引擎一致）：
%   mu_total（不含瑞利/相干散射）= mu_PE + mu_Compton
%
% 输入：
%   material    - 材料名：'NaI','GAGG','Pb','W','Vacuum'
%   energy_keV  - 光子能量
% 输出：
%   coeff = [mu_total, mu_PE, mu_Compton]  （1/mm）
%
% 数据来源：physics_data/nist_xcom_materials_1_1000keV.csv。
%   该表由 download_nist_xcom.py 从 NIST XCOM 官方 CGI 下载，包含每个整数
%   能量 1..1000 keV 的光电与非相干（Compton）质量相互作用系数，并按材料
%   密度转换为 1/mm。本函数对非整数能量做线性插值。
%
% 密度：
%   NaI  3.67 g/cm^3   GAGG 6.60 g/cm^3   Pb  11.35 g/cm^3   W  19.35 g/cm^3
%   Vacuum: 无衰减（准直器选 Vacuum 等效于无准直器，光子自由穿过）

	if strcmp(material, 'Vacuum')
		coeff = [0.0, 0.0, 0.0];
		density_g_cm3 = 0.0;
		return;
	end
	if ~isscalar(energy_keV) || ~isfinite(energy_keV) || energy_keV < 1 || energy_keV > 1000
		error('material_db: energy %.9g keV is outside the NIST XCOM table range 1..1000 keV.', energy_keV);
	end

	[energy_axis, linear_coefficients] = load_xcom_table();
	switch material
		case 'NaI'
			columns = [1, 2];
			density_g_cm3 = 3.67;
		case 'GAGG'
			columns = [3, 4];
			density_g_cm3 = 6.60;
		case 'Pb'
			columns = [5, 6];
			density_g_cm3 = 11.35;
		case 'W'
			columns = [7, 8];
			density_g_cm3 = 19.35;
		otherwise
			error('material_db: unknown material "%s". Supported: NaI, GAGG, Pb, W, Vacuum.', material);
	end

	mu_pe = interp1(energy_axis, linear_coefficients(:, columns(1)), energy_keV, 'linear');
	mu_compton = interp1(energy_axis, linear_coefficients(:, columns(2)), energy_keV, 'linear');
	coeff = [mu_pe + mu_compton, mu_pe, mu_compton];
end

function [energy_axis, linear_coefficients] = load_xcom_table()
	persistent cached_energy cached_coefficients
	if isempty(cached_energy)
		this_dir = fileparts(mfilename('fullpath'));
		csv_path = fullfile(fileparts(this_dir), 'physics_data', ...
			'nist_xcom_materials_1_1000keV.csv');
		if ~isfile(csv_path)
			error('material_db: NIST XCOM table not found: %s', csv_path);
		end
		raw = readmatrix(csv_path, 'NumHeaderLines', 4);
		if size(raw, 1) ~= 1000 || size(raw, 2) ~= 17 || any(raw(:, 1) ~= (1:1000).')
			error('material_db: malformed NIST XCOM table: %s', csv_path);
		end
		cached_energy = raw(:, 1);
		cached_coefficients = raw(:, [4, 5, 8, 9, 12, 13, 16, 17]);
	end
	energy_axis = cached_energy;
	linear_coefficients = cached_coefficients;
end
