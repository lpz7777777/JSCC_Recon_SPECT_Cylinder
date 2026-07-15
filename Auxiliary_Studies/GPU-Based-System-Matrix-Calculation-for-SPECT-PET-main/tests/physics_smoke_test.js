#!/usr/bin/env node

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const gpu = process.env.SMOKE_TEST_GPU || '0';
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'spect-physics-smoke-'));
const peBinary = process.env.PEGEN_BINARY
  || path.join(root, 'PEGen_RayTracing_CircularHole', 'PEGen_CircularHole');
const scatterBinary = process.env.SCATTERGEN_BINARY
  || path.join(root, 'ScatterGen_RayTracing_CircularHole', 'ScatterGen_CircularHole');

function writeFloat32(filename, values) {
  const data = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => data.writeFloatLE(value, index * 4));
  fs.writeFileSync(path.join(work, filename), data);
}

function readFloat32(filename) {
  const data = fs.readFileSync(path.join(work, filename));
  const values = [];
  for (let offset = 0; offset < data.length; offset += 4) {
    values.push(data.readFloatLE(offset));
  }
  return values;
}

function run(binary, args, extraEnv = {}) {
  const result = spawnSync(binary, args, {
    cwd: work,
    encoding: 'utf8',
    env: {
      ...process.env,
      SCATTER_CRYSTAL_CHUNK: '2',
      SCATTER_WRITE_COMPONENTS: '1',
      SCATTER_TARGET_FACE_SUBDIV: '2',
      SCATTER_NEAR_TARGET_FACE_SUBDIV: '4',
      DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS: '9',
      DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES: '48',
      DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES: '48',
      ...extraEnv,
    },
  });
  if (result.status !== 0) {
    throw new Error(`${binary} failed (${result.status})\n${result.stdout}\n${result.stderr}`);
  }
  return `${result.stdout}\n${result.stderr}`;
}

function runScatter(physics, extraEnv = {}) {
  writeFloat32('Params_Physics.dat', physics);
  const log = run(
    scatterBinary,
    ['-PE', 'PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat', '-cuda', gpu],
    extraEnv,
  );
  return {
    log,
    scatter: readFloat32('Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat'),
    combined: readFloat32('SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat'),
    components: {
      intercrystal: readFloat32('C_intercrystal.sysmat'),
      highZ: readFloat32('C_highZ_to_crystal.sysmat'),
      localRecoil: readFloat32('C_local_recoil.sysmat'),
      localSelf: readFloat32('C_local_self_photoelectric.sysmat'),
      collimator: readFloat32('C_collimator_to_crystal.sysmat'),
      total: readFloat32('C_total.sysmat'),
    },
  };
}

function writeDetectors(materials, relativeFwhm) {
  const detectors = [materials.length];
  materials.forEach(([muTotal, muPe, muCompton], index) => {
    detectors.push(
      index * 5, 10, 0,
      4, 10, 4,
      muTotal, muPe, muCompton,
      relativeFwhm, 0, 1,
    );
  });
  writeFloat32('Params_Detector.dat', detectors);
}

function maximumLocalDifference(lhs, rhs, baseline) {
  let maximumAbsolute = 0;
  let maximumRelative = 0;
  lhs.forEach((value, index) => {
    const lhsLocal = value - baseline[index];
    const rhsLocal = rhs[index] - baseline[index];
    const absolute = Math.abs(lhsLocal - rhsLocal);
    const relative = absolute / Math.max(Math.abs(rhsLocal), 1e-12);
    maximumAbsolute = Math.max(maximumAbsolute, absolute);
    maximumRelative = Math.max(maximumRelative, relative);
  });
  return { maximumAbsolute, maximumRelative };
}

try {
	const mixedJscc = process.env.SMOKE_TEST_MIXED_JSCC === '1';
  const collimator = new Array(109).fill(0);
  collimator[0] = 1;
  collimator[10] = 1;
  collimator[11] = 20;
  collimator[12] = 10;
  collimator[13] = 20;
  collimator[14] = 0;
  collimator[15] = 0.86451;
  collimator[16] = 0.76481;
  collimator[17] = 0.09970;
  collimator[100] = 0;
  collimator[101] = -5;
  collimator[102] = 5;
  collimator[103] = 0;
  collimator[104] = 2.5;

		const detectorMaterials = mixedJscc
			? [[0.18336, 0.11634, 0.06702], [1.15215, 0.97859, 0.17356]]
			: [[0.09389, 0.05804, 0.03586], [0.09389, 0.05804, 0.03586]];

	  writeFloat32('Params_Collimator.dat', collimator);
	  writeDetectors(detectorMaterials, 0.199033214);
  writeFloat32('Params_Image.dat', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 20]);
  const directPhysics = [1, 1, 1, 1, 0, 0, 0, 218, 1, 1, 0, 0];
  writeFloat32('Params_Physics.dat', directPhysics);

  const peLog = run(peBinary, ['-cuda', gpu]);
  const raw = readFloat32('PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat');
  const windowed = readFloat32('PE_Windowed_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat');
  if (raw.length !== 2 || windowed.length !== 2) throw new Error('unexpected PE matrix size');

  const expectedAcceptance = 0.760968;
  raw.forEach((value, index) => {
    if (!(value >= 0) || !Number.isFinite(value)) throw new Error(`invalid raw PE value ${value}`);
    if (value > 0) {
      const ratio = windowed[index] / value;
      if (Math.abs(ratio - expectedAcceptance) > 2e-6) {
        throw new Error(`PE window acceptance mismatch: ${ratio}`);
      }
    }
  });

	  const legacy = runScatter(directPhysics);
	  const comptonDisabled = runScatter([0, ...directPhysics.slice(1, 10), 1, 1]);
	  const selfEnabled = runScatter([...directPhysics.slice(0, 10), 0, 1]);
	  const selfBins17 = runScatter([...directPhysics.slice(0, 10), 0, 1], {
	    DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS: '17',
	  });
	  const selfBins33 = runScatter([...directPhysics.slice(0, 10), 0, 1], {
	    DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS: '33',
	  });
  const recoilWindowPhysics = [1, 1, 1, 1, 1, 40, 100, 218, 1, 1, 1, 0];
  const recoilEnabled = runScatter(recoilWindowPhysics);
  const allEnabled = runScatter([...directPhysics.slice(0, 10), 1, 1]);

  const scatterLog = allEnabled.log;
  const scatter = allEnabled.scatter;
  const combined = allEnabled.combined;
  scatter.forEach((value, index) => {
    if (!(value >= 0) || !Number.isFinite(value)) throw new Error(`invalid scatter value ${value}`);
    const expected = Math.fround(windowed[index] + value);
    if (combined[index] !== expected) {
      throw new Error(`combined mismatch at ${index}: ${combined[index]} != ${expected}`);
    }
    const components = allEnabled.components;
    const componentSum = components.intercrystal[index]
      + components.highZ[index]
      + components.localRecoil[index]
      + components.localSelf[index]
      + components.collimator[index];
    if (Math.abs(components.total[index] - value) > 2e-7
        || Math.abs(componentSum - value) > 2e-7) {
      throw new Error(`scatter component mismatch at ${index}: total=${value}, componentSum=${componentSum}`);
    }
  });

	  if (!selfEnabled.scatter.some((value, index) => value > legacy.scatter[index])) {
	    throw new Error(`self Compton+PE switch did not add any response: legacy=${legacy.scatter}, self=${selfEnabled.scatter}`);
	  }
	  if (comptonDisabled.scatter.some((value) => value !== 0)) {
	    throw new Error(`global Compton switch did not suppress local responses: ${comptonDisabled.scatter}`);
	  }
  if (!recoilEnabled.scatter.some((value) => value > 0)) {
    throw new Error(`recoil-escape switch did not produce low-window response: ${recoilEnabled.scatter}`);
  }
  if (!allEnabled.scatter.every((value, index) => value >= selfEnabled.scatter[index])) {
    throw new Error('enabling both local components reduced a scatter element');
  }
	  if (!legacy.log.includes('recoil_escape=0 self_compton_photoelectric=0')) {
    throw new Error('disabled local-scatter switch log missing');
  }
  if (!selfEnabled.log.includes('recoil_escape=0 self_compton_photoelectric=1')) {
    throw new Error('self-photoelectric switch log missing');
  }
  if (!recoilEnabled.log.includes('recoil_escape=1 self_compton_photoelectric=0')) {
    throw new Error('recoil switch log missing');
  }
	  if (!selfEnabled.log.includes('max_partition_error=')) {
	    throw new Error('local-scatter probability partition validation log missing');
	  }
	  if (!comptonDisabled.log.includes('compton=0 recoil_escape=1 self_compton_photoelectric=1')) {
	    throw new Error('global Compton-disable switch log missing');
	  }
	  const lutConvergence = maximumLocalDifference(
	    selfBins17.scatter, selfBins33.scatter, legacy.scatter,
	  );
	  if (lutConvergence.maximumAbsolute > 2e-7
	      || lutConvergence.maximumRelative > 0.02) {
	    throw new Error(`17x17 to 33x33 local-scatter LUT did not converge: ${JSON.stringify(lutConvergence)}`);
	  }

  if (!peLog.includes('Energy-windowed Photon Electric Sysmat Written.')) {
    throw new Error('PE completion marker missing');
  }
	const expectedMaterialLog = mixedJscc
		? 'Detector XCOM materials: NaI=0 GAGG=1 Pb=0 W=1'
		: 'Detector XCOM materials: NaI=2';
  if (!scatterLog.includes(expectedMaterialLog)) {
    throw new Error('XCOM detector material identification failed');
  }
	  if (!scatterLog.includes('physical collimator-volume relationship')) {
	    throw new Error('physical collimator volume path did not execute');
	  }

	  // Source=440 keV with a forced 218 keV window.  Compare every local
	  // component against the same legacy scatter baseline so A recoil and
	  // same-crystal full-energy responses are tested independently.
	  collimator[15] = 0.20753475;
	  collimator[16] = 0.127347;
	  collimator[17] = 0.08018775;
	  writeFloat32('Params_Collimator.dat', collimator);
	  writeDetectors([
	    [0.03690919, 0.00855110, 0.02835809],
	    [0.03690919, 0.00855110, 0.02835809],
	  ], 0.14009656);
	  const physics440 = [1, 1, 1, 1, 0, 0, 0, 440, 1, 1, 0, 0];
	  writeFloat32('Params_Physics.dat', physics440);
	  run(peBinary, ['-cuda', gpu]);

	  const direct440Legacy = runScatter(physics440);
	  const direct440Self = runScatter([...physics440.slice(0, 10), 0, 1]);
	  const crossBase = [1, 1, 1, 1, 1, 196.30538, 239.69462, 440, 1, 1, 0, 0];
	  const crossLegacy = runScatter(crossBase);
	  const crossRecoil = runScatter([...crossBase.slice(0, 10), 1, 0]);
	  const crossSelf = runScatter([...crossBase.slice(0, 10), 0, 1]);
	  const crossRecoilDelta = crossRecoil.scatter.map(
	    (value, index) => value - crossLegacy.scatter[index],
	  );
	  const crossSelfDelta = crossSelf.scatter.map(
	    (value, index) => value - crossLegacy.scatter[index],
	  );
	  if (!crossRecoilDelta.some((value) => value > 0)) {
	    throw new Error(`440 keV recoil did not enter the forced 218 keV window: ${crossRecoilDelta}`);
	  }
	  if (crossSelfDelta.some((value) => Math.abs(value) > 1e-12)) {
	    throw new Error(`440 keV full-energy tail unexpectedly entered the 218 keV window: ${crossSelfDelta}`);
	  }
	  if (!direct440Self.scatter.some(
	    (value, index) => value > direct440Legacy.scatter[index],
	  )) {
	    throw new Error('same-crystal Compton+PE did not enter the direct 440 keV photopeak window');
	  }

  console.log(JSON.stringify({
    status: 'PASS',
    gpu,
	mixedJscc,
    raw,
    windowed,
	    legacyScatter: legacy.scatter,
	    comptonDisabledScatter: comptonDisabled.scatter,
    selfEnabledScatter: selfEnabled.scatter,
	    recoilLowWindowScatter: recoilEnabled.scatter,
	    lutConvergence17To33: lutConvergence,
	    cross440To218RecoilDelta: crossRecoilDelta,
	    cross440To218SelfDelta: crossSelfDelta,
	    direct440SelfDelta: direct440Self.scatter.map(
	      (value, index) => value - direct440Legacy.scatter[index],
	    ),
    scatter,
    combined,
    temporaryDirectory: work,
  }));
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
