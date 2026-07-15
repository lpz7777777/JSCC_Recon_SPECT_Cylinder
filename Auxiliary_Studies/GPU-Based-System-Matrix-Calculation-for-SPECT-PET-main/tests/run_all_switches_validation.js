#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const projectRoot = path.resolve(root, '..', '..');
const gpu = process.env.SMOKE_TEST_GPU || '0';
const peBinary = process.env.PEGEN_BINARY
  || path.join(root, 'PEGen_RayTracing_CircularHole', 'PEGen_CircularHole');
const scatterBinary = process.env.SCATTERGEN_BINARY
  || path.join(root, 'ScatterGen_RayTracing_CircularHole', 'ScatterGen_CircularHole');

function timestamp() {
  const now = new Date();
  const pieces = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '_',
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ];
  return pieces.join('');
}

const outputDirectory = process.env.SCATTER_VALIDATION_OUTPUT
  ? path.resolve(process.env.SCATTER_VALIDATION_OUTPUT)
  : path.join(
    projectRoot,
    'run_logs',
    `ScatterSurface_AllSwitches_Validation_${timestamp()}`,
  );
if (fs.existsSync(outputDirectory)) {
  throw new Error(`Validation output already exists: ${outputDirectory}`);
}
fs.mkdirSync(outputDirectory, { recursive: true });

function writeFloat32(filename, values) {
  const data = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => data.writeFloatLE(value, index * 4));
  fs.writeFileSync(path.join(outputDirectory, filename), data);
}

function readFloat32(filename) {
  const data = fs.readFileSync(path.join(outputDirectory, filename));
  const values = [];
  for (let offset = 0; offset < data.length; offset += 4) {
    values.push(data.readFloatLE(offset));
  }
  return values;
}

function run(binary, args, logName, extraEnvironment = {}) {
  const result = spawnSync(binary, args, {
    cwd: outputDirectory,
    encoding: 'utf8',
    env: { ...process.env, ...extraEnvironment },
  });
  const log = `${result.stdout || ''}${result.stderr || ''}`;
  fs.writeFileSync(path.join(outputDirectory, logName), log, 'utf8');
  if (result.status !== 0) {
    throw new Error(`${binary} failed (${result.status}); see ${logName}.`);
  }
  return log;
}

// One physical Pb layer with one circular hole exercises the collimator path.
const collimator = new Array(109).fill(0);
collimator[0] = 1;
collimator[10] = 1;
collimator[11] = 20;
collimator[12] = 10;
collimator[13] = 20;
collimator[14] = 0;
collimator[15] = 0.20753475;
collimator[16] = 0.127347;
collimator[17] = 0.08018775;
collimator[100] = 0;
collimator[101] = -5;
collimator[102] = 5;
collimator[103] = 0;
collimator[104] = 2.5;

// GAGG and W records at 440 keV exercise active-crystal and high-Z sources.
const detectors = [2];
detectors.push(
  0, 10, 0,
  4, 10, 4,
  0.07060950, 0.01782144, 0.05278806,
  0.14009656, 0, 1,
);
detectors.push(
  5, 10, 0,
  4, 10, 4,
  0.29623571, 0.15717921, 0.13905650,
  0.14009656, 0, 2,
);

writeFloat32('Params_Collimator.dat', collimator);
writeFloat32('Params_Detector.dat', detectors);
writeFloat32('Params_Image.dat', [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 20]);
writeFloat32('Params_Physics.dat', [
  1, 1, 1, 1,
  1, 196.30538, 239.69462, 440,
  1, 1, 1, 1,
]);

const peLog = run(peBinary, ['-cuda', gpu], 'pegen.log');
const scatterEnvironment = {
  SCATTER_WRITE_COMPONENTS: '1',
  SCATTER_STRUCTURED_TRAVERSAL: '1',
  SCATTER_KINEMATIC_PRUNING: '1',
  SCATTER_COMPTON_INTEGRAND_LUT: '1',
  SCATTER_TARGET_FACE_SUBDIV: '2',
  SCATTER_NEAR_TARGET_FACE_SUBDIV: '8',
  SCATTER_NEAR_TARGET_DISTANCE_FACTOR: '2.0',
  SCATTER_CRYSTAL_CHUNK: '2',
  DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS: '17',
  DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES: '96',
  DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES: '96',
};
const scatterLog = run(
  scatterBinary,
  ['-PE', 'PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat', '-cuda', gpu],
  'scattergen.log',
  scatterEnvironment,
);

const scatter = readFloat32(
  'Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat',
);
const components = {
  intercrystal: readFloat32('C_intercrystal.sysmat'),
  highZToCrystal: readFloat32('C_highZ_to_crystal.sysmat'),
  localRecoil: readFloat32('C_local_recoil.sysmat'),
  localSelfPhotoelectric: readFloat32('C_local_self_photoelectric.sysmat'),
  collimatorToCrystal: readFloat32('C_collimator_to_crystal.sysmat'),
  total: readFloat32('C_total.sysmat'),
};
const componentNames = Object.keys(components).filter((name) => name !== 'total');
let maximumTotalDifference = 0;
let maximumComponentClosureError = 0;
scatter.forEach((value, index) => {
  if (!(value >= 0) || !Number.isFinite(value)) {
    throw new Error(`Invalid scatter value at ${index}: ${value}`);
  }
  const componentSum = componentNames.reduce(
    (sum, name) => sum + components[name][index],
    0,
  );
  maximumTotalDifference = Math.max(
    maximumTotalDifference,
    Math.abs(value - components.total[index]),
  );
  maximumComponentClosureError = Math.max(
    maximumComponentClosureError,
    Math.abs(value - componentSum),
  );
});
if (maximumTotalDifference > 2e-7 || maximumComponentClosureError > 2e-7) {
  throw new Error(
    `Component closure failed: total=${maximumTotalDifference}, sum=${maximumComponentClosureError}`,
  );
}

const summary = {
  status: 'PASS',
  gpu,
  sourceEnergyKeV: 440,
  forcedWindowKeV: [196.30538, 239.69462],
  detectorCount: 2,
  imageVoxelCount: 1,
  physicsFlags: {
    compton: 1,
    savePE: 1,
    saveScatter: 1,
    saveCombined: 1,
    forcedWindow: 1,
    buildCrystalGeometry: 1,
    buildCollimatorGeometry: 1,
    localRecoil: 1,
    localSelfPhotoelectric: 1,
  },
  environment: scatterEnvironment,
  scatter,
  components,
  maximumTotalDifference,
  maximumComponentClosureError,
  logChecks: {
    peCompleted: peLog.includes('Energy-windowed Photon Electric Sysmat Written.'),
    surfaceKernel: scatterLog.includes('Launching crystalScatterSurfaceSysMatCuda'),
    componentMatrices: scatterLog.includes('Diagnostic component matrices: enabled'),
    nearQuadrature: scatterLog.includes('near=8x8'),
    mixedMaterials: scatterLog.includes('GAGG=1') && scatterLog.includes('W=1'),
  },
  outputDirectory,
};
if (Object.values(summary.logChecks).some((value) => !value)) {
  throw new Error(`Missing validation log marker: ${JSON.stringify(summary.logChecks)}`);
}
fs.writeFileSync(
  path.join(outputDirectory, 'validation_summary.json'),
  `${JSON.stringify(summary, null, 2)}\n`,
  'utf8',
);
console.log(JSON.stringify(summary));
