#!/usr/bin/env node

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const gpu = process.env.SMOKE_TEST_GPU || '0';
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'spect-scatter-rotation-'));

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

try {
  writeFloat32('Params_Collimator.dat', [0]);
  writeFloat32('Params_Detector.dat', [
    1,
    0, 10, 0,
    4, 10, 4,
    0.0938786, 0.0580227, 0.0358559,
    0.199033216, 0, 1,
  ]);
  writeFloat32('Params_Image.dat', [
    1, 1, 1,
    1, 1, 1,
    2, Math.PI,
    0, 0, 0,
    20,
  ]);
  writeFloat32('Params_Physics.dat', [
    1, 1, 1, 0,
    0, 0, 0, 218,
    1, 1,
    0, 1,
  ]);

  // One detector x one voxel x two rotations. The synthetic PE input
  // isolates slice selection: rotation 0 is zero and rotation 1 is nonzero.
  writeFloat32('two_rotation_pe.sysmat', [0, 1e-4]);

  const binary = path.join(
    root, 'ScatterGen_RayTracing_CircularHole', 'ScatterGen_CircularHole',
  );
  const result = spawnSync(binary, [
    '-PE', 'two_rotation_pe.sysmat', '-cuda', gpu,
  ], {
    cwd: work,
    encoding: 'utf8',
    env: {
      ...process.env,
      SCATTER_CRYSTAL_CHUNK: '1',
      DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS: '5',
      DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES: '24',
      DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES: '24',
    },
  });
  if (result.status !== 0) {
    throw new Error(`ScatterGen failed (${result.status})\n${result.stdout}\n${result.stderr}`);
  }

  const scatter = readFloat32(
    'Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat',
  );
  if (scatter.length !== 2) {
    throw new Error(`expected two rotation elements, received ${scatter.length}`);
  }
  if (scatter[0] !== 0) {
    throw new Error(`rotation 0 must remain zero, received ${scatter[0]}`);
  }
  if (!(scatter[1] > 0) || !Number.isFinite(scatter[1])) {
    throw new Error(`rotation 1 must use its nonzero PE slice, received ${scatter[1]}`);
  }

  console.log(JSON.stringify({
    status: 'PASS',
    gpu,
    peByRotation: [0, 1e-4],
    scatterByRotation: scatter,
  }));
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
