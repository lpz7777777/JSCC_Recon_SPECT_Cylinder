#!/usr/bin/env node

const fs = require("fs");

const [path, expectedElementsText, minimumMaxText = "0"] = process.argv.slice(2);
if (!path || !expectedElementsText) {
    console.error("Usage: validate_sysmat_values.js FILE EXPECTED_ELEMENTS [MINIMUM_MAX]");
    process.exit(2);
}

const expectedElements = Number(expectedElementsText);
const minimumMax = Number(minimumMaxText);
if (!Number.isSafeInteger(expectedElements) || expectedElements <= 0 || !Number.isFinite(minimumMax)) {
    console.error("Invalid expected element count or minimum maximum value.");
    process.exit(2);
}

const stat = fs.statSync(path);
if (stat.size !== expectedElements * 4) {
    console.error(`${path}: expected ${expectedElements * 4} bytes, found ${stat.size}`);
    process.exit(1);
}

const fd = fs.openSync(path, "r");
const buffer = Buffer.allocUnsafe(16 * 1024 * 1024);
let position = 0;
let count = 0;
let nonzero = 0;
let minimum = Infinity;
let maximum = -Infinity;
let invalid = 0;
let negative = 0;

try {
    while (position < stat.size) {
        const bytesRead = fs.readSync(fd, buffer, 0, Math.min(buffer.length, stat.size - position), position);
        if (bytesRead <= 0) break;
        for (let offset = 0; offset < bytesRead; offset += 4) {
            const value = buffer.readFloatLE(offset);
            count += 1;
            if (!Number.isFinite(value)) {
                invalid += 1;
                continue;
            }
            if (value < 0) negative += 1;
            if (value !== 0) nonzero += 1;
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
        position += bytesRead;
    }
} finally {
    fs.closeSync(fd);
}

const result = { path, count, nonzero, minimum, maximum, invalid, negative };
console.log(JSON.stringify(result));

if (count !== expectedElements || invalid > 0 || negative > 0 || nonzero === 0 || maximum < minimumMax) {
    process.exit(1);
}
