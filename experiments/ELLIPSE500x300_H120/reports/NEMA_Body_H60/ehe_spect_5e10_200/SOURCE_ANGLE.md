# Source-angle and dose audit

All 20 registered and deployed macros were read byte-for-byte. Their only
commands are `/xcat/clear`, `/xcat/centerY`, `/xcat/angle`, `/xcat/add` and
`/run/beamOn`. They contain no `/gps/ang` or other hemisphere restriction.
`/xcat/angle` rotates source positions through 0, 18, …, 342 degrees; it does
not restrict photon momentum directions.

The actual accepted executable has SHA256
`a49fbdef69b187df2cfebfb481c7974394d00264ab2d4aeb4cc83e65eeacb882`.
Its frozen/current `PrimaryGeneratorAction.cc` SHA256 is
`21a0a1ee1c743f76c786c7f3f44dd1a68d2d4e9b9807f29731f2170f7ea1d89e`.
The active Xcat branch uses `G4ParticleGun(1)` and samples
`cos(theta)=2*U-1`, `phi=2*pi*U`, setting the momentum direction anew for
every primary. This is a uniform full-sphere 4π distribution. The GPS
fallback is not the source branch selected by these macros.

Therefore the registered 1000 workers × 50,000,000 beamOn events mean
50,000,000,000 actual gamma primaries and the same full-4π-equivalent dose.
The multiplier is **1**, not 2. The earlier actual 5e9 acquisition used
the same full-sphere generator and source commands, with 25,000,000 events
per worker.

For a separate uniform 2π hemisphere source, an equivalent full-sphere
normalization may require a factor of 2 when the omitted hemisphere and
direction weights justify that equivalence. That is not this acquisition's
source law. No hemisphere setting or normalization was changed in this audit.

`source_angular_audit.json` binds all local/deployed macro bytes, the current
remote generator source, actual accepted binary SHA and the started worker0
Linux macro (only CRLF→LF). This is a source-law/identity audit; it does not
invent runtime direction samples or claim a new physical calibration.
