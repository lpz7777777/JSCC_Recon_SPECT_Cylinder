# Full-FOV uniform sources for Compton sensitivity

These two macros generate the Geant4 `List.csv` inputs used to estimate
density-basis `Sensi_d` for the current JSCC polar grid.

Geometry:

```text
Geant4 center:       (0, -245, 0) mm
source radius:       153 mm
source full height:  60 mm
source volume:       pi * 153^2 * 60 = 4412492.545673008 mm3
emission direction:  isotropic
```

The radius is 153 mm, rather than the outer sample-center radius of 150 mm,
because the `r=150 mm` cells have midpoint boundaries at 147 and 153 mm. The
height similarly follows the complete cell boundaries `z=[-30,30] mm`.

Run 218 and 440 keV separately. The production `List.csv` does not retain a
reliable primary-energy label, so a mixed 218+440 run cannot be separated into
the two monoenergetic sensitivity inputs after simulation. Run each macro in a
clean directory because `Geant4Code` appends to `List.csv`.

The supplied `/run/beamOn 1000000000` means one billion emitted photons of the
macro's single energy. It is not a number of 225Ac decays. Independent workers
may run the same macro with independent random seeds; sum their beamOn counts
when supplying `--source-photons`, and concatenate or jointly pass all List CSV
files to the sensitivity tool.

Do not add a water or PMMA cylinder unless the system matrix is regenerated
with the same attenuating material. Here `Cylinder` defines the GPS source
distribution only; the Geant4 material environment remains the production
JSCC environment.
