OpenFOAM actuator surface simulation
====================================
by [Pete Bachant](http://petebachant.me)

These case files simulate a cylindrical actuator surface--to mimic a vertical axis turbine--in a towing tank.
Note that there may be significant modifications to `processing.py` necessary for it to work on other
systems.

## Installing `swak4Foam` (for post-processing)

For Ubuntu 14.04 and similar:

```
git clone https://github.com/Unofficial-Extend-Project-Mirror/openfoam-extend-Breeder2.0-libraries-swak4Foam swak4Foam
cd swak4Foam
./maintainanceScripts/compileRequirements.sh
```

Add entry to `~/.bashrc` as instructed.

```
source ~/.bashrc
./Allwmake
```
