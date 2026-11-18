# FAST MULTIGRID-BASED COMPUTATION

## INTRODUCTION

E-field simulation for Transcranial Magnetic Stimulation (TMS) is an important problem. Medical professionals need accurate E-field data to verify beforehand the placement of TMS coil. Existing E-field simulation pipeline is time-consuming and computationally expensive to conduct. It takes on average 5 minutes or more to compute the E-field, given a coil position and head model. We decides to bridge the gap by incorporating the fast multigrid-based computation proposed by Ilkka Laakso and Akimasa Hirata in their paper "Fast multigrid-based computation of the induced
electric field for transcranial magnetic stimulation".

## GUIDE

1. Ensure that you have CMake installed and ready.

2. Run the following command in the working directory.

```bash
cmake -S . -B build -DENABLE_OPENMP=ON -DENABLE_ZLIB=ON
cmake --build build --config Release -j
```

3. Follow the format to run on any patient T1 .nii.gz and coil .ccd input:

```bash
.\build\Release\tms_fem_cli.exe --input path\to\file --coil coils/Magstim_70mm_Fig8.ccd --scalp-target -38,-22,58 --scalp-offset 5.0 --handle-yaw 45 --res 1.0 --output-dir path\to\directory
```

## ACKNOWLEDGEMENT

Our algorithm made extensive use of the ideas outlined by the following paper, Laakso, Ilkka; Hirata, Akimasa . (2012). Fast multigrid-based computation of the induced electric field for transcranial magnetic stimulation. Physics in Medicine and Biology, 57(23), 7753–7765.