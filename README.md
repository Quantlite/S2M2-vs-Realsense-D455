# Intel RealSense D455 vs. S²M²-S: Stereo Depth Evaluation

A practical evaluation project comparing the native depth output of the
**Intel RealSense D455** with depth estimated by **S²M²-S** from the
camera's stereo infrared images.

> **Project status:** Experimental / evaluation in progress. The
> measurements below are preliminary runtime observations, not a
> completed depth-accuracy benchmark.

## Overview

This project investigates whether a software-based stereo matching model
can provide useful depth estimates compared with the D455's native
stereo depth pipeline.

The D455 provides synchronized stereo infrared images and a
hardware-supported depth stream. S²M²-S processes stereo images to
estimate disparity/depth through neural-network inference.

The main goal is to evaluate the trade-off between:

-   Depth accuracy and consistency
-   Fine detail and object boundaries
-   Missing or invalid depth regions
-   Inference latency and frame rate
-   Practical suitability for robotics, obstacle detection, and RGB-D
    SLAM

## Hardware and software

### Hardware

-   Intel RealSense D455 depth camera
-   NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) for model inference
-   Linux development environment

### Software

-   Ubuntu 22.04
-   ROS 2 Humble (for the wider robotics/SLAM workflow)
-   Intel RealSense camera software and ROS 2 camera driver
-   Python and the S²M²-S inference environment
-   PyTorch with CUDA support

Exact package versions, model checkpoints, and camera settings should be
recorded for each reproducible run.

## System comparison

  -----------------------------------------------------------------------
  Feature                 D455 native depth       S²M²-S
  ----------------------- ----------------------- -----------------------
  Depth source            Camera's stereo depth   Software stereo
                          pipeline                matching

  Input                   Stereo IR sensors       Left and right IR
                          (internally processed)  images

  Output                  Depth image from camera Estimated
                                                  disparity/depth

  Processing              Camera/host pipeline    Neural-network
                                                  inference on host

  Main evaluation         Baseline depth stream   Alternative depth
                                                  estimate

  Runtime considerations  Camera stream rate and  Model inference latency
                          USB/host load           and GPU load
  -----------------------------------------------------------------------

The D455 is the baseline sensor in this experiment. S²M²-S is evaluated
as an alternative processing method; it should not be assumed to be more
accurate without ground-truth measurements.

## Repository layout

The following paths were used or discussed during development. Adjust
this section if the repository layout changes.

``` text
s2m2/
├── demo/
│   └── live_d455_s2m2.py
├── README.md
└── ...
```

-   `demo/live_d455_s2m2.py` --- live D455/S²M²-S testing entry point.
-   Model files, configuration files, and dependencies should be
    documented in their respective project directories.

## Preliminary runtime observations

During an initial live test, the following S²M²-S inference timings were
observed:

  Measurement                                            Observed value
  ----------------------------------------- ---------------------------
  First inference                                             1373.2 ms
  Subsequent inference                        approximately 151--152 ms
  Approximate steady-state inference rate         approximately 6.6 FPS

The first inference includes one-time startup effects such as model
initialization and GPU warm-up. The subsequent timing is a more useful
estimate of steady-state model inference.

**These figures are preliminary.** They are not a guaranteed end-to-end
camera-to-depth frame rate. Capture, synchronization, preprocessing,
visualization, and output publishing may add latency. An earlier
preliminary estimate of approximately 2.83 FPS was also discussed; rerun
the benchmark under a documented, consistent setup before reporting a
final FPS figure.

No verified RMSE, MAE, or ground-truth depth-accuracy result is
available yet.

## Evaluation methodology

### 1. Capture synchronized stereo images

Capture the left and right infrared images from the D455 with matching
timestamps. Record:

-   Image resolution and frame rate
-   Exposure and gain settings
-   Camera calibration parameters
-   Timestamp difference between left and right frames
-   Camera firmware and driver versions

Avoid comparing unsynchronized frames because motion between exposures
can distort stereo matching results.

### 2. Obtain both depth outputs

For the same scene and capture interval:

1.  Record the D455 native depth output.
2.  Run S²M²-S on the corresponding left/right IR images.
3.  Convert model output to metric depth using the correct calibration
    and disparity-to-depth conversion.
4.  Apply only documented invalid-depth handling and filtering.
5.  Save raw outputs before visualization or additional filtering.

Do not compare a disparity image directly with a metric depth image.
Ensure both outputs use the same units, coordinate convention,
resolution, and valid-depth mask.

### 3. Measure depth accuracy

Use a known-distance target or a suitable independent depth reference.
Evaluate multiple distances and scene types.

Recommended metrics:

-   **MAE:** Mean absolute depth error.
-   **RMSE:** Root mean square depth error.
-   **Valid-depth ratio:** Percentage of pixels with valid depth.
-   **Temporal stability:** Variation in depth for a stationary scene.
-   **Edge quality:** Preservation of object boundaries and thin
    structures.

Report the evaluated distance range and the number of valid pixels for
every metric.

### 4. Measure runtime performance

For each pipeline, record:

-   Average processing latency
-   Median and 95th-percentile latency
-   Sustained FPS
-   GPU utilization and VRAM usage
-   CPU utilization
-   Dropped frames
-   End-to-end latency from image capture to depth output

Run a warm-up period before timing and use the same input resolution and
test scenes wherever possible.

### 5. Test representative scenes

Include:

-   Textured indoor scenes
-   Low-texture walls and floors
-   Thin objects and object boundaries
-   Near and far objects within the usable range
-   Reflective, dark, or challenging surfaces
-   Static scenes and scenes with motion

Document conditions in which either method produces invalid, noisy, or
unstable depth.

## Running the live test

Activate the Python environment used for S²M²-S and ensure the D455 is
connected and accessible.

Then run:

``` bash
python3 demo/live_d455_s2m2.py
```

The exact environment activation and dependency installation commands
depend on the local setup and should be added once the project's
environment files are finalized.

If the script uses a particular model checkpoint or configuration, make
sure the required files are present and update the script's arguments or
configuration accordingly.

## ROS 2 / SLAM integration considerations

The wider robotics workflow uses ROS 2 Humble and RTAB-Map. Before using
S²M²-S output in a SLAM pipeline, verify:

-   Correct camera intrinsics and stereo calibration
-   Correct depth scale and units
-   Correct image timestamps and synchronization
-   Correct optical frame IDs and TF transforms
-   Consistent image dimensions
-   Appropriate ROS 2 encoding and invalid-depth representation
-   Stable publishing rate and acceptable end-to-end latency

The D455 native depth stream should remain the baseline for mapping and
localization until S²M²-S has been validated for accuracy, temporal
consistency, and real-time performance.

## Results status

  Evaluation item                                Status
  ---------------------------------------------- ----------------------
  D455 live image acquisition                    Used in live testing
  S²M²-S inference                               Tested
  Preliminary inference timing                   Available
  Ground-truth depth comparison                  Not yet completed
  MAE / RMSE benchmark                           Not yet completed
  Valid-depth coverage comparison                Not yet completed
  Long-duration stability test                   Not yet completed
  RTAB-Map comparison using both depth sources   Not yet completed

## Reproducibility checklist

Before publishing benchmark results, record:

-   [ ] Repository commit / model version
-   [ ] Model checkpoint and configuration
-   [ ] Python, PyTorch, CUDA, and driver versions
-   [ ] D455 firmware and camera-driver versions
-   [ ] Image resolution and capture frame rate
-   [ ] Camera calibration and synchronization details
-   [ ] Warm-up duration and benchmark duration
-   [ ] Test scene and target distances
-   [ ] Raw depth outputs and ground-truth measurements
-   [ ] CPU/GPU utilization and dropped-frame counts

## Current conclusion

The D455 native depth stream is the established baseline for this
project. S²M²-S has demonstrated preliminary live inference, with
subsequent inference around 151--152 ms per frame in the observed test.

That corresponds to approximately 6.6 inferences per second, before
accounting for the complete capture-to-output pipeline. This is useful
for an initial feasibility assessment, but it does not establish
superior depth accuracy or suitability for real-time SLAM.

The next step is a controlled, ground-truth-based comparison of depth
accuracy, valid-depth coverage, temporal stability, and end-to-end
runtime.

## Notes

-   Keep raw D455 and S²M²-S outputs so results can be reprocessed
    consistently.
-   Clearly distinguish model inference FPS from end-to-end pipeline
    FPS.
-   Do not claim that S²M²-S outperforms the D455 until a repeatable
    benchmark supports that conclusion.
