Data Directory
==============

Place your calibration and test images here.

Expected structure for the end-to-end pipeline:

  data/
    calibration/       Single-camera calibration target photos
    sfm/               Multi-view scene photos for SfM (Camera 1)
    inference/         Single-view photo for pose estimation (Camera 2)
    markers/           Generated ArUco marker sheets for printing

p1.png is a sample SLS circular-dot calibration target image (1920x1200).
It contains an 11x9 grid of white circles on a dark background with 5
larger fiducial circles for grid identification.
