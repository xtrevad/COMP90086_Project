COMP90086 2024 Project Dataset

This dataset includes two image sets:
  train = 7680 images that you can use to develop and train models
  test = 1920 images that make up the Kaggle test set

The train.csv file includes some metadata for the training images:
  shapeset (categorical) = indicates whether the stack contains (1) only cubes or (2) multiple shapes
  type (categorical) = (1) easy or (2) hard
  total_height (integer) = the number of objects in the stack
  instability_type (categorical) = (0) no instability (stable stack), (1) unstable due to unsupported centre of mass, (2) unstable due to stacking on non-planar surface
  cam_angle (categorical) = (1) low or (2) high
  stable_height (integer) = the stable height of the stack

This dataset is adapted from:

O. Groth, F. B. Fuchs, I. Posner, and A. Vedaldi, "Shapestacks: Learning vision-based physical intuition for generalised object stacking," in Computer Vision – ECCV 2018, V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer International Publishing, 2018, pp. 724–739.










