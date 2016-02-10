SIFT
----

Scale-invariant feature transform (SIFT) is an algorithm for detecting and describing local features in
images (Lowe, D., IJCV Vol 64(2), pp.91–110, 2004). Key stages for SIFT are the following: 
(i) scale-space extrema detection, 
(ii) keypoint localization, 
(iii) Orientation assignment, and 
(iv) keypoint descriptor. 
The program implements SIFT and shows the results for given images SIFT-test1.png and SIFT-test2.png.

STEPS:
------
1. Point of interests are called keypoints in the SIFT framework. For an input image I(x; y), we implement Difference of Gaussian (DoG)
2. The convolved images are grouped by octave (an octave is constructed from set of images convolved with different values of sigma)
3. Then the DoG images
are taken from adjacent Gaussian-blurred images per octave. Once DoG images have been obtained, keypoints
are identified as local minima/maxima of the DoG images across scales. This is done by comparing each pixel in the DoG images to its eight neighbors at the same scale and nine corresponding neighboring pixels in each of the neighbouring scales. If the pixel value is the maximum or minimum among all compared pixels, it is selected as a candidate keypoint.
4. This stage attempts to eliminate more points from the list of keypoints
by finding those that have low contrast or are poorly localized on an edge. Low contrast based removal: for
each candidate keypoint, interpolation of nearby data is used to accurately determine its position. The initial approach was to just locate each keypoint at the location and scale of the candidate keypoint.
To eliminate extrema based on poor localization, it is noted that in these cases there is a large principle
curvature across the edge but a small curvature in the perpendicular direction of the DoG function. If
this difference is less than the ratio of largest to smallest eigenvector, from the Hessian matrix (H) at the
location and scale of the keypoint, the keypoint is rejected.
5. Each keypoint is assigned one or more orientations based on local image
gradient directions. This is the key step in achieving invariance to rotation as the keypoint descriptor can be represented relative to this orientation and therefore achieve invariance to image rotation. After computing gradient magnitude m and orientation theta.
6. The local gradient data, used above, is also used to create keypoint descriptors.
The gradient information is rotated to line up with the orientation of the keypoint and then weighted by a Gaussian with variance of 1.5 times sigma. This data is then used to create a set of histograms over a window centered on the keypoint. Keypoint descriptors typically uses a set of 16 histograms, aligned in a 4x4 grid, each with 8 orientation bins. This results in a feature vector containing 128 elements.

These resulting vectors are known as SIFT keys. I have used the SIFT-input1.png and SIFT-input2.png
images to identify SIFT keys.
Notice that SIFT-input2.png image is rotated, scaled, translated version of the SIFT-input1.png file, and there are some illumination differences as well.