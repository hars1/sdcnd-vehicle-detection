## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_car.png
[image3]: ./output_images/hog-non-car.png
[image4]: ./output_images/box_sliding_window.png
[image5]: ./output_images/sliding_windows_scaled.png
[image6]: ./output_images/heatmap_before.png
[image7]: ./output_images/heatmap_after.png
[image8]: ./output_images/processed_images.png
[image9]: ./output_images/autti.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

![alt text][image3]

#### Data Augmentation
Additional data from https://github.com/udacity/self-driving-car/tree/master/annotations was used to add more training and test examples.  Sample data from the data set is shown below

![alt text][image9]
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and fed the data extracted to Linear SVM classifier. I realized during the iterative process that the selection of the parameters affects one major factor: *The number of features per image* and this has impact on two things:

* The **speed** of the classifier
* The **accuracy** of the classifier

When you extract many features there is a good chance the **accuracy** gets higher but the **speed** of the prediction is increased. By trial and error using values close to the ones suggested during the lesson I found a combination of parameters that the training accuracy is on an adequate level but the prediction time for a sample is not very long so we simulate real-time prediction circumstances.

The parameter set I ended up with is:

| Parameter  | Value |
|:----------:|:-----:|
| `colorspace` | YUV |
| `orient` | 11 |
| `pix_per_cell`| (16, 16)|
| `cell_per_block` | (2, 2) |
| `hog_channel`    | ALL |
| `spatial_size`      | (16, 16) |
| `hist_bins`          | 32|

With this parameter set the size of feature vector is 2052. That means that each image is described by 2052 features which is a sensible number for a classifier running on contemporary hardware. On my machine it took 426.82  seconds to extract all the features for the complete dataset.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I created the dataset based on the script we used during the lesson. and it is located in the **Augment with Autti dataset ** section of the notebook. Data was augmented for both car and non car images from Autti dataset.
I created a `y` label vector with **1** for the car images and **0** for the non-car images.  Then I combined the `X` data from `car_features` and `noncar_features` and I scaled them using the `StandardScaler` and then split the dataset on train and test set.

| Set | Car Images | Non Car Images |
|:---:|:----------:|:--------------:|
| `X_train` | 11012   |  38497 |
| `X_test`  | 2763   |  9615 |


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used at first hog subsampling technique introduced on the lesson and the code is located in **HOG subsampling** section of the notebook. I adapted the function `find_cars` in order to return the bounding boxes. The method originally combines the HOG feature extraction with a sliding window search. The HOG feature extraction is done on the selected region of image. In my case I removed the upper part of the image since there is not chance to meet cars in the sky(at least for the time being...). Then these image features are subsampled based on the `window_size` and also on that stage the color histogram and spatial binning features are extracted. Then the combined features are fed to the classifier. The method returns a list of the regions that the classifier predicted the existence of a car.

Here is a first example of the method applied on the test images for a scale of 1.0 and 50% window overlap.

![alt text][image4]

Then I decided to create a method `find_car_windows` which calls the `find_cars` method but for each sub region of the image it uses different scaling. This simulates the fact that the cars which are closer to the camera appear larger than the ones that are distant. In that way and exploring different scalings I ended up with a list of subregions and a respective list of scales which are fed to `find_cars` in order to predict for cars. The result of this can be shown in the image below:

![alt text][image5]

As it was obvious some of the false positives were removed but some remained. In order to get rid of them I used heatmap and thresholding in order to filter them out.

The heatmap before thrsholding:

![alt text][image6]

And after applying threshold with value **1**

![alt text][image7]

Then the `draw_labeled_boxes` function get the thresholded heatmap and using the results of `scipy.ndimage.measurements.label()` which collects spatially contiguous areas of the heatmap and assigns each one a label, draws the final boxes.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the end the result of all the test images using the final method of `processImages` which gets the image and draws the boxes around the cars provided the following result.

![alt text][image8]

The function combines all the methods created on the steps above and given the classifier the scaler and a certain threshold produces the result. From the previously discussed parameters I didn't have to change anything so far cause the result of the images was satisfying.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./videos_output/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for processing the video is located in **Video Pipeline** section of the notebook. Here a class `VideoProcessor` is implemented with `processFrame` function which is an altered version of previously defined `processImage`. In the function the target was to keep track of the heatmaps of a certain number of previous frames. This is because I wanted to increase the threshold of the heatmaps in order to get rid of false positives the first time I ran the videos. By using a buffer I was able to make sure the algorithm can have the threshold increased while still keeping track of the cars reliably.





---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First the tradeoff balance between the accuracy and the speed of the prediction of the classifier was challenging. The tradeoff between accuracy and speed was the biggest problem. So the search for the parameters used to extract the features was a very time consuming process.The implemenation worked well with images, but with video the detection can be improved . Given the accuracy one can understand that you can't avoid false positives in the frames.

In order to scan in detail each frame with bigger window overlap would increase the processing power making the algorithm not be able to perform real-time without adequate computing power. The heatmap buffer provides on one hand security that the car will be consistently being tracked but on the other hand problems may occur in high dynamic behaviors. This is something that can be further examined and tuned even better. Perhaps a different approach would work better.
Training with more data might improve the detection. Deep learning based approaches can be tried for better detection
