# Tic Tac Toe Game Analysis
The task is to find a winner of a Tic-Tac-Toe game and draw a line that crosses the winning combination
## Solution
I used OpenCV and scikit-image to solve the problem.  
Here are the steps I took with illustrations:  
0. Load the image
![Step 0](steps_merged/step0.jpg)
1. Segment the image (using scikit-image's `clear_border()`) to highlight the game drawing
![Step 1](steps_merged/step1.png)
2. Convert to grayscale, binarize using thresholding, remove noise
![Step 2](steps_merged/step2.png)
3. Find contours
![Step 3](steps_merged/step3.png)
4. Find lines using Hough Lines on skeletonized image. Then select only 4 lines out of many overlapping ones. And find the intersection points.
![Step 4](steps_merged/step4.png)
5. Roughly estimate the grid cells, where X and O should be. It doesn't have to be very accurate, but the centers of X or O should lie inside predicted cells.
![Step 5](steps_merged/step5.png)
6. Out of all contours find the ones that correspond to X, O and their positions. Then determine whether it is X or O.  
The main properties to tell them apart are:  
The convexity of a contour (area of the contour / area of the convex hull) - O is most likely close to convex and X is not.  
The distance from the center of mass to the closest point of the contour. It can be calculated using `cv2.pointPolygonTest` and it is low for X and large for O.  
If based on contours we cannot tell it is X or O, we crop the cell (in step 5 we estimated the cells), apply projective transformation and then recognize it as X or O or None.
![Step 6](steps_merged/step6.png)
7. Find a winning combination and draw a line.
![Step 7](steps_merged/step7.png)
# Structure
```
├── README.md  
├── requirements.txt  
├── steps_merged - merged images of steps  
├── tic_images - images of games and the outputs produced by tic_tac_toe.py
│   ├── tic_01.jpg
│   ├── tic_01_output.png
│   ├── tic_02.jpg
│   ├── tic_02_output.png
│   ├── tic_03.jpg
│   ├── tic_03_output.png
│   ├── tic_04.jpg
│   ├── tic_04_output.png
│   ├── tic_05.jpg
│   ├── tic_05_output.png
│   ├── tic_06.jpg
│   ├── tic_06_output.png
│   ├── tic_07.jpg
│   ├── tic_07_output.png
│   ├── tic_08.jpg
│   ├── tic_08_output.png
│   ├── tic_09.jpg
│   ├── tic_09_output.png
│   ├── tic_10.jpg
│   ├── tic_10_output.png
│   ├── tic_11.jpg
│   ├── tic_11_output.png
│   ├── tic_12.jpg
│   └── tic_12_output.png
├── tic_tac_toe.ipynb - Jupyter Notebook with the ability to look at results step by step
└── tic_tac_toe.py - Python file with the same code as in Jupyter Notebook, but collected in a function
```
# How to use
You can use the `tic_tac_toe` function of `tic_tac_toe.py` to get the output image with a line drawn.  
Example:
```
tic_tac_toe('tic_images/tic_12.jpg')
```
It will create the output image (with name `tic_12_output`) in the same folder. 
