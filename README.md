# Tic Tac Toe Game Analysis
The task is to find a winner of a Tic-Tac-Toe game and draw a line that crosses the winning combination
## Solution
I used OpenCV and scikit-image to solve the problem.  
Here are the steps I took with illustrations:  
0. Load the image
![Step 0](steps_merged/step0.jpg)
1. Segment the image to highlight the game drawing
![Step 1](steps_merged/step1.png)
2. Convert to grayscale, binarize using thresholding, remove noise
![Step 2](steps_merged/step2.png)