# Tic Tac Toe Game Analysis
The task is to find a winner of a Tic-Tac-Toe game and draw a line that crosses the winning combination
## Solution
I used OpenCV and scikit-image to solve the problem.  
Here are the steps I took with illustrations:
0. Load the image
![Step 0](https://drive.google.com/file/d/1FbsrqZvvsCGHMpHnNWu2R-FEDBs4GB5a/view?usp=drive_link)
1. Segment the image to highlight the game drawing
![Step 1](https://drive.google.com/file/d/1U0EYxoOmFfMvD6tWA6OxCU-z6fRdEYK9/view?usp=drive_link)
2. Convert to grayscale, binarize using thresholding, remove noise
![Step 2](https://drive.google.com/file/d/1qn5RX_hOBz9Pzo2NbmHu3MKKFFaMmiM7/view?usp=drive_link)