#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to detect a color in the given frame and return bounding boxes
vector<Rect> detectColor(const Mat& hsvFrame, Scalar lowerBound, Scalar upperBound) {
    Mat mask;
    inRange(hsvFrame, lowerBound, upperBound, mask);

    // Find contours to detect significant color regions
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> boundingBoxes;
    for (const auto& contour : contours) {
        // Ignore small regions
        if (contourArea(contour) > 500) {
            boundingBoxes.push_back(boundingRect(contour));
        }
    }
    return boundingBoxes;
}

int main(int argc, char** argv) {
    // Open external camera (device 0 is usually the default webcam)
    VideoCapture cap(0); // 0 for default webcam, change if needed for external cam
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    // Define color ranges in HSV for red, yellow, and green
    Scalar lowerRed1(0, 120, 70), upperRed1(10, 255, 255);   // Lower red range
    Scalar lowerRed2(170, 120, 70), upperRed2(180, 255, 255); // Upper red range
    // Adjusted yellow range to be more sensitive to the yellow light
    Scalar lowerYellow(15, 100, 100), upperYellow(35, 255, 255); // Expanded yellow range
    Scalar lowerGreen(40, 50, 50), upperGreen(90, 255, 255); // Green range

    Mat frame, hsvFrame;
    vector<Mat> hsvChannels(3); // H, S, V channels
    
    while (true) {
        cap >> frame; // Read a frame from the camera
        if (frame.empty()) break; // Exit if no frame is received

        // Convert to HSV color space
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
        split(hsvFrame, hsvChannels);

        // Apply histogram equalization to the V channel (brightness)
        equalizeHist(hsvChannels[2], hsvChannels[2]);

        // Merge the channels back into the hsvFrame
        merge(hsvChannels, hsvFrame);

        // Apply Gaussian Blur to reduce noise
        GaussianBlur(hsvFrame, hsvFrame, Size(5, 5), 0);

        // Detect colors
        vector<Rect> redRegions = detectColor(hsvFrame, lowerRed1, upperRed1);
        vector<Rect> redRegions2 = detectColor(hsvFrame, lowerRed2, upperRed2);
        vector<Rect> yellowRegions = detectColor(hsvFrame, lowerYellow, upperYellow);
        vector<Rect> greenRegions = detectColor(hsvFrame, lowerGreen, upperGreen);

        // Combine redRegions and redRegions2
        redRegions.insert(redRegions.end(), redRegions2.begin(), redRegions2.end());

        // Draw rectangles and display text for each detected color
        bool anyLightDetected = false;

        if (!redRegions.empty()) {
            for (const auto& box : redRegions) {
                rectangle(frame, box, Scalar(0, 0, 255), 2); // Red rectangle
            }
            putText(frame, "Red Light", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2); // Red text
            anyLightDetected = true;
        }
        if (!yellowRegions.empty()) {
            for (const auto& box : yellowRegions) {
                rectangle(frame, box, Scalar(0, 255, 255), 2); // Yellow rectangle
            }
            putText(frame, "Yellow Light", Point(50, 90), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2); // Yellow text
            anyLightDetected = true;
        }
        if (!greenRegions.empty()) {
            for (const auto& box : greenRegions) {
                rectangle(frame, box, Scalar(0, 255, 0), 2); // Green rectangle
            }
            putText(frame, "Green Light", Point(50, 130), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2); // Green text
            anyLightDetected = true;
        }

        // If no light is detected, display a message
        if (!anyLightDetected) {
            putText(frame, "No Light Detected", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2); // White text
        }

        // Display the processed frame
        imshow("Traffic Light Detection", frame);

        // Break loop on ESC key press
        if (waitKey(30) == 27) break;  // 27 is the ASCII code for the ESC key
    }

    // Release the video capture object and close all windows
    cap.release();
    destroyAllWindows();
    return 0;
}
