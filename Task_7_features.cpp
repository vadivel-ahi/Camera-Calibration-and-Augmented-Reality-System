/*
 * Name: Ahilesh Vadivel
 * Date: March 27th 2026
 * 
 * Project 4 - Task 7: Detect Robust Features (Harris Corners)
 *
 * Detects Harris corners in a live video stream.
 * Press '+' and '-' to adjust the detection threshold in real time.
 * Press 'q' to quit.
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame, gray, gray_f, harris_response, harris_norm;

    //Harris corner detection parameters
    int block_size = 2;    // neighborhood size for corner detection
    int aperture = 3;    // aperture size for Sobel derivative
    double k = 0.04; // Harris detector free parameter
    double threshold = 150; // response threshold for marking corners

    std::cout << "Controls:\n"
        << "  '+' - increase threshold (fewer corners)\n"
        << "  '-' - decrease threshold (more corners)\n"
        << "  'q' - quit\n\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        //Convert to grayscale and then to float for Harris
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray_f, CV_32F);

        //Compute Harris corner response for every pixel
        //harris_response is a float map where high values = strong corners
        cv::cornerHarris(gray_f, harris_response, block_size, aperture, k);

        //Normalize response to 0-255 range for thresholding and display
        cv::normalize(harris_response, harris_norm, 0, 255,
            cv::NORM_MINMAX, CV_32F);

        //Count and draw detected corners above threshold
        int corner_count = 0;
        for (int r = 0; r < harris_norm.rows; ++r) {
            for (int c = 0; c < harris_norm.cols; ++c) {
                if (harris_norm.at<float>(r, c) > threshold) {
                    //Draw a small circle at each detected corner
                    cv::circle(frame, cv::Point(c, r), 5,
                        cv::Scalar(0, 0, 255), 1);
                    corner_count++;
                }
            }
        }

        //Display current threshold and corner count on frame
        cv::putText(frame,
            "Threshold: " + std::to_string((int)threshold),
            { 20, 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 0), 2);
        cv::putText(frame,
            "Corners: " + std::to_string(corner_count),
            { 20, 80 }, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 0), 2);

        cv::imshow("Task 7 - Harris Corner Detection", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q') break;

        //Adjust threshold interactively
        if (key == '+' || key == '=') {
            threshold = std::min(threshold + 10.0, 255.0);
            std::cout << "Threshold: " << threshold << "\n";
        }
        if (key == '-') {
            threshold = std::max(threshold - 10.0, 1.0);
            std::cout << "Threshold: " << threshold << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}