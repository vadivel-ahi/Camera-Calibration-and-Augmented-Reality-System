/*
 * Name: Ahilesh Vadivel
 * Date: March 19th 2026
 * 
 * Project 4 - Tasks 1, 2 & 3: Select Calibration Images and Calibrate Camera
 *
 * Press 's' to save the current frame for calibration (board must be detected).
 * Press 'c' to run calibration (requires at least 5 saved frames).
 * Press 'w' to write calibration results to camera_params.yml.
 * Press 'q' to quit.
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const cv::Size PATTERN_SIZE(9, 6);
const float SQUARE_SIZE = 1.0f;

//Build 3D world coordinates for checkerboard corners.
//Origin (0,0,0) is the upper-left internal corner.
//X increases to the right, Y decreases downward (Z points toward viewer).
std::vector<cv::Vec3f> buildPointSet() {
    std::vector<cv::Vec3f> pts;
    for (int r = 0; r < PATTERN_SIZE.height; ++r)
        for (int c = 0; c < PATTERN_SIZE.width; ++c)
            pts.push_back(cv::Vec3f(c * SQUARE_SIZE, -r * SQUARE_SIZE, 0.0f));
    return pts;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame, gray;

    //Get frame size for camera matrix initialization
    cap >> frame;
    if (frame.empty()) { std::cerr << "Cannot read frame\n"; return -1; }

    //point_list: 3D world points for each saved frame
    //corner_list: 2D image corners for each saved frame
    std::vector<cv::Vec3f> point_set = buildPointSet();
    std::vector<std::vector<cv::Vec3f>>  point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    //Store last successfully detected corners and frame
    std::vector<cv::Point2f> last_corners;
    cv::Mat last_frame;
    bool last_found = false;

    //Initialize camera matrix with estimated values
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        1, 0, frame.cols / 2.0,
        0, 1, frame.rows / 2.0,
        0, 0, 1);

    //Distortion coefficients: length 5 enables k1, k2, p1, p2, k3
    cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    bool calibrated = false;
    double reprojection_error = 0.0;

    const cv::TermCriteria termCrit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    std::cout << "Controls:\n"
        << "  's' - save current frame for calibration\n"
        << "  'c' - run calibration (need at least 5 frames)\n"
        << "  'w' - write calibration results to camera_params.yml\n"
        << "  'q' - quit\n\n";

    std::cout << "Initial camera matrix:\n" << camera_matrix << "\n\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect and refine corners
        std::vector<cv::Point2f> corner_set;
        bool found = cv::findChessboardCorners(
            gray, PATTERN_SIZE, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH |
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK);

        if (found) {
            cv::cornerSubPix(gray, corner_set,
                cv::Size(11, 11), cv::Size(-1, -1), termCrit);
            cv::drawChessboardCorners(frame, PATTERN_SIZE, corner_set, found);

            // Save last good detection for use when 's' is pressed
            last_corners = corner_set;
            last_frame = frame.clone();
            last_found = true;
        }
        else {
            last_found = false;
        }

        //Display status on frame
        std::string status = found ? "Board detected" : "Board NOT detected";
        cv::putText(frame, status, { 20, 40 },
            cv::FONT_HERSHEY_SIMPLEX, 0.9,
            found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::putText(frame,
            "Saved frames: " + std::to_string(corner_list.size()),
            { 20, 80 }, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(255, 200, 0), 2);
        if (calibrated) {
            cv::putText(frame,
                "RMS: " + std::to_string(reprojection_error) + " px",
                { 20, 120 }, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("Tasks 2&3 - Calibration", frame);
        char key = static_cast<char>(cv::waitKey(30));

        //Save last good frame when 's' is pressed
        if (key == 's') {
            if (last_found) {
                corner_list.push_back(last_corners);
                point_list.push_back(point_set);

                //Save the image to disk with corners drawn
                std::string filename = "calib_" +
                    std::to_string(corner_list.size() - 1) + ".jpg";
                cv::imwrite(filename, last_frame);

                std::cout << "Frame saved. Total: " << corner_list.size()
                    << " | Image: " << filename << "\n";
            }
            else {
                std::cout << "No valid board detected. Frame not saved.\n";
            }
        }

        //Run calibration when 'c' is pressed
        if (key == 'c') {
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 frames (have "
                    << corner_list.size() << ")\n";
            }
            else {
                std::vector<cv::Mat> rvecs, tvecs;

                //Reset camera matrix to initial estimate before calibrating
                camera_matrix = (cv::Mat_<double>(3, 3) <<
                    1, 0, frame.cols / 2.0,
                    0, 1, frame.rows / 2.0,
                    0, 0, 1);
                dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

                std::cout << "\nCamera matrix before calibration:\n"
                    << camera_matrix << "\n\n";

                reprojection_error = cv::calibrateCamera(
                    point_list, corner_list,
                    frame.size(),
                    camera_matrix, dist_coeffs,
                    rvecs, tvecs,
                    cv::CALIB_FIX_ASPECT_RATIO);

                calibrated = true;

                std::cout << "Camera matrix after calibration:\n"
                    << camera_matrix << "\n\n";
                std::cout << "Distortion coefficients:\n"
                    << dist_coeffs << "\n\n";
                std::cout << "RMS reprojection error: "
                    << reprojection_error << " px\n\n";
            }
        }

        //Write calibration results to file when 'w' is pressed
        if (key == 'w') {
            if (!calibrated) {
                std::cout << "Run calibration first (press 'c').\n";
            }
            else {
                cv::FileStorage fs("camera_params.yml",
                    cv::FileStorage::WRITE);
                fs << "camera_matrix" << camera_matrix;
                fs << "dist_coeffs" << dist_coeffs;
                fs << "rms_error" << reprojection_error;
                fs.release();
                std::cout << "Parameters saved to camera_params.yml\n";
            }
        }

        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}