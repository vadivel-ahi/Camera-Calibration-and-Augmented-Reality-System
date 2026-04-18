/*
 * Name : Ahilesh Vadivel
 * Date: March 21th 2026
 * 
 * Project 4 - Tasks 4 & 5: Pose Estimation and Project 3D Axes
 *
 * Loads calibration data from camera_params.yml.
 * For each frame, detects the checkerboard, uses solvePnP to compute
 * the board pose, then projects the four outer corners and 3D axes
 * onto the image plane using projectPoints.
 *
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const cv::Size PATTERN_SIZE(9, 6);
const float SQUARE_SIZE = 1.0f;

//Build 3D world coordinates matching the calibration point set.
//Origin at upper-left corner, Y is negative downward, Z toward viewer.
std::vector<cv::Vec3f> buildPointSet() {
    std::vector<cv::Vec3f> pts;
    for (int r = 0; r < PATTERN_SIZE.height; ++r)
        for (int c = 0; c < PATTERN_SIZE.width; ++c)
            pts.push_back(cv::Vec3f(c * SQUARE_SIZE, -r * SQUARE_SIZE, 0.0f));
    return pts;
}

//Draw 3D coordinate axes on the board origin using projectPoints.
//X = red, Y = green, Z = blue. Length is in board square units.
void drawAxes(cv::Mat& frame,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const cv::Mat& cam_mat, const cv::Mat& dist,
    float length = 3.0f) {
    //Define axis endpoint vectors in 3D world space
    std::vector<cv::Vec3f> axis_pts = {
        {0, 0, 0},           // origin
        {length, 0, 0},      // X axis end
        {0, -length, 0},     // Y axis end (negative because Y is downward)
        {0, 0, length}       // Z axis end (toward viewer)
    };

    //Project 3D axis points to 2D image coordinates
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(axis_pts, rvec, tvec, cam_mat, dist, img_pts);

    cv::Point2f origin = img_pts[0];
    cv::line(frame, origin, img_pts[1], cv::Scalar(0, 0, 255), 3);   // X = red
    cv::line(frame, origin, img_pts[2], cv::Scalar(0, 255, 0), 3);   // Y = green
    cv::line(frame, origin, img_pts[3], cv::Scalar(255, 0, 0), 3);   // Z = blue

    cv::putText(frame, "X", img_pts[1], cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(frame, "Y", img_pts[2], cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, "Z", img_pts[3], cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(255, 0, 0), 2);
}

//Draw the four outer corners of the board using projectPoints.
void drawOuterCorners(cv::Mat& frame,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const cv::Mat& cam_mat, const cv::Mat& dist) {
    //Four outer corner positions in 3D world space
    std::vector<cv::Vec3f> corner_pts = {
        {0, 0, 0},                                               // top-left
        {(PATTERN_SIZE.width - 1) * SQUARE_SIZE, 0, 0},         // top-right
        {0, -(PATTERN_SIZE.height - 1) * SQUARE_SIZE, 0},       // bottom-left
        {(PATTERN_SIZE.width - 1) * SQUARE_SIZE,
         -(PATTERN_SIZE.height - 1) * SQUARE_SIZE, 0}           // bottom-right
    };

    //Project the 3D corner points to 2D image coordinates
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(corner_pts, rvec, tvec, cam_mat, dist, img_pts);

    //Draw circles at each projected corner
    for (int i = 0; i < 4; i++) {
        cv::circle(frame, img_pts[i], 8, cv::Scalar(0, 255, 255), -1);
    }

    //Draw lines connecting the four corners to form a rectangle
    cv::line(frame, img_pts[0], img_pts[1], cv::Scalar(0, 255, 255), 2);
    cv::line(frame, img_pts[0], img_pts[2], cv::Scalar(0, 255, 255), 2);
    cv::line(frame, img_pts[1], img_pts[3], cv::Scalar(0, 255, 255), 2);
    cv::line(frame, img_pts[2], img_pts[3], cv::Scalar(0, 255, 255), 2);
}

int main() {
    //Load calibration data from Task 2/3
    cv::Mat camera_matrix, dist_coeffs;
    cv::FileStorage fs("camera_params.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open camera_params.yml. Run calibration first.\n";
        return -1;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    std::cout << "Calibration data loaded.\n";

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { std::cerr << "Cannot open camera\n"; return -1; }

    cv::Mat frame, gray;
    std::vector<cv::Vec3f> point_set = buildPointSet();

    const cv::TermCriteria termCrit(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    std::cout << "Press 'q' to quit.\n\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        //Detect checkerboard corners
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

            //Compute board pose
            cv::Mat rvec, tvec;
            cv::solvePnP(point_set, corner_set,
                camera_matrix, dist_coeffs,
                rvec, tvec);

            //Print pose to console
            std::cout << "Rotation:\n" << rvec << "\n";
            std::cout << "Translation:\n" << tvec << "\n\n";

            //Project and draw 3D axes at board origin
            drawAxes(frame, rvec, tvec, camera_matrix, dist_coeffs);

            //Project and draw the four outer corners of the board
            drawOuterCorners(frame, rvec, tvec, camera_matrix, dist_coeffs);

            //Overlay translation values on frame
            cv::putText(frame,
                "tx: " + std::to_string(tvec.at<double>(0)),
                { 20, 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
            cv::putText(frame,
                "ty: " + std::to_string(tvec.at<double>(1)),
                { 20, 75 }, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
            cv::putText(frame,
                "tz: " + std::to_string(tvec.at<double>(2)),
                { 20, 110 }, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::putText(frame, "Board NOT detected",
                { 20, 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Tasks 4&5 - Pose and Projection", frame);
        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}