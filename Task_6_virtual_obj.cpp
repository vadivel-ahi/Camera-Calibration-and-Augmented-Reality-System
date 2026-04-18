/*
 * Name : Ahilesh Vadivel
 * Date: March 25th 2026
 * 
 * Project 4 - Task 6: Create a Virtual Object
 *
 * Builds on Tasks 4 & 5. Projects a 3D rocket ship made of lines
 * floating above the checkerboard. The rocket has a rectangular body,
 * a nose cone, and two asymmetric fins in different colors.
 * 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const cv::Size PATTERN_SIZE(9, 6);
const float SQUARE_SIZE = 1.0f;

//Build 3D world coordinates matching the calibration point set
std::vector<cv::Vec3f> buildPointSet() {
    std::vector<cv::Vec3f> pts;
    for (int r = 0; r < PATTERN_SIZE.height; ++r)
        for (int c = 0; c < PATTERN_SIZE.width; ++c)
            pts.push_back(cv::Vec3f(c * SQUARE_SIZE, -r * SQUARE_SIZE, 0.0f));
    return pts;
}

//Helper: project a single 3D point to 2D image space
cv::Point2f projectPt(const cv::Vec3f& p,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const cv::Mat& cam, const cv::Mat& dist) {
    std::vector<cv::Vec3f> in = { p };
    std::vector<cv::Point2f> out;
    cv::projectPoints(in, rvec, tvec, cam, dist, out);
    return out[0];
}

// Helper: draw a line between two 3D world points projected to image space
void drawLine3D(cv::Mat& frame,
    const cv::Vec3f& a, const cv::Vec3f& b,
    const cv::Scalar& color, int thickness,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const cv::Mat& cam, const cv::Mat& dist) {
    cv::Point2f pa = projectPt(a, rvec, tvec, cam, dist);
    cv::Point2f pb = projectPt(b, rvec, tvec, cam, dist);
    cv::line(frame, pa, pb, color, thickness);
}

// Draw the rocket ship virtual object floating above the board.
//
// The rocket is centered at (4, -2.5) in board space (middle of the board).
// Positive Z floats above the board toward the viewer.
//
// Structure:
//   - Rectangular body from Z=1 to Z=4  (cyan)
//   - Nose cone from Z=4 to apex at Z=6  (yellow)
//   - Left fin at X=3 side               (red)
//   - Right fin at X=5 side              (green, different shape = asymmetric)
void drawRocket(cv::Mat& frame,
    const cv::Mat& rvec, const cv::Mat& tvec,
    const cv::Mat& cam, const cv::Mat& dist) {
    cv::Scalar cyan(255, 255, 0);
    cv::Scalar yellow(0, 255, 255);
    cv::Scalar red(0, 0, 255);
    cv::Scalar green(0, 255, 0);

    //Body corners - bottom face (Z=1)
    cv::Vec3f b0(3, -2, 1), b1(5, -2, 1), b2(5, -3, 1), b3(3, -3, 1);
    //Body corners - top face (Z=4)
    cv::Vec3f t0(3, -2, 4), t1(5, -2, 4), t2(5, -3, 4), t3(3, -3, 4);
    //Nose cone apex
    cv::Vec3f apex(4, -2.5, 6);

    //Bottom face of body
    drawLine3D(frame, b0, b1, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b1, b2, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b2, b3, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b3, b0, cyan, 2, rvec, tvec, cam, dist);

    // Top face of body
    drawLine3D(frame, t0, t1, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t1, t2, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t2, t3, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t3, t0, cyan, 2, rvec, tvec, cam, dist);

    //Vertical edges of body
    drawLine3D(frame, b0, t0, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b1, t1, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b2, t2, cyan, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, b3, t3, cyan, 2, rvec, tvec, cam, dist);

    //Nose cone - 4 lines from top face corners to apex
    drawLine3D(frame, t0, apex, yellow, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t1, apex, yellow, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t2, apex, yellow, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, t3, apex, yellow, 2, rvec, tvec, cam, dist);

    //Left fin - triangle on the X=3 side (red)
    cv::Vec3f lf0(3, -2.5, 1);   // base center at body bottom
    cv::Vec3f lf1(2, -2.5, 1);   // fin tip outward
    cv::Vec3f lf2(3, -2.5, 2);   // fin tip upward
    drawLine3D(frame, lf0, lf1, red, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, lf1, lf2, red, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, lf2, lf0, red, 2, rvec, tvec, cam, dist);

    //Right fin - wider triangle on X=5 side (green, different shape)
    cv::Vec3f rf0(5, -2.5, 1);   // base center at body bottom
    cv::Vec3f rf1(6.5, -2.5, 1); // fin tip outward (wider than left fin)
    cv::Vec3f rf2(5, -2.5, 2.5); // fin tip upward (taller than left fin)
    drawLine3D(frame, rf0, rf1, green, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, rf1, rf2, green, 2, rvec, tvec, cam, dist);
    drawLine3D(frame, rf2, rf0, green, 2, rvec, tvec, cam, dist);
}

int main() {
    //Load calibration data
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

    std::cout << "Press 'q' to quit.\n";

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

            //Compute board pose
            cv::Mat rvec, tvec;
            cv::solvePnP(point_set, corner_set,
                camera_matrix, dist_coeffs,
                rvec, tvec);

            //Draw the virtual rocket above the board
            drawRocket(frame, rvec, tvec, camera_matrix, dist_coeffs);

        }
        else {
            cv::putText(frame, "Board NOT detected",
                { 20, 40 }, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Task 6 - Virtual Rocket", frame);
        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}