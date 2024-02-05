// ----> Includes
#include <iostream>
#include <sstream>
#include <string>

#include <chrono>
#include <thread>

// StereoLabs includes
#include "videocapture.hpp"

// OpenCV includes
#include <opencv2/opencv.hpp>

// Sample includes
#include "calibration.hpp"
#include "stopwatch.hpp"
#include "stereo.hpp"
#include "ocv_display.hpp"
// <---- Includes

#define EMBEDDED_ARM  // Low resolution
#define USE_OCV_TAPI // Comment to use "normal" cv::Mat instead of CV::UMat
#define DOWNSCALE_FACTOR 2

int main(int argc, char *argv[])
{
    // ----> Silence unused warning
    (void)argc;
    (void)argv;
    // <---- Silence unused warning

    sl_oc::VERBOSITY verbose = sl_oc::VERBOSITY::INFO;

    int duration = 10;  // recording duration in seconds
    if (argc > 1)
        duration = std::stoi(argv[1]);
    
    std::string recording_name = "recording";
    if (argc > 2)
        recording_name = argv[2];

    // ----> Set Video parameters
    sl_oc::video::VideoParams params;
#ifdef EMBEDDED_ARM
    params.res = sl_oc::video::RESOLUTION::VGA;
#else
    params.res = sl_oc::video::RESOLUTION::HD720;
#endif
    params.verbose = verbose;
    params.fps = sl_oc::video::FPS::FPS_15;

    const int frequency = 15; // 15 Hz
    const std::chrono::milliseconds dt(1000 / frequency);
    const int loop_count = duration * frequency;
    // <---- Set Video parameters

    // ----> Create Video Capture
    sl_oc::video::VideoCapture cap(params);
    if( !cap.initializeVideo(-1))
    {
        std::cerr << "Cannot open camera video capture" << std::endl;
        std::cerr << "See verbosity level for more details." << std::endl;

        return EXIT_FAILURE;
    }
    int sn = cap.getSerialNumber();
    std::cout << "Connected to camera sn: " << sn << std::endl;
    // <---- Create Video Capture

    // ----> Retrieve calibration file from Stereolabs server
    std::string calibration_file;
    // ZED Calibration
    unsigned int serial_number = sn;
    // Download camera calibration file
    if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file))
    {
        std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Calibration file found. Loading..." << std::endl;
    // <---- Retrieve calibration file from Stereolabs server

    // ----> Frame size
    int w,h;
    cap.getFrameSize(w,h);

    int w_out = w / 2 / DOWNSCALE_FACTOR;
    int h_out = h / DOWNSCALE_FACTOR;

    std::cout << "Frame size: " << w_out << ' ' << h_out << std::endl;
    // <---- Frame size

    // ----> Initialize calibration
    double baseline = 0;
    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;
    cv::Mat cameraMatrix_left, cameraMatrix_right;
    
    sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), map_left_x, map_left_y, map_right_x, map_right_y,
                                  cameraMatrix_left, cameraMatrix_right, &baseline);

    double fx = cameraMatrix_left.at<double>(0,0);
    double fy = cameraMatrix_left.at<double>(1,1);
    double cx = cameraMatrix_left.at<double>(0,2);
    double cy = cameraMatrix_left.at<double>(1,2);

    double disparity_number = fx * baseline;

    std::cout << " Camera Matrix L: \n" << cameraMatrix_left << std::endl << std::endl;
    std::cout << " Camera Matrix R: \n" << cameraMatrix_right << std::endl << std::endl;

#ifdef USE_OCV_TAPI
    cv::UMat map_left_x_gpu = map_left_x.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_left_y_gpu = map_left_y.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_x_gpu = map_right_x.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    cv::UMat map_right_y_gpu = map_right_y.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
#endif
    // ----> Initialize calibration

    // ----> Declare OpenCV images
#ifdef USE_OCV_TAPI
    cv::UMat frameYUV;  // Full frame side-by-side in YUV 4:2:2 format
    cv::UMat frameBGR(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full frame side-by-side in BGR format

    cv::UMat left_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left unrectified image
    cv::UMat right_raw(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right unrectified image

    cv::UMat left_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left rectified image
    cv::UMat right_rect(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right rectified image

    cv::UMat left_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Left image for the stereo matcher
    cv::UMat right_for_matcher(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Right image for the stereo matcher

    cv::UMat disparity_half(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Half sized disparity map
    cv::UMat disparity(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Full output disparity
    cv::UMat disparity_float(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Final disparity map in float32

    cv::UMat depth_map(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Depth map in float32
    cv::UMat depth_image(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Normalized and color remapped depth map to be saved

    cv::UMat left_grey(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Grey version of the left raw image
    cv::UMat right_grey(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Grey version of the right raw image

    cv::UMat temporary(cv::USAGE_ALLOCATE_DEVICE_MEMORY);  // intermediate image
    cv::UMat visualization(cv::USAGE_ALLOCATE_DEVICE_MEMORY); // Output visualization (left raw camera + depth map + right raw)
#else
    cv::Mat frameYUV, frameBGR;
    cv::Mat left_raw, right_raw, left_rect, right_rect;
    cv::Mat left_for_matcher, right_for_matcher;
    cv::Mat disparity_half, disparity, disparity_float;
    cv::Mat depth_map, depth_image;
    cv::Mat left_grey, right_grey;
    cv::Mat temporary, visualization;
#endif
    // <---- Declare OpenCV images

    // ----> Stereo matcher initialization
    sl_oc::tools::StereoSgbmPar stereoPar;

    //Note: you can use the tool 'zed_open_capture_depth_tune_stereo' to tune the parameters and save them to YAML
    if(!stereoPar.load())
        stereoPar.save(); // Save default parameters.

    cv::Ptr<cv::StereoSGBM> stereo_matcher = cv::StereoSGBM::create(stereoPar.minDisparity, stereoPar.numDisparities, stereoPar.blockSize);
    stereo_matcher->setMinDisparity(stereoPar.minDisparity);
    stereo_matcher->setNumDisparities(stereoPar.numDisparities);
    stereo_matcher->setBlockSize(stereoPar.blockSize);
    stereo_matcher->setP1(stereoPar.P1);
    stereo_matcher->setP2(stereoPar.P2);
    stereo_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    stereo_matcher->setMode(stereoPar.mode);
    stereo_matcher->setPreFilterCap(stereoPar.preFilterCap);
    stereo_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    stereo_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    stereo_matcher->setSpeckleRange(stereoPar.speckleRange);

    stereoPar.print();
    // <---- Stereo matcher initialization

    uint64_t last_ts=0;  // Used to check new frame arrival

    // ----> Video Writer definition
    // Define the output video file name and codec
    std::string outputFilename = recording_name + ".avi";
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // Codec for gray AVI format

    // Create a VideoWriter object to write the video to a file : left_grey | distance | right_grey
    cv::VideoWriter videoWriter(outputFilename, fourcc, (double) params.fps, cv::Size(w_out*3, h_out), false);

    // Check if the VideoWriter was successfully opened
    if (!videoWriter.isOpened())
    {
        std::cerr << "Could not open the output video file for writing" << std::endl;
        return EXIT_FAILURE;
    }
    // <---- Video Writer definition

    // Infinite video grabbing loop
    for (int k=0; k<loop_count; k++)
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Get a new frame from camera
        const sl_oc::video::Frame frame = cap.getLastFrame();

        // ----> If the frame is valid we can convert, rectify and display it
        if(frame.data != nullptr && frame.timestamp != last_ts)
        {
            last_ts = frame.timestamp;

            // ----> Conversion from YUV 4:2:2 to BGR for visualization
#ifdef USE_OCV_TAPI
            cv::Mat frameYUV_cpu = cv::Mat(frame.height, frame.width, CV_8UC2, frame.data);
            frameYUV = frameYUV_cpu.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_HOST_MEMORY);
#else
            frameYUV = cv::Mat(frame.height, frame.width, CV_8UC2, frame.data);
#endif
            cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_YUYV);
            // <---- Conversion from YUV 4:2:2 to BGR for visualization

            // ----> Extract left and right images from side-by-side
            left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
            right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows));
            // <---- Extract left and right images from side-by-side

            // ----> Apply rectification
            sl_oc::tools::StopWatch remap_clock;
#ifdef USE_OCV_TAPI
            cv::remap(left_raw, left_rect, map_left_x_gpu, map_left_y_gpu, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x_gpu, map_right_y_gpu, cv::INTER_AREA );
#else
            cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_AREA );
#endif
            double remap_elapsed = remap_clock.toc();
            std::stringstream remapElabInfo;
            remapElabInfo << "Rectif. processing: " << remap_elapsed << " sec - Freq: " << 1./remap_elapsed;
            // <---- Apply rectification

            // ----> Stereo matching
            sl_oc::tools::StopWatch stereo_clock;

            // Resize the original images to improve performances
            cv::resize(left_rect,  left_for_matcher,  cv::Size(w_out, h_out), 1./DOWNSCALE_FACTOR, 1./DOWNSCALE_FACTOR, cv::INTER_AREA);
            cv::resize(right_rect, right_for_matcher, cv::Size(w_out, h_out), 1./DOWNSCALE_FACTOR, 1./DOWNSCALE_FACTOR, cv::INTER_AREA);

            // Apply stereo matching
            stereo_matcher->compute(left_for_matcher, right_for_matcher, disparity_half);
            disparity_half.convertTo(disparity_float, CV_32FC1);

            cv::multiply(disparity_float, 1./16., disparity_float); // Last 4 bits of SGBM disparity are decimal
            
            double elapsed = stereo_clock.toc();
            std::stringstream stereoElabInfo;
            stereoElabInfo << "Stereo processing: " << elapsed << " sec - Freq: " << 1./elapsed;
            // <---- Stereo matching

            // ----> Extract Depth map
            // The DISPARITY MAP can be now transformed in DEPTH MAP using the formula
            // depth = (f * B) / disparity
            // where 'f' is the camera focal, 'B' is the camera baseline, 'disparity' is the pixel disparity

            cv::divide(disparity_number, disparity_float, depth_map);
            cv:min(depth_map, static_cast<double>(stereoPar.maxDepth_mm), depth_map);

            // Map depth_map to a 0-255 grayscale image 
            cv::add(depth_map, -static_cast<double>(stereoPar.minDepth_mm), depth_map);  // Minimum depth offset correction
            cv::multiply(depth_map, 1./(stereoPar.maxDepth_mm-stereoPar.minDepth_mm), depth_image, 255., CV_8UC1);  // Normalization and rescaling

            // Convert left & right image to grayscale for visualization
            cv::cvtColor(left_for_matcher, left_grey, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_for_matcher, right_grey, cv::COLOR_BGR2GRAY);

            // Create a 3 columns image visualization
            cv::hconcat(left_grey, depth_image, temporary);
            cv::hconcat(temporary, right_grey, visualization);

            videoWriter.write(visualization);
            // <---- Extract Depth map
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        if (elapsedTime < dt)
            std::this_thread::sleep_for(dt - elapsedTime);
        else
            std::cout << "Out of time..." << std::endl;

    }

    // Release VideoWriter object
    videoWriter.release();

    std::cout << "Video saved to " << outputFilename << std::endl;

    return EXIT_SUCCESS;
}