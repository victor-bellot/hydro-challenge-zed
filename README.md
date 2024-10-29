# hydro-challenge-zed

Part of the HydroChallenge project.

![les-bgs](ressources/hydrochal-cover.JPG)

## From https://github.com/stereolabs/zed-open-capture.git

## TODO

- Fix croping effect -> tune disparity computation parameters
- Object detection and tracking implementation

### Add udev rule
Stereo cameras such as ZED 2 and ZED Mini have built-in sensors (e.g. IMU) that are identified as USB HID devices.
To be able to access the USB HID device, you must add a udev rule contained in the `udev` folder:

    $ cd udev
    $ bash install_udev_rule.sh
    $ cd ..

### Build

#### Build library and examples

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j$(nproc)

#### Build only the library

    $ mkdir build
    $ cd build
    $ cmake .. -DBUILD_EXAMPLES=OFF
    $ make -j$(nproc)

#### Build only the video capture library

    $ mkdir build
    $ cd build
    $ cmake .. -DBUILD_SENSORS=OFF -DBUILD_EXAMPLES=OFF
    $ make -j$(nproc)

#### Build only the sensor capture library

    $ mkdir build
    $ cd build
    $ cmake .. -DBUILD_VIDEO=OFF -DBUILD_EXAMPLES=OFF
    $ make -j$(nproc)

## Run

To install the library, go to the `build` folder and launch the following commands:

    $ sudo make install
    $ sudo ldconfig

### Get video data

Include the `videocapture.hpp` header, declare a `VideoCapture` object and retrieve a video frame (in YUV 4:2:2 format) with `getLastFrame()`:

    #include "videocapture.hpp"
    sl_oc::video::VideoCapture cap;
    cap.initializeVideo();
    const sl_oc::video::Frame frame = cap.getLastFrame();
