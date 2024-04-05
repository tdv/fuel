#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat adjust_gamma(const cv::Mat& img, double gamma = 1.0) {
    gamma = 1.0 / gamma;
    std::vector<uchar> tbl;
    tbl.reserve(256);
    for (int i = 0; i < 256; ++i) {
        tbl.push_back(static_cast<uchar>((pow(i / 255.0, gamma)) * 255));
    }
    cv::Mat lut(1, 256, CV_8U, tbl.data());
    cv::Mat result;
    cv::LUT(img, lut, result);
    return result;
}

cv::Mat equalize_hist(const cv::Mat& img) {
    // Convert input image to YUV color space
    cv::Mat yuv;
    cv::cvtColor(img, yuv, cv::COLOR_BGR2YUV);

    // Split YUV channels
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);

    // Ensure the Y channel is single-channel and of the correct type
    cv::Mat y_channel;
    if (channels[0].channels() > 1) {
        cv::cvtColor(channels[0], y_channel, cv::COLOR_BGR2GRAY);
    } else {
        y_channel = channels[0];
    }
    if (y_channel.depth() != CV_8U) {
        y_channel.convertTo(y_channel, CV_8U);
    }

    // Apply CLAHE to the Y channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2, cv::Size(2, 2));
    clahe->apply(y_channel, y_channel);

    // Merge the processed Y channel back with the U and V channels
    std::vector<cv::Mat> processed_channels = { y_channel, channels[1], channels[2] };
    cv::merge(processed_channels, yuv);

    // Convert YUV image back to RGB color space
    cv::cvtColor(yuv, yuv, cv::COLOR_YUV2RGB);

    return yuv;
}

cv::Mat imageToBlob(cv::Mat const image)
{
    if (image.empty())
        throw std::runtime_error{"Failed to read image."};


    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    //cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar stddev = cv::Scalar(0.229, 0.224, 0.225);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    resized_image -= mean;
    resized_image /= stddev;

    cv::Mat blob = cv::dnn::blobFromImage(resized_image);

    //std::cout << "Shape of the input blob: " << blob.size << std::endl;

    return blob;
}

cv::Mat extractPersonMask(cv::Mat const& output_blob, int image_height, int image_width) {
    // Reshape the output blob to have dimensions [21 x 224 x 224]
    cv::Mat reshaped_blob = output_blob.reshape(0, 21);

    // Extract the 15th channel (index 14, 0-based indexing)
    cv::Mat person_channel = reshaped_blob.row(20);

    // Reshape the channel to match the spatial dimensions of the input image
    person_channel = person_channel.reshape(1, image_height);

    // Apply thresholding
    cv::Mat person_mask;
    cv::threshold(person_channel, person_mask, 0.5, 255, cv::THRESH_BINARY_INV);

    // Resize the mask to match the spatial dimensions of the input image
    cv::resize(person_mask, person_mask, cv::Size(image_width, image_height));

    return person_mask;
}

cv::Mat cropByMask(cv::Mat const& image, cv::Mat const& mask) {
    // Ensure both image and mask have the same type
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8U);

    // Perform bitwise AND operation between the image and the mask
    cv::Mat cropped_image;
    cv::bitwise_and(image, image, cropped_image, mask_8u);

    return cropped_image;
}
void test_resnet_101_coco(std::string const &model_path,
                     std::string const &image_path)
{
    auto resnet = cv::dnn::readNetFromONNX(model_path);
    resnet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    resnet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    auto img = cv::imread(image_path);

    cv::VideoCapture cap{"/dev/video0"};

    if (!cap.isOpened())
        throw std::runtime_error{"Failed to open webcam."};

    while (cv::waitKey(10) != 27)
    {
        if (!cap.read(img))
            throw std::runtime_error{"Failed to read image."};

        /*cv::Mat img1;
        cv::medianBlur(img, img1, 3);
        img = img1;*/
        img = equalize_hist(img);
        //img = adjust_gamma(img, 0.7);

        auto blob = imageToBlob(img);
        resnet.setInput(blob);
        auto output = resnet.forward();
        //std::cout << "Shape of the output blob: " << output.size << std::endl;

        auto mask = extractPersonMask(output, 256, 256);
        //std::cout << "Shape of the mask: " << mask.size << std::endl;
        //std::cout << "Mask: " << mask << std::endl;

        cv::Mat resized_image;
        cv::resize(img, resized_image, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        auto cropped_image = cropByMask(resized_image, mask);

        cv::Mat result;
        cv::resize(cropped_image, result, cv::Size(800, 600), 0, 0, cv::INTER_LINEAR);

        //cv::imshow("Source", resized_image);
        //cv::imshow("Mask", mask);
        //cv::imshow("Cropped image", cropped_image);
        cv::imshow("Result", result);
    }
}


int main()
{
    try {
        //auto const *model_path = "../models/resnet101_coco_224_224.onnx";
        auto const *model_path = "../models/resnet101_coco_256_256.onnx";
        auto const *image_path = "../data/src.jpg";
        test_resnet_101_coco(model_path, image_path);
    }
    catch (std::exception const &e) {
        std::cerr << "An error has occured. Error: " << e.what() << std::endl;
    }

    return 0;
}
