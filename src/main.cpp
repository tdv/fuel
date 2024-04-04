#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat imageToBlob(cv::Mat const image)
{
    if (image.empty())
        throw std::runtime_error{"Failed to read image."};


    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar stddev = cv::Scalar(0.229, 0.224, 0.225);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    resized_image -= mean;
    resized_image /= stddev;

    cv::Mat blob = cv::dnn::blobFromImage(resized_image);

    std::cout << "Shape of the input blob: " << blob.size << std::endl;

    return blob;
}

cv::Mat extractPersonMask(cv::Mat const& output_blob, int image_height, int image_width) {
    // Reshape the output blob to have dimensions [21 x 224 x 224]
    cv::Mat reshaped_blob = output_blob.reshape(1, 21);

    // Extract the 15th channel (index 14, 0-based indexing)
    cv::Mat person_channel = reshaped_blob.row(20);

    // Reshape the channel to match the spatial dimensions of the input image
    person_channel = person_channel.reshape(1, image_height);

    // Apply thresholding
    cv::Mat person_mask;
    cv::threshold(person_channel, person_mask, 2.0, 255, cv::THRESH_BINARY_INV);

    // Resize the mask to match the spatial dimensions of the input image
    cv::resize(person_mask, person_mask, cv::Size(image_width, image_height));

    return person_mask;
}

void test_resnet_101_coco(std::string const &model_path,
                     std::string const &image_path)
{
    auto resnet = cv::dnn::readNetFromONNX(model_path);
    resnet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    resnet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    auto img = cv::imread(image_path);

    auto blob = imageToBlob(img);
    resnet.setInput(blob);
    auto output = resnet.forward();
    std::cout << "Shape of the output blob: " << output.size << std::endl;

    auto mask = extractPersonMask(output, 224, 224);
    std::cout << "Shape of the mask: " << mask.size << std::endl;
    //std::cout << "Mask: " << mask << std::endl;

    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    while (cv::waitKey(100) != 27)
    {
        cv::imshow("Source", resized_image);
        cv::imshow("Mask", mask);
    }
}


int main()
{
    try {
        auto const *model_path = "/media/dmitry/stg1/Tst/test_ltorch_bg_remove/resnet101/resnet101.onnx";
        auto const *image_path = "../data/src.jpg";
        test_resnet_101_coco(model_path, image_path);
    }
    catch (std::exception const &e) {
        std::cerr << "An error has occured. Error: " << e.what() << std::endl;
    }

    return 0;
}
