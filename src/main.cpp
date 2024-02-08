// STD
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fuel
{

struct object
    : public std::enable_shared_from_this<object>
{
    virtual ~object() = default;
};

using object_ptr = std::shared_ptr<object>;

};

namespace fuel
{

using mat = cv::Mat;

struct source
    : public object
{
    virtual mat get() const = 0;
};

using source_ptr = std::shared_ptr<source>;

}   // namespace fuel


namespace fuel
{
namespace impl
{

class web_cam_exception
    : public std::runtime_error
{
public:
    web_cam_exception(std::string const &message)
        : std::runtime_error{message}
    {
    }
};

class web_cam
    : public source
{
public:
    explicit web_cam(std::string const &cam_name)
        : cam_name_{cam_name}
        , cap_{cam_name_}
    {
        if (!cap_.isOpened())
        {
            throw web_cam_exception{"Failed to open webcam \"" +
                    cam_name_ + "\"."};
        }

        std::this_thread::sleep_for(start_delay_);
    }

private:
    std::chrono::milliseconds const start_delay_{500};

    std::string cam_name_;
    mutable cv::VideoCapture cap_;

    // source
    virtual mat get() const override final
    {
        mat img;

        if (!cap_.read(img))
        {
            throw web_cam_exception{"Failed to read frame from webcam \"" +
                    cam_name_ + "\"."};
        }

        mat res;
        cv::flip(img, res, 1);

        return res;
    }
};
}   // namespace impl

source_ptr make_web_cam_source(std::string const &cam_name)
{
    return std::make_shared<impl::web_cam>(cam_name);
}

}   // namespace fuel


namespace fuel
{

struct processor
    : public object
{
    virtual mat process(mat img) = 0;
};

using processor_ptr = std::shared_ptr<processor>;

}   // namespace fuel

namespace fuel
{
namespace impl
{

class denoising_processor
    : public processor
{
public:
    explicit denoising_processor(int ksize)
        : ksize_{ksize}
    {
        if (ksize_ < 3)
            ksize_ = 3;
        if (!(ksize_ % 2))
            ++ksize_;
    }

private:
    int ksize_;

    virtual mat process(mat img) override final
    {
        mat res;

        cv::medianBlur(img, res, ksize_);

        return res;
    }
};

}   // namespace impl

processor_ptr make_denoising_processor(int ksize = 3)
{
    return std::make_shared<impl::denoising_processor>(ksize);
}

}   // namespace fuel

namespace fuel
{
namespace impl
{

class auto_correct_processor
    : public processor
{
public:
private:
    virtual mat process(mat img) override final
    {
        //https://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/

        mat res;

        BrightnessAndContrastAuto(img, res, 0);

        return res;
    }

    //----------------

    void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
    {

        CV_Assert(clipHistPercent >= 0);
        CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

        int histSize = 256;
        float alpha, beta;
        double minGray = 0, maxGray = 0;

        //to calculate grayscale histogram
        cv::Mat gray;
        if (src.type() == CV_8UC1) gray = src;
        else if (src.type() == CV_8UC3) cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        else if (src.type() == CV_8UC4) cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        if (clipHistPercent == 0)
        {
            // keep full available range
            cv::minMaxLoc(gray, &minGray, &maxGray);
        }
        else
        {
            cv::Mat hist; //the grayscale histogram

            float range[] = { 0, 256 };
            const float* histRange = { range };
            bool uniform = true;
            bool accumulate = false;
            calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

            // calculate cumulative distribution from the histogram
            std::vector<float> accumulator(histSize);
            accumulator[0] = hist.at<float>(0);
            for (int i = 1; i < histSize; i++)
            {
                accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
            }

            // locate points that cuts at required value
            float max = accumulator.back();
            clipHistPercent *= (max / 100.0); //make percent as absolute
            clipHistPercent /= 2.0; // left and right wings
            // locate left cut
            minGray = 0;
            while (accumulator[minGray] < clipHistPercent)
                minGray++;

            // locate right cut
            maxGray = histSize - 1;
            while (accumulator[maxGray] >= (max - clipHistPercent))
                maxGray--;
        }

        // current range
        float inputRange = maxGray - minGray;

        alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
        beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

        // Apply brightness and contrast normalization
        // convertTo operates with saurate_cast
        src.convertTo(dst, -1, alpha, beta);

        // restore alpha channel from source
        if (dst.type() == CV_8UC4)
        {
            int from_to[] = { 3, 3};
            cv::mixChannels(&src, 4, &dst,1, from_to, 1);
        }
        return;
    }
};

}   // namespace impl

processor_ptr make_auto_correct_processor()
{
    return std::make_shared<impl::auto_correct_processor>();
}

}   // namespace fuel

namespace fuel
{
namespace impl
{

class background_processor
    : public processor
{
public:
    background_processor(std::string const &new_bg_path,
            std::string const &model_path,
            std::string const &out_layer_name)
        : out_layer_name_{out_layer_name}
    {
        net_ = cv::dnn::readNet(model_path);

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    }

private:
    cv::Size const input_size_{256, 256};

    std::string out_layer_name_;

    cv::dnn::Net net_;

    virtual mat process(mat img) override final
    {
        auto const h = img.size[0];
        auto const w = img.size[1];
        auto const original_size = cv::Size{w, h};

        auto blob = cv::dnn::blobFromImage(img, 1.0/255, input_size_,
                cv::Scalar(), false, false, CV_32F);

        net_.setInput(blob);

        auto output = net_.forward(out_layer_name_);
        auto mask = output.reshape(1, 256) > 0.5;

        mat resized_mask;
        cv::resize(mask, resized_mask, original_size, cv::INTER_LINEAR);

        mat res(img.size(), img.type(), cv::Scalar(70, 70, 70));
        //img.copyTo(res, resized_mask);

        cv::bitwise_and(img, img, res, resized_mask);

        cv::dilate(res, res, cv::Mat(), cv::Point(-1, -1), 1);
        cv::erode(res, res, cv::Mat(), cv::Point(-1, -1), 1);

        return res;
    }
};

}   // namespace impl

processor_ptr make_background_processor(std::string const &new_bg_path,
        std::string const &model_path,
        std::string const &out_layer_name = "")
{
    return std::make_shared<impl::background_processor>(new_bg_path,
            model_path, out_layer_name);
}

}   // namespace fuel

bool areEqual(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat temp;
    cv::bitwise_xor(a,b,temp);
    return !(cv::countNonZero(temp.reshape(1)));
}

int main()
{
    try
    {
        //cv::blendLinear()
        auto const web_cam_name = "/dev/video0";

        auto src = fuel::make_web_cam_source(web_cam_name);

        auto const model_path = "../models/segmentation/openvino/model_float32.pb";
        auto const bg_path = "";

        std::vector<fuel::processor_ptr> processors{
            fuel::make_denoising_processor(3),
            fuel::make_auto_correct_processor(),
            fuel::make_background_processor(bg_path, model_path, "")

        };

        fuel::mat prev;

        while (true)
        {

            auto img = src->get();

            if (prev.empty())
                prev = img;

            if (areEqual(prev, img))
                continue;

            prev = img;

            for (auto &processor : processors)
                img = processor->process(std::move(img));


            cv::imshow("Image", img);

            if (cv::waitKey(10) == 32)
                break;
        }
    }
    catch (std::exception const &e)
    {
        std::cerr << "An error has occured. Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
