#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

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
    virtual mat process(mat &&img) = 0;
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

    virtual mat process(mat &&img) override final
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

class auto_correct
    : public processor
{
public:
private:
    virtual mat process(mat &&img) override final
    {
        return img;
    }
};

}   // namespace impl

processor_ptr make_auto_correct()
{
    return std::make_shared<impl::auto_correct>();
}

}   // namespace fuel

int main()
{
    try
    {
        //cv::blendLinear()
        auto const web_cam_name = "/dev/video0";

        auto src = fuel::make_web_cam_source(web_cam_name);

        auto proc = fuel::make_denoising_processor();

        while (true)
        {

            auto src_img = src->get();
            auto src_img_copy = src_img.clone();

            auto processed_img = proc->process(std::move(src_img));
            auto processed_img_copy = processed_img.clone();

            cv::imshow("Source imgge", src_img_copy);
            cv::imshow("Processed imgge", processed_img_copy);

            if (cv::waitKey(1) == 32)
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
