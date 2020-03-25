#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>

const char* labels[] =
{
    "5 carreau",
    "8 carreau",
    "10 trefle",
    "2 pique",
    "valet trefle",
    "roi trefle",
    "as pique",
    "2 coeur",
    "5 pique",
    "valet coeur",
    "9 pique",
    "5 coeur",
    "8 trefle",
    "10 pique",
    "5 trefle",
    "roi coeur",
    "7 coeur",
    "4 carreau",
    "7 trefle",
    "roi pique",
    "7 carreau",
    "9 coeur",
    "10 coeur",
    "4 pique",
    "valet carreau",
    "9 trefle",
    "2 trefle",
    "2 carreau",
    "6 coeur",
    "dame pique",
    "3 pique",
    "6 carreau",
    "7 pique",
    "10 carreau",
    "as trefle",
    "dame carreau",
    "roi carreau",
    "6 pique",
    "valet pique",
    "3 trefle",
    "8 pique",
    "as coeur",
    "3 carreau",
    "4 trefle",
    "4 coeur",
    "9 carreau",
    "dame trefle",
    "8 coeur",
    "dame coeur",
    "6 trefle",
    "as carreau",
    "3 coeur"
};

int main(int num_args, char** args)
{
    if(num_args != 3)
    {
        std::cerr << "Bad command line!" << std::endl;
        exit(1);
    }

    const std::string model_path = args[1];
    const std::string image_path = args[2];

    //const std::string model_path = "/home/victor/developpement/dyonisos/model.zip";
    //const std::string image_path = "/home/victor/datasets/dyonisos/dataset/3_coeur/001015.png";

    // load image.

    cv::Mat3b image0 = cv::imread(image_path);
    if(image0.data == nullptr)
    {
        std::cout << "Coudl not load image!" << std::endl;
        exit(0);
    }

    // convert to RGB.

    cv::cvtColor(image0, image0, cv::COLOR_BGR2RGB);

    // convert to floating point.

    cv::Mat3f image;
    image0.convertTo(image, CV_32FC3);

    // normalize image.

    {
        const cv::Vec3f mu(0.485f, 0.456f, 0.406f);
        const cv::Vec3f sigma(0.229f, 0.224f, 0.225f);

        for(int i=0; i<image.rows; i++)
        {
            float* ptr = reinterpret_cast<float*>( image.ptr(i) );

            for(int j=0; j<image.cols; j++)
            {
                ptr[0] = (ptr[0] / 255.0f - mu[0]) / sigma[0];
                ptr[1] = (ptr[1] / 255.0f - mu[1]) / sigma[1];
                ptr[2] = (ptr[2] / 255.0f - mu[2]) / sigma[2];

                /*
                ptr[0] = (ptr[0] - mu[0]) / sigma[0];
                ptr[1] = (ptr[1] - mu[1]) / sigma[1];
                ptr[2] = (ptr[2] - mu[2]) / sigma[2];
                */

                ptr += 3;
            }
        }
    }

    // load model and perform inference.

    torch::Device device(torch::kCUDA);

    torch::jit::script::Module model = torch::jit::load(model_path);

    if(image.isContinuous() == false)
    {
        std::cout << "Internal error!" << std::endl;
        exit(1);
    }

    torch::Tensor input = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kF32);

    input = input.permute({0, 3, 1, 2});

    input = input.to(device);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    at::Tensor output = model.forward(inputs).toTensor();

    at::Tensor exp_output = at::exp(output);

    at::Tensor proba = exp_output / at::sum(exp_output);

    if( proba.dim() != 2 || proba.size(0) != 1 || proba.size(1) != 52 )
    {
        std::cout << "Internal error!" << std::endl;
        exit(1);
    }

    std::vector<float> values(52);
    std::vector<size_t> index(52);
    for(size_t i=0; i<52; i++)
    {
        values[i] = proba[0][i].item<float>();
        index[i] = i;
    }

    std::sort(index.begin(), index.end(), [&values] (size_t a, size_t b) { return values[a] < values[b]; } );

    for(size_t i=0; i<3 /* 52 */; i++)
    {
        const size_t j = index[i];
        std::cout << "P(" << labels[j] << ") = " << values[j] << std::endl;
    }

    return 0;
}

