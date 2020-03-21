#include <iostream>
#include <opencv2/imgcodecs.hpp>
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

    // load image.

    cv::Mat3b image = cv::imread(image_path);
    if(image.data == nullptr)
    {
        std::cout << "Coudl not load image!" << std::endl;
        exit(0);
    }

    // normalize image.

    cv::Vec3d sum1(0.0, 0.0, 0.0);
    cv::Vec3d sum2(0.0, 0.0, 0.0);
    for( const cv::Vec3d& x : image )
    {
        sum1[0] += x[0];
        sum1[1] += x[1];
        sum1[2] += x[2];

        sum2[0] += x[0]*x[0];
        sum2[1] += x[1]*x[0];
        sum2[2] += x[2]*x[0];
    }

    cv::Vec3d mean;
    cv::Vec3d std_dev;
    for(int k=0; k<3; k++)
    {
        mean[k] = sum1[k] / static_cast<double>(image.size().area());
        std_dev[k] = sum2[k] / static_cast<double>(image.size().area()) - mean[k]*mean[k];
    }
    const cv::Vec3d new_mean(0.485, 0.456, 0.406);
    const cv::Vec3d new_std_dev(0.229, 0.224, 0.225);

    cv::Mat3f new_image(image.size());

    for(int i=0; i<new_image.rows; i++)
    {
        for(int j=0; j<new_image.cols; j++)
        {
            for(int k=0; k<3; k++)
            {
                new_image(i,j)[k] = static_cast<float>( new_mean[k] + new_std_dev[k] * ( image(i,j)[k] - mean[k]) / std_dev[k] );
            }
        }
    }

    // load model and perform inference.

    torch::Device device(torch::kCUDA);

    torch::jit::script::Module model = torch::jit::load(model_path);

    torch::Tensor input = torch::from_blob(new_image.data, {1, image.rows, image.cols, 3}, torch::kF32);

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

    for(size_t i=0; i<52; i++)
    {
        const size_t j = index[i];
        std::cout << "P(" << labels[j] << ") = " << values[j] << std::endl;
    }

    return 0;
}

