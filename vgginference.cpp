#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <dirent.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

vector<string> list_dir(const char* path) {
	vector<string> files;
	struct dirent *entry;
	DIR *dir = opendir(path);

	if (dir == nullptr) {
		return files;
	}

	while ((entry = readdir(dir)) != nullptr) {
		//cout << entry->d_name << endl;
		files.push_back(entry->d_name);
	}
	closedir(dir);

	return files;
}

/*
 * Case Sensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string 'toMatch'
 */
bool is_image(const std::string &mainStr)
{
    vector<string> exts{"png", "jpg", "jpeg"};
    bool ret = false;

    for(auto ext : exts)
    {
        if(mainStr.size() >= ext.size() &&
                mainStr.compare(mainStr.size() - ext.size(), ext.size(), ext) == 0)
                return true;
            else
                ret = false;
    }

    return ret;
}

bool LoadImage(std::string file_name, cv::Mat &image) {
  image = cv::imread(file_name);  // CV_8UC3
  if (image.empty() || !image.data) {
    return false;
  }
  cv::cvtColor(image, image, COLOR_BGR2RGB);
  std::cout << "== image size: " << image.size() << " ==" << std::endl;

  // scale image to fit
  cv::Size scale(224, 224);
  cv::resize(image, image, scale);
  std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  return true;
}

int main(int argc, char *argv[]) 
{
   if(argc < 3)
   {
       cerr << "usage: torch_ext-app <path-to-exported-script-module> "
                 << "<path-to-image>\n";
       return -1;
   }
    std::string model_path = argv[1];
    std::string test_path = argv[2];

    auto time_start = system_clock::now();
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
    assert(module != nullptr);
    
    // Dont calculate gradients, same like with torch.no_grad()
    torch::NoGradGuard no_grad;

    auto time_load = system_clock::now();
    auto duration1 = duration_cast<microseconds>(time_load - time_start);
    cout << "Time for loading model: "
         << double(duration1.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    
    // Pre-operation for images
    // int img_size = 224;
    // std::vector<torch::jit::IValue> inputs;
    // Mat src, image;
    // src = imread(test_path);  // H, W, C
    // cv::imshow("lol1", src);

    // resize(src, image, Size(img_size, img_size)); // resize img
    // cvtColor(image, image, COLOR_BGR2RGB);  // bgr -> rgb   
    // at::TensorOptions options(at::ScalarType::Byte);
    // at::Tensor img_tensor = torch::from_blob(image.data, {1, img_size, img_size, 3}, options);
    // img_tensor = img_tensor.permute({0, 3, 1, 2});  //change the input order for torch 1,3,224,224
    // img_tensor = img_tensor.toType(torch::kFloat);
    // img_tensor = img_tensor.div(255);
    // img_tensor[0][0].sub_(0.485).div_(0.229);  //minus avg and divided by standard deviation
    // img_tensor[0][1].sub_(0.456).div_(0.224);
    // img_tensor[0][2].sub_(0.406).div_(0.225);

    // inputs.emplace_back(img_tensor);
    // at::Tensor result = module->forward(inputs).toTensor();
    // auto prob = result.softmax(1);
    // auto idx = prob.argmax();
    // cout << "The index is " << idx.item<float>() << endl;
    // cout << "The prob is " << prob[0][idx].item<float>() << endl;

    for (const auto &p : list_dir(test_path.c_str())){
        // std::vector<torch::jit::IValue> inputs;
        //std::cout << p << '\n';
        if(!is_image(p))
            continue;
        std::string s = test_path + p;
        //std::cout << "path: " << s << endl;
        Mat image = imread(s);
        resize(image, image, Size(224, 224)); // resize img
        cvtColor(image, image, COLOR_BGR2RGB);  // bgr -> rgb   
        image.convertTo(image, CV_32FC3, 1.0 / 255);
        //at::TensorOptions options(at::ScalarType::Byte); , options
        at::Tensor img_tensor = torch::from_blob(image.data, {1, 224, 224, 3});
        img_tensor = img_tensor.permute({0, 3, 1, 2});  //change the input order for torch 1,3,224,224
        
        // img_tensor = img_tensor.div(255);
        img_tensor[0][0].sub_(0.485).div_(0.229);  //minus avg and divided by standard deviation
        img_tensor[0][1].sub_(0.456).div_(0.224);
        img_tensor[0][2].sub_(0.406).div_(0.225);
        img_tensor = img_tensor.toType(torch::kFloat);

        // // std::ifstream is(model_path, std::ifstream::binary);
        // inputs.emplace_back(img_tensor);
        at::Tensor result = module->forward({img_tensor}).toTensor();
        //auto max_result = result.max(0, true);
        //std ::cout << std::get<1>(max_result);
        //auto max_index = std::get<1>(max_result).item<float>();
        //std::cout << max_index << std::endl;
        auto pred = result.argmax(1);
        std ::cout << pred.item<float>() << std::endl;
    }

    auto time_end = system_clock::now();
    auto duration2 = duration_cast<microseconds>(time_end - time_start);
    cout << "Time for inference:"
         << double(duration2.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    //cv::waitKey(0);
    return 0;
}