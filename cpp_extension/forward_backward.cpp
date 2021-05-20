#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <assert.h>

#define Tensor torch::Tensor
#define IntArrayRef at::IntArrayRef

template <typename scalar_t>
static void unfolded2d_acc(
   scalar_t* finput_data,
   scalar_t* input_data,
   int64_t kH,
   int64_t kW,
   int64_t dH,
   int64_t dW,
   int64_t padH,
   int64_t padW,
   int64_t n_input_plane,
   int64_t input_height,
   int64_t input_width,
   int64_t output_height,
   int64_t output_width) {
   #pragma omp parallel for
   for (auto nip = 0; nip < n_input_plane; nip++) {
     int64_t kw, kh, y, x;
     int64_t ix, iy;
     for (kh = 0; kh < kH; kh++) {
       for (kw = 0; kw < kW; kw++) {
         scalar_t* src = finput_data +
             nip * ((size_t)kH * kW * output_height * output_width) +
             kh * ((size_t)kW * output_height * output_width) +
             kw * ((size_t)output_height * output_width);
         scalar_t* dst =
             input_data + nip * ((size_t)input_height * input_width);
         if (padW > 0 || padH > 0) {
           int64_t lpad, rpad;
           for (y = 0; y < output_height; y++) {
             iy = (int64_t)y * dH - padH + kh;
             if (iy < 0 || iy >= input_height) {
             } else {
                 for (x = 0; x < output_width; x++) {
                   ix = (int64_t)x * dW - padW + kw;
                   if (ix < 0 || ix >= input_width) {
                   } else {
                     scalar_t* dst_slice = dst + (size_t)iy * input_width + ix;
                     *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
                   }
                 }
             }
           }
         } else {
           for (y = 0; y < output_height; y++) {
             iy = (int64_t)y * dH + kh;
             ix = 0 + kw;
               for (x = 0; x < output_width; x++) {
                 scalar_t* dst_slice =
                     dst + (size_t)iy * input_width + ix + x * dW;
                 *dst_slice = *dst_slice + src[(size_t)y * output_width + x];
               }
           }
         }
       }
     }
   }
}

void unfolded2d_acc_kernel(
   Tensor& finput,
   Tensor& input,
   int64_t kH,
   int64_t kW,
   int64_t dH,
   int64_t dW,
   int64_t padH,
   int64_t padW,
   int64_t n_input_plane,
   int64_t input_height,
   int64_t input_width,
   int64_t output_height,
   int64_t output_width) {
 // This function assumes that
 // output_height*dH does not overflow a int64_t
 // output_width*dW does not overflow a int64_t

   auto input_data = (float*) input.data_ptr();
   auto finput_data =(float*) finput.data_ptr();

   unfolded2d_acc(
       finput_data,
       input_data,
       kH,
       kW,
       dH,
       dW,
       padH,
       padW,
       n_input_plane,
       input_height,
       input_width,
       output_height,
       output_width);
}

template <typename scalar_t>
static void unfolded2d_copy(
   scalar_t* input_data,
   scalar_t* finput_data,
   int64_t kH,
   int64_t kW,
   int64_t dH,
   int64_t dW,
   int64_t padH,
   int64_t padW,
   int64_t n_input_plane,
   int64_t input_height,
   int64_t input_width,
   int64_t output_height,
   int64_t output_width) {

   auto start = 0;
   auto end = (int64_t)n_input_plane * kH * kW;
   #pragma omp parallel for
   for (auto k = start; k < end; k++) {
         int64_t nip = k / (kH * kW);
         int64_t rest = k % (kH * kW);
         int64_t kh = rest / kW;
         int64_t kw = rest % kW;
         int64_t x, y;
         int64_t ix, iy;
         scalar_t* dst = finput_data +
             nip * ((size_t)kH * kW * output_height * output_width) +
             kh * ((size_t)kW * output_height * output_width) +
             kw * ((size_t)output_height * output_width);
         scalar_t* src =
             input_data + nip * ((size_t)input_height * input_width);
         if (padW > 0 || padH > 0) {
           int64_t lpad, rpad;
           for (y = 0; y < output_height; y++) {
             iy = (int64_t)y * dH - padH + kh;
             if (iy < 0 || iy >= input_height) {
               memset(
                   dst + (size_t)y * output_width,
                   0,
                   sizeof(scalar_t) * output_width);
             } else {
               if (dW == 1) {
                 ix = 0 - padW + kw;
                 lpad = std::max<int64_t>(0, padW - kw);
                 rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                 if (output_width - rpad - lpad <= 0) {
                   memset(
                       dst + (size_t)y * output_width,
                       0,
                       sizeof(scalar_t) * output_width);
                 } else {
                   if (lpad > 0)
                     memset(
                         dst + (size_t)y * output_width,
                         0,
                         sizeof(scalar_t) * lpad);
                   memcpy(
                       dst + (size_t)y * output_width + lpad,
                       src + (size_t)iy * input_width + ix + lpad,
                       sizeof(scalar_t) * (output_width - rpad - lpad));
                   if (rpad > 0)
                     memset(
                         dst + (size_t)y * output_width + output_width - rpad,
                         0,
                         sizeof(scalar_t) * rpad);
                 }
               } else {
                 for (x = 0; x < output_width; x++) {
                   ix = (int64_t)x * dW - padW + kw;
                   if (ix < 0 || ix >= input_width)
                     memset(
                         dst + (size_t)y * output_width + x,
                         0,
                         sizeof(scalar_t) * 1);
                   else
                     memcpy(
                         dst + (size_t)y * output_width + x,
                         src + (size_t)iy * input_width + ix,
                         sizeof(scalar_t) * (1));
                 }
               }
             }
           }
         } else {
           for (y = 0; y < output_height; y++) {
             iy = (int64_t)y * dH + kh;
             ix = 0 + kw;
             if (dW == 1)
               memcpy(
                   dst + (size_t)y * output_width,
                   src + (size_t)iy * input_width + ix,
                   sizeof(scalar_t) * output_width);
             else {
               for (x = 0; x < output_width; x++)
                 memcpy(
                     dst + (size_t)y * output_width + x,
                     src + (size_t)iy * input_width + ix + (int64_t)x * dW,
                     sizeof(scalar_t) * (1));
             }
           }
         }
       }
}

void unfolded2d_copy_kernel(
   Tensor& finput,
   Tensor& input,
   int64_t kH,
   int64_t kW,
   int64_t dH,
   int64_t dW,
   int64_t padH,
   int64_t padW,
   int64_t n_input_plane,
   int64_t input_height,
   int64_t input_width,
   int64_t output_height,
   int64_t output_width) {

       auto input_data = (float*) input.data_ptr();
       auto finput_data =(float*) finput.data_ptr();//.data_ptr<scalar_t>();

       unfolded2d_copy(
           input_data,
           finput_data,
           kH,
           kW,
           dH,
           dW,
           padH,
           padW,
           n_input_plane,
           input_height,
           input_width,
           output_height,
           output_width);
}

static void slow_conv2d_update_output_frame(
   Tensor& input,
   Tensor& output,
   const Tensor& weight,
   const Tensor& bias,
   Tensor& finput,
   int64_t kernel_height,
   int64_t kernel_width,
   int64_t stride_height,
   int64_t stride_width,
   int64_t pad_height,
   int64_t pad_width,
   int64_t n_input_plane,
   int64_t input_height,
   int64_t input_width,
   int64_t n_output_plane,
   int64_t output_height,
   int64_t output_width) {

   unfolded2d_copy_kernel(
     finput,
     input,
     kernel_height,
     kernel_width,
     stride_height,
     stride_width,
     pad_height,
     pad_width,
     n_input_plane,
     input_height,
     input_width,
     output_height,
     output_width);


 auto output2d =
     output.reshape({n_output_plane, output_height * output_width});
 if (bias.defined()) {
   for (int64_t i = 0; i < n_output_plane; i++) {
     output[i].fill_(bias[i].item());
   }
 } else {
   output.zero_();
 }
 output2d.addmm_(weight, finput, 1, 1);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_forward_out_cpu(
   Tensor& output,
   Tensor& finput,
   Tensor& fgrad_input,
   const Tensor& self,
   const Tensor& weight_,
   IntArrayRef kernel_size,
   const Tensor& bias,
   IntArrayRef stride,
   IntArrayRef padding)  {
 const int64_t kernel_height = kernel_size[0];
 const int64_t kernel_width = kernel_size[1];
 const int64_t pad_height = padding[0];
 const int64_t pad_width = padding[1];
 const int64_t stride_height = stride[0];
 const int64_t stride_width = stride[1];


 assert(weight_.dim()==2);
 const Tensor weight_2d = weight_;


 const Tensor input = self.contiguous(); // input is "self"
 const int64_t ndim = input.dim();
 const int64_t dim_planes = 1;
 const int64_t dim_height = 2;
 const int64_t dim_width = 3;

 const int64_t n_input_plane = input.size(dim_planes);
 const int64_t input_height = input.size(dim_height);
 const int64_t input_width = input.size(dim_width);
 const int64_t n_output_plane = weight_2d.size(0);
 const int64_t output_height =
     (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
 const int64_t output_width =
     (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

 const int64_t batch_size = input.size(0);


 finput.resize_({batch_size,
                 n_input_plane * kernel_height * kernel_width,
                 output_height * output_width});
 output.resize_({batch_size, n_output_plane, output_height, output_width});

 at::NoGradGuard no_grad;
 at::AutoNonVariableTypeMode non_variable_type_mode(true);
 
   #pragma omp parallel for
   for (int64_t t = 0; t < batch_size; t++) {
     Tensor input_t = input[t];
     Tensor output_t = output[t];
     Tensor finput_t = finput[t];
     slow_conv2d_update_output_frame(
         input_t,
         output_t,
         weight_2d,
         bias,
         finput_t,
         kernel_height,
         kernel_width,
         stride_height,
         stride_width,
         pad_height,
         pad_width,
         n_input_plane,
         input_height,
         input_width,
         n_output_plane,
         output_height,
         output_width);
   }

 return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}


std::tuple<Tensor, Tensor, Tensor> slow_conv2d_forward_cpu(
   const Tensor& self,
   const Tensor& weight,
   IntArrayRef kernel_size,
   const Tensor& bias,
   IntArrayRef stride,
   IntArrayRef padding) {

 auto output = at::empty({0}, self.options());
 auto finput = at::empty({0}, self.options());
 auto fgrad_input = at::empty({0}, self.options());

 slow_conv2d_forward_out_cpu(
     output,
     finput,
     fgrad_input,
     self,
     weight,
     kernel_size,
     bias,
     stride,
     padding);
 return std::make_tuple(output, finput, fgrad_input);
}

void slow_conv2d_backward_update_grad_input_frame(
   Tensor& grad_input,
   const Tensor& grad_output,
   const Tensor& weight,
   Tensor& fgrad_input,
   int64_t kernel_height,
   int64_t kernel_width,
   int64_t stride_height,
   int64_t stride_width,
   int64_t pad_height,
   int64_t pad_width) {
 auto grad_output_2d = grad_output.reshape(
     {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();//here
 fgrad_input.addmm_(weight, grad_output_2d, 0, 1); // can even be add mm_(grad_output_2d, weight, 1, 0).transpose(0, 1) (?)
//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();//here
//std::chrono::duration<double, std::milli> time_span = t2-t1;
//std::cerr<< "time for addmm weight and grad_output_2d = " << time_span.count() << std::endl;
 grad_input.zero_();
 unfolded2d_acc_kernel(
     fgrad_input,
     grad_input,
     kernel_height,
     kernel_width,
     stride_height,
     stride_width,
     pad_height,
     pad_width,
     grad_input.size(0),
     grad_input.size(1),
     grad_input.size(2),
     grad_output.size(1),
     grad_output.size(2));
}

void slow_conv2d_backward_out_cpu_template(
   Tensor& grad_input,
   const Tensor& grad_output_,
   const Tensor& input_,
   const Tensor& weight_,
   const Tensor& finput,
   Tensor& fgrad_input,
   IntArrayRef kernel_size,
   IntArrayRef stride,
   IntArrayRef padding) {
 const int64_t kernel_height = kernel_size[0];
 const int64_t kernel_width = kernel_size[1];
 const int64_t pad_height = padding[0];
 const int64_t pad_width = padding[1];
 const int64_t stride_height = stride[0];
 const int64_t stride_width = stride[1];

 assert(weight_.dim() == 2);
 const Tensor weight = weight_;


 const Tensor input = input_.contiguous();
 const Tensor grad_output = grad_output_.contiguous();
 grad_input.resize_as_(input);
 fgrad_input.resize_as_(finput);
 fgrad_input.zero_();
//std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();//here
 Tensor tw = weight.transpose(0, 1);
 if(tw.is_sparse() && !tw.is_coalesced()){
   tw = tw.coalesce();
 }
 const Tensor tweight = tw;
//std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();//here
//std::chrono::duration<double, std::milli> time_span = t2-t1;
//std::cerr<< "time for sparse weight transpose and coalesce = " << time_span.count() << std::endl;
 const int64_t batch_size = input.size(0);
 #pragma omp parallel for
 for (int64_t t = 0; t < batch_size; t++) {
 Tensor grad_input_t = grad_input[t];
 Tensor grad_output_t = grad_output[t];
 Tensor fgrad_input_t = fgrad_input[t];
 slow_conv2d_backward_update_grad_input_frame(
     grad_input_t,
     grad_output_t,
     tweight,
     fgrad_input_t,
     kernel_height,
     kernel_width,
     stride_height,
     stride_width,
     pad_height,
     pad_width);
 }
}

void slow_conv2d_backward_parameters_frame(
   Tensor& grad_weight,
   Tensor& grad_bias,
   Tensor& grad_output,
   const Tensor& finput) {
 auto grad_output_2d = grad_output.view(
     {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
 if (grad_weight.defined()) {
   const Tensor tfinput = finput.transpose(0, 1);
   grad_weight.addmm_(grad_output_2d, tfinput);// core computation for grad_weight
 }

 if (grad_bias.defined()) {
   AT_DISPATCH_FLOATING_TYPES_AND(
       at::ScalarType::BFloat16,
       grad_output.scalar_type(),
       "slow_conv2d_backward_parameters",
       [&] {
         auto grad_output_2d_acc = grad_output_2d.accessor<scalar_t, 2>();
         auto grad_bias_acc = grad_bias.accessor<scalar_t, 1>();
         const auto sz = grad_output_2d.size(1);
         for (int64_t i = 0; i < grad_bias.size(0); i++) {
           scalar_t sum = 0;
           for (int64_t k = 0; k < sz; k++) {
             sum = sum + grad_output_2d_acc[i][k];
           }
           grad_bias_acc[i] = grad_bias_acc[i] + sum;
         }
       });
 }
}

static void slow_conv2d_backward_parameters_out_cpu_template(
   Tensor& grad_weight,
   Tensor& grad_bias,
   const Tensor& input_,
   const Tensor& grad_output_,
   const Tensor& finput,
   Tensor fgrad_input,
   IntArrayRef kernel_size,
   IntArrayRef stride,
   IntArrayRef padding) {

 const int64_t kernel_height = kernel_size[0];
 const int64_t kernel_width = kernel_size[1];
 const int64_t pad_height = padding[0];
 const int64_t pad_width = padding[1];
 const int64_t stride_height = stride[0];
 const int64_t stride_width = stride[1];
 
 Tensor grad_weight_2d = grad_weight;

 auto input = input_.contiguous();
 auto grad_output = grad_output_.contiguous();

 const int64_t batch_size = input.size(0);
 for (int64_t t = 0; t < batch_size; t++) {
   Tensor grad_output_t = grad_output[t];
   Tensor finput_t;
   if (grad_weight_2d.defined()) {
     finput_t = finput[t];
   }

   slow_conv2d_backward_parameters_frame(
       grad_weight_2d, grad_bias, grad_output_t, finput_t);
 }
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cpu(
   Tensor& grad_input,
   Tensor& grad_weight,
   Tensor& grad_bias,
   const Tensor& grad_output,
   const Tensor& self,
   const Tensor& weight,
   IntArrayRef kernel_size,
   IntArrayRef stride,
   IntArrayRef padding,
   const Tensor& finput,
   const Tensor& fgrad_input) {
 //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();//here
 if (grad_input.defined()) {
   slow_conv2d_backward_out_cpu_template( // input
       grad_input,
       grad_output,
       self,
       weight,
       finput,
       const_cast<Tensor&>(fgrad_input),   // cast away auto-generated const of buffer
       kernel_size,
       stride,
       padding);
 }
 //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();//here
 //std::chrono::duration<double, std::milli> time_span = t2-t1;
 //std::cerr<< "time for slow_conv2d_backward_out_cpu_template (input) = " << time_span.count() << std::endl;

 if (grad_weight.defined()) {
   grad_weight.resize_(weight.sizes());
   grad_weight.zero_();
 }

 if (grad_bias.defined()) {
   grad_bias.resize_({grad_output.size(1)});
   grad_bias.zero_();
 }
//std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();//here
 if (grad_weight.defined() || grad_bias.defined()) { // weight and bias
   slow_conv2d_backward_parameters_out_cpu_template(
       grad_weight,
       grad_bias,
       self,
       grad_output,
       finput,
       fgrad_input,
       kernel_size,
       stride,
       padding);
 }
//std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();//here
//std::chrono::duration<double, std::milli> time_span_1 = t4-t3;
//std::cerr<< "time for slow_conv2d_backward_parameters_out_cpu_template (weight and bias) = " << time_span_1.count() << std::endl;

 return std::tuple<Tensor&, Tensor&, Tensor&>(
     grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cpu(
   const Tensor& grad_output,
   const Tensor& self,
   const Tensor& weight,
   IntArrayRef kernel_size,
   IntArrayRef stride,
   IntArrayRef padding,
   const Tensor& finput,
   const Tensor& fgrad_input,
   std::array<bool, 3> output_mask) {
 Tensor grad_input;
 Tensor grad_weight;
 Tensor grad_bias;

 if (output_mask[0]) {
   grad_input = at::empty({0}, grad_output.options());
 }

 if (output_mask[1]) {
   grad_weight = at::empty({0}, grad_output.options());
 }

 if (output_mask[2]) {
   grad_bias = at::empty({0}, grad_output.options());
 }
 //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();//here
 slow_conv2d_backward_out_cpu(
     grad_input,
     grad_weight,
     grad_bias,
     grad_output,
     self,
     weight,
     kernel_size,
     stride,
     padding,
     finput,
     fgrad_input);
 //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();//here
 //std::chrono::duration<double, std::milli> time_span = t2-t1;
 //std::cerr<< "time for slow_conv2d_backward_out_cpu = " << time_span.count() << std::endl;
 return std::make_tuple(grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("forward", &slow_conv2d_forward_cpu, "Conv Forward");
 m.def("backward", &slow_conv2d_backward_cpu, "Conv Backward");
}


