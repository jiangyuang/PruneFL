import torch
import torch.sparse as sparse

sparse_conv2d_imported = True
try:
    import sparse_conv2d
except ImportError:
    sparse_conv2d_imported = False


class AddmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, weight: sparse.FloatTensor, dense_weight_placeholder, inp):
        if bias is None:
            out = sparse.mm(weight, inp)
        else:
            out = sparse.addmm(bias, weight, inp)
        ctx.save_for_backward(bias, weight, inp)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        bias, weight, inp = ctx.saved_tensors
        grad_bias = grad_input = None
        if bias is not None:
            grad_bias = grad_output.sum(1).reshape((-1, 1))
        grad_weight = grad_output.mm(inp.t())
        if ctx.needs_input_grad[3]:
            grad_input = torch.mm(weight.t(), grad_output)

        return grad_bias, None, grad_weight, grad_input


if sparse_conv2d_imported:
    class SparseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weight, dense_weight_placeholder, kernel_size, bias, stride, padding):
            out, f_input, fgrad_input = sparse_conv2d.forward(inp, weight, kernel_size, bias, stride, padding)
            ctx.save_for_backward(inp, weight, f_input, fgrad_input)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input, grad_weight, grad_bias = sparse_conv2d.backward(grad_output,
                                                                        ctx.saved_tensors[0],
                                                                        ctx.saved_tensors[1],
                                                                        ctx.kernel_size,
                                                                        ctx.stride,
                                                                        ctx.padding,
                                                                        ctx.saved_tensors[2],
                                                                        ctx.saved_tensors[3],
                                                                        (True, True, True))
            return grad_input, None, grad_weight, None, grad_bias, None, None


    class DenseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, weight, kernel_size, bias, stride, padding):
            weight2d = weight.data.reshape((weight.size(0), -1))
            out, f_input, fgrad_input = sparse_conv2d.forward(inp, weight2d, kernel_size, bias, stride, padding)
            ctx.save_for_backward(inp, weight2d, f_input, fgrad_input, weight)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input, grad_weight2d, grad_bias = sparse_conv2d.backward(grad_output,
                                                                          ctx.saved_tensors[0],
                                                                          ctx.saved_tensors[1],
                                                                          ctx.kernel_size,
                                                                          ctx.stride,
                                                                          ctx.padding,
                                                                          ctx.saved_tensors[2],
                                                                          ctx.saved_tensors[3],
                                                                          (True, True, True))
            grad_weight = grad_weight2d.reshape_as(ctx.saved_tensors[4])
            return grad_input, grad_weight, None, grad_bias, None, None

else:
    class SparseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def apply(inp, weight, dense_weight_placeholder, kernel_size, bias, stride, padding):
            size_4d = (weight.size(0), -1, *kernel_size)
            with torch.no_grad():
                dense_weight_placeholder.zero_()
                dense_weight_placeholder.add_(weight.to_dense())
            return torch.nn.functional.conv2d(inp, dense_weight_placeholder.view(size_4d), bias, stride, padding)


    class DenseConv2dFunction(torch.autograd.Function):
        @staticmethod
        def apply(inp, weight, kernel_size, bias, stride, padding):
            size_4d = (weight.size(0), -1, *kernel_size)
            return torch.nn.functional.conv2d(inp, weight.reshape(size_4d), bias, stride, padding)
