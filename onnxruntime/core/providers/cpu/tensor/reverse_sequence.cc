// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence.h"
#include "onnx/defs/schema.h"

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "gsl/gsl_algorithm"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "core/framework/utils.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(ReverseSequence,
                        kOnnxDomain,
                        10,
                        kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
                        ReverseSequenceOp);

template <typename T>
static void ReverseSequenceImpl(const Tensor& X, Tensor& Y, gsl::span<const int64_t> sequence_lengths,
                                int64_t max_seq_len, int64_t batch_size, int64_t input_size, bool time_major,
                                onnxruntime::concurrency::ThreadPool* ttp);

Status ReverseSequenceOp::Compute(OpKernelContext* context) const {
  Status status = Status::OK();

  const auto& X = *context->Input<Tensor>(0);
  const auto data_type = X.DataType();
  const auto& dims = X.Shape();

  const auto batch_size = time_major_ ? dims[1] : dims[0];
  const auto max_seq_len = time_major_ ? dims[0] : dims[1];
  const auto input_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context->Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }

  auto& Y = *context->Output(0, dims);

  auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
  auto thread_pool = ctx_internal->GetOperatorThreadPool();

  DispatchOnTensorType(data_type, ReverseSequenceImpl, X, Y, seq_lengths.DataAsSpan<int64_t>(),
                       max_seq_len, batch_size, input_size, time_major_, const_cast<concurrency::ThreadPool*>(thread_pool));

  return status;
}

static int64_t TimeMajorInputOffset(const int64_t max_seq_len,
                                    const int64_t batch_size,
                                    const int64_t input_size,
                                    const int64_t batch_num,
                                    const int64_t seq_num) {
  ORT_UNUSED_PARAMETER(max_seq_len);
  return seq_num * batch_size * input_size + batch_num * input_size;
}

static int64_t BatchMajorInputOffset(const int64_t max_seq_len,
                                     const int64_t batch_size,
                                     const int64_t input_size,
                                     const int64_t batch_num,
                                     const int64_t seq_num) {
  ORT_UNUSED_PARAMETER(batch_size);
  return batch_num * max_seq_len * input_size + seq_num * input_size;
}

static int64_t TimeMajorOutputOffset(const int64_t max_seq_len,
                                     const int64_t batch_size,
                                     const int64_t input_size,
                                     const int64_t batch_num,
                                     const int64_t seq_num,
                                     const int64_t seq_len) {
  ORT_UNUSED_PARAMETER(max_seq_len);
  return (seq_len - seq_num - 1) * batch_size * input_size + batch_num * input_size;
}

static int64_t BatchMajorOutputOffset(const int64_t max_seq_len,
                                      const int64_t batch_size,
                                      const int64_t input_size,
                                      const int64_t batch_num,
                                      const int64_t seq_num,
                                      const int64_t seq_len) {
  ORT_UNUSED_PARAMETER(batch_size);
  return batch_num * max_seq_len * input_size + (seq_len - seq_num - 1) * input_size;
}

template <typename T>
static void ReverseSequenceImpl(const Tensor& X,
                                Tensor& Y,
                                gsl::span<const int64_t> sequence_lengths,
                                const int64_t max_seq_len,
                                const int64_t batch_size,
                                const int64_t input_size,
                                bool time_major,
                                onnxruntime::concurrency::ThreadPool* ttp) {
  gsl::span<const T> inputs = X.DataAsSpan<T>();
  gsl::span<T> inputs_reverse = Y.MutableDataAsSpan<T>();

  auto input_offset = time_major ? TimeMajorInputOffset : BatchMajorInputOffset;

  auto reversed_output_offset = time_major ? TimeMajorOutputOffset : BatchMajorOutputOffset;

  for (int i = 0; i < batch_size; i++) {
    int64_t seq_len = sequence_lengths[i];

    if (seq_len == 0)
      continue;

    int64_t shard_size = ttp->CalculateShardSize(seq_len);
    std::function<void(int64_t,int64_t)> work_object_one = [&](int64_t first, int64_t last) {
      for (int64_t j = first; j < last; j++) {
        gsl::span<const T> src = inputs.subspan(input_offset(max_seq_len, batch_size, input_size, i, j), input_size);
        gsl::span<T> dest = inputs_reverse.subspan(
            reversed_output_offset(max_seq_len, batch_size, input_size, i, j, seq_len), input_size);

        // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
        gsl::copy(src, dest);
      }
    };
    ttp->ParallelFor(seq_len, shard_size, work_object_one);

    shard_size = ttp->CalculateShardSize(max_seq_len - seq_len);
    std::function<void(int64_t,int64_t)> work_object_two = [&](int64_t first, int64_t last) {
      for (int64_t j = first; j < last; j++) {
        const auto offset = input_offset(max_seq_len, batch_size, input_size, i, j + seq_len);
        gsl::span<const T> src = inputs.subspan(offset, input_size);
        gsl::span<T> dest = inputs_reverse.subspan(offset, input_size);

        // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
        gsl::copy(src, dest);
      }
    };
    ttp->ParallelFor((max_seq_len - seq_len), shard_size, work_object_two);
  }
}

}  // namespace onnxruntime
