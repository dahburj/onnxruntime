// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iostream>
#include <cstddef>
#include <mutex>
#include "controller.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "single_consumer.h"

template <typename InputIterator>
class AsyncRingBuffer {
 private:
  static VOID NTAPI ThreadPoolEntry(_Inout_ PTP_CALLBACK_INSTANCE, _Inout_opt_ PVOID data, _Inout_ PTP_WORK work) {
    CloseThreadpoolWork(work);
    (*(RunnableTask*)data)();
  }

  static size_t CalcItemSize(const std::vector<int64_t>& tensor_shape) {
    int64_t r = 1;
    for (int64_t i : tensor_shape) r *= i;
    return static_cast<size_t>(r);
  }

  enum class BufferState { EMPTY, FILLING, FULL, TAKEN };
  const size_t batch_size_;
  using InputType = typename InputIterator::value_type;
  DataProcessing* p_;
  OutputCollector<InputType>* c_;
  size_t capacity_;
  struct QueueItem {
    OrtValue* value = nullptr;
    std::vector<InputType> taskid_list;

    QueueItem() = default;
    ~QueueItem() { OrtReleaseValue(value); }
    QueueItem(const QueueItem&) = delete;
    QueueItem& operator=(const QueueItem&) = delete;
  };
  SingleConsumerFIFO<QueueItem> queue_;
  using TensorListEntry = typename SingleConsumerFIFO<QueueItem>::ListEntry;
  Controller& threadpool_;
  std::vector<int64_t> CreateTensorShapeWithBatchSize(const std::vector<int64_t>& input, size_t batch_size) {
    std::vector<int64_t> shape(input.size() + 1);
    shape[0] = batch_size;
    size_t len = shape.size();
    for (size_t i = 1; i != len; ++i) {
      shape[i] = input[i - 1];
    }
    return shape;
  }
  std::mutex m;

  struct BufferManager {
    size_t capacity_;
    size_t item_size_in_bytes_;
    size_t write_index_ = 0;
    std::vector<BufferState> buffer_state;
    std::vector<InputType> input_task_id_for_buffers_;

    // TODO: if there is an alignment requirement, this buffer need do padding between the tensors.
    std::vector<uint8_t> buffer_;

    BufferManager(size_t capacity, size_t item_size_in_bytes)
        : capacity_(capacity),
          item_size_in_bytes_(item_size_in_bytes),
          buffer_state(capacity, BufferState::EMPTY),
          input_task_id_for_buffers_(capacity),
          buffer_(item_size_in_bytes * capacity) {}

    size_t GetId(const uint8_t* p) const { return (p - buffer_.data()) / item_size_in_bytes_; }

    bool CompareAndSet(size_t i, BufferState old, BufferState new_state) {
      if (buffer_state[i] != old) return false;
      buffer_state[i] = new_state;
      return true;
    }

    bool CompareAndSet(size_t index, size_t index_end, BufferState old, BufferState new_state) {
      assert(index_end >= index);
      for (size_t i = index; i != index_end; ++i) {
        if (buffer_state[i] != old) return false;
      }
      for (size_t i = index; i != index_end; ++i) {
        buffer_state[i] = new_state;
      }
      return true;
    }

    bool TakeRange(size_t index, size_t index_end, std::vector<InputType>& task_id_list) {
      assert(index_end >= index);
      if (!CompareAndSet(index, index_end, BufferState::FULL, BufferState::TAKEN)) {
        return false;
      }
      auto* p = &input_task_id_for_buffers_[index];
      auto* p_end = p + (index_end - index);
      task_id_list.assign(p, p_end);
      return true;
    }

    uint8_t* data() { return buffer_.data(); }
    uint8_t* Next(InputType taskid) {
      for (size_t i = 0; i != capacity_; ++i) {
        size_t index = (write_index_ + i) % capacity_;
        if (buffer_state[i] == BufferState::EMPTY) {
          buffer_state[i] = BufferState::FILLING;
          input_task_id_for_buffers_[i] = taskid;
          return &buffer_[index * item_size_in_bytes_];
        }
      }
      return nullptr;
    }
  };
  BufferManager buffer_;
  InputIterator input_begin_;
  const InputIterator input_end_;
  char* errmsg_;
  // unsafe
  bool is_input_eof() const { return input_end_ == input_begin_; }

 public:
  size_t parallelism = 8;
  std::atomic<size_t> current_running_downloders = 0;
  size_t current_task_id = 0;

  AsyncRingBuffer(size_t batch_size, size_t capacity, Controller& threadpool, const InputIterator& input_begin,
                  const InputIterator& input_end, DataProcessing* p, OutputCollector<InputType>* c)
      : batch_size_(batch_size),
        p_(p),
        c_(c),
        capacity_((capacity + batch_size_ - 1) / batch_size_ * batch_size_),
        queue_(capacity_ / batch_size_),
        threadpool_(threadpool),
        buffer_(capacity_, p->GetOutputSizeInBytes(1)),
        input_begin_(input_begin),
        input_end_(input_end) {
    OrtAllocatorInfo* allocator_info;
    ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
    uint8_t* output_data = buffer_.data();
    size_t off = p->GetOutputSizeInBytes(batch_size_);
    std::vector<int64_t> input_shape = p_->GetOutputShape(batch_size_);
    queue_.Init([allocator_info, off, &output_data, &input_shape](TensorListEntry& e) {
      ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, output_data, off, input_shape.data(),
                                                           input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                           &e.value.value));
      output_data += off;
    });
    OrtReleaseAllocatorInfo(allocator_info);
  }

  void ReturnAndTake(TensorListEntry*& input_tensor) {
    std::lock_guard<std::mutex> g(m);
    if (input_tensor != nullptr) {
      size_t tensor_id = queue_.Return(input_tensor);
      size_t buffer_id = tensor_id * batch_size_;
      if (!buffer_.CompareAndSet(buffer_id, buffer_id + batch_size_, BufferState::TAKEN, BufferState::EMPTY)) {
        throw std::runtime_error("ReturnAndTake: internal state error");
      }
    }
    input_tensor = queue_.Take();
  }

  void OnDownloadFinished(const uint8_t* dest) {
    size_t buffer_id = buffer_.GetId(dest);
    // printf("set %zd to full\n", buffer_id);
    TensorListEntry* input_tensor = nullptr;
    {
      std::lock_guard<std::mutex> g(m);
      if (!buffer_.CompareAndSet(buffer_id, BufferState::FILLING, BufferState::FULL)) {
        throw std::runtime_error("ReturnAndTake: internal state error");
      }
      size_t tensor_id = buffer_id / batch_size_;
      std::vector<InputType> task_id_list;
      buffer_id = tensor_id * batch_size_;
      if (buffer_.TakeRange(buffer_id, buffer_id + batch_size_, task_id_list)) {
        queue_.Put(tensor_id)->value.taskid_list = task_id_list;
        input_tensor = queue_.Take();
      }
    }

    while (true) {
      int tasks = StartDownloadTasks();
      if (tasks < 0) return;

      if (input_tensor == nullptr) break;
      (*c_)(input_tensor->value.taskid_list, input_tensor->value.value);
      ReturnAndTake(input_tensor);
    }
  }

  void CheckError() {
    class FinishTask : public RunnableTask {
		void operator()() noexcept override {
		}
    };
    
	if (--current_running_downloders == 0) {
      std::lock_guard<std::mutex> g(m);
      if (input_begin_ == input_end_) {
      }
    }
  }

  void Fail(const char* errmsg) {
    std::lock_guard<std::mutex> g(m);
    input_begin_ = input_end_;
    errmsg_ = my_strdup(errmsg);
  }

  // call this function when a download task is just finished or any buffer became FREE.
  int StartDownloadTasks() {
    class DownloadTask : public RunnableTask {
     public:
      AsyncRingBuffer* requester;
      InputType source;
      uint8_t* dest;
      DownloadTask(AsyncRingBuffer* r, const InputType& s, uint8_t* d) : requester(r), source(s), dest(d) {}

      void operator()() noexcept override {
        AsyncRingBuffer* r = requester;
        InputType s = source;
        uint8_t* d = dest;
        delete this;
        try {
          (*r->p_)(&s, d);
          r->OnDownloadFinished(d);
        } catch (const std::exception& ex) {
          r->Fail(ex.what());
        }
        --r->current_running_downloders;
      }
    };

    // search empty slots, launch a download task for each of them
    std::vector<DownloadTask*> tasks_to_launch;
    {
      std::lock_guard<std::mutex> g(m);
      // if we have
      // 1. cpu  (current_running_downloders < parallelism)
      // 2. memory (buffer available)
      // 3. input_task
      // then schedule it to thread pool
      for (; current_running_downloders + tasks_to_launch.size() < parallelism && !is_input_eof();
           ++input_begin_, ++current_running_downloders) {
        uint8_t* b = buffer_.Next(*input_begin_);
        if (b == nullptr) break;  // no empty buffer
        tasks_to_launch.push_back(new DownloadTask(this, *input_begin_, b));
      }
    }

    for (DownloadTask* p : tasks_to_launch) {
      threadpool_.CreateAndSubmitThreadpoolWork(ThreadPoolEntry, p);      
    }
    return tasks_to_launch.size();
  }
};