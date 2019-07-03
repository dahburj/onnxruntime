// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <functional>

/**
 * A special FIFO that is restricted to only have one consumer
 * The consumer must return the previous borrowed item before taking the next
 */
template <typename ValueType>
class SingleConsumerFIFO {
 public:
  struct ListEntry {
    ValueType value;
    ListEntry* next = nullptr;
  };

 private:
  // fixed size
  ListEntry* values_;
  ListEntry* free_list_ = nullptr;
  // whenever free_list_ is nullptr, free_list_tail_ should equal to &free_list_;
  ListEntry** free_list_tail_ = &free_list_;
  bool is_consumer_running_ = false;
  size_t len_;

 public:
  explicit SingleConsumerFIFO(size_t len) : values_(new ListEntry[len]), len_(len) {}
  ~SingleConsumerFIFO() noexcept { delete[] values_; }

  template <typename T>
  void Init(const T& t) {
    for (size_t i = 0; i != len_; ++i) {
      t(values_[i]);
    }
  }

  /**
   * Return a borrowed item
   * @param e a pointer returned from the Take() function
   * @return ID of the entry, in [0,len)
   */
  size_t Return(ListEntry* e) {
    is_consumer_running_ = false;
    return e - values_;
  }

  ListEntry* Put(size_t element_id) {
    assert(element_id < len_);
    // printf("Append %zd to the free list\n", element_id);
    ListEntry* t = &values_[element_id];
    t->next = nullptr;
    (*free_list_tail_) = t;
    free_list_tail_ = &t;
    return t;
  }

  ListEntry* Take() {
    if (is_consumer_running_) return nullptr;
    if (free_list_ == nullptr) {
      is_consumer_running_ = false;
      return nullptr;
    }
    auto input_tensor = free_list_;
    is_consumer_running_ = true;
    if ((free_list_ = free_list_->next) == nullptr) free_list_tail_ = &free_list_;
    return input_tensor;
  }
};