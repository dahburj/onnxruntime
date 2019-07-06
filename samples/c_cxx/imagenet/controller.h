// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sync_api.h"
#include <atomic>

class Controller {
 private:
  PTP_CLEANUP_GROUP const cleanup_group_;
  TP_CALLBACK_ENVIRON env_;
  ONNXRUNTIME_EVENT event_;
  std::atomic<bool> is_running_ = true;
  char* errmsg_ = nullptr;
  static void ShutdownCallback(_Inout_ PTP_CALLBACK_INSTANCE instance, _Inout_opt_ PVOID context, _Inout_ PTP_WORK work);
 public:
  Controller();
  void SetFailBit(_In_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, _In_opt_ const char* err_msg);
  //TODO: SetEof. It can reuse the SetFailBit function
  void Wait();
  void CreateAndSubmitThreadpoolWork(_In_ ONNXRUNTIME_CALLBACK_FUNCTION callback, _In_ void* data);
};