// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "controller.h"

void Controller::ShutdownCallback(_Inout_ PTP_CALLBACK_INSTANCE instance, _Inout_opt_ PVOID context, _Inout_ PTP_WORK work) {
  //For each Controller, only one thread can reach here
  OnnxRuntimeCloseThreadpoolWork(work);
  Controller* c = reinterpret_cast<Controller*>(context);
  
  
}
Controller::Controller() : cleanup_group_(CreateThreadpoolCleanupGroup()), event_(CreateOnnxRuntimeEvent()) {
  InitializeThreadpoolEnvironment(&env_);
  SetThreadpoolCallbackPool(&env_, nullptr);
  SetThreadpoolCallbackCleanupGroup(&env_, cleanup_group_, nullptr);
}

void Controller::CreateAndSubmitThreadpoolWork(ONNXRUNTIME_CALLBACK_FUNCTION callback, void* data) {
  ::CreateAndSubmitThreadpoolWork(callback, data, &env_);
}

void Controller::Wait() { 
	WaitAndCloseEvent(event_);
	CloseThreadpoolCleanupGroupMembers(cleanup_group_,errmsg_ == nullptr?FALSE:TRUE, nullptr);
	CloseThreadpoolCleanupGroup(cleanup_group_);
}

void Controller::SetFailBit(_In_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci, _In_opt_ const char* err_msg) {
  bool old_value = true;
  if (is_running_.compare_exchange_strong(old_value, false)) {
    if (err_msg != nullptr) {
      errmsg_ = my_strdup(err_msg);
    }
	::CreateAndSubmitThreadpoolWork(ShutdownCallback, this, nullptr);
  }
  
}