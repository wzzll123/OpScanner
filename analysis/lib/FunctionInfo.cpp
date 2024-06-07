#include "FunctionInfo.h"

using namespace llvm;

extern "C" PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
      .APIVersion = LLVM_PLUGIN_API_VERSION,
      .PluginName = "FunctionInfo",
      .PluginVersion = LLVM_VERSION_STRING,
      .RegisterPassBuilderCallbacks =
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) -> bool {
                  if (Name == "collect-api") {
                    MPM.addPass(CollectAPIPass());
                    return true;
                  }
                  if (Name == "scan-api") {
                    MPM.addPass(ScanAPIPass());
                    return true;
                  }
                  return false;
                });
          } // RegisterPassBuilderCallbacks
  };        // struct PassPluginLibraryInfo
}
