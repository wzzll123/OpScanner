#ifndef FUNCTION_INFO_PASS_H
#define FUNCTION_INFO_PASS_H

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>

using namespace llvm;

class CollectAPIPass final : public PassInfoMixin<CollectAPIPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  void printFuncInfo(Function &F);
private:
  std::vector<std::string> strLowType = {"struct.c10::Half","struct.c10::BFloat16","bfloat16","struct.Eigen::half"};
  std::vector<std::string> strHighType = {"float"};
  std::string getTypeName(Type *Ty);
  
};


class ScanAPIPass final : public PassInfoMixin<ScanAPIPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  std::string getSourceLoc(Instruction &inst);
private:
  std::map<StoreInst*, int> visited_store;
  std::map<std::string, int> all_warnings_loc;
  std::map<std::string, int> implicit_call_loc;
  std::map<std::string, std::string> strIncreaseAPI = {
    // half
    {"_ZNK3c104HalfcvfEv", "operator float"}, 
    {"_ZN3c10mlENS_4HalfEf", "operator*"},
    {"_ZN3c10miEfNS_4HalfE", "operator-"},
    {"_ZN3c10plEfNS_4HalfE", "operator+"},
    {"_ZN3c10dvENS_4HalfEf", "operator/"},

    {"_ZN3c10mlEfNS_4HalfE", "operator*"},
    {"_ZN3c10miENS_4HalfEf", "operator-"},
    {"_ZN3c10plENS_4HalfEf", "operator+"},
    {"_ZN3c10dvEfNS_4HalfE", "operator/"},

    {"_ZN3c10pLERfRKNS_4HalfE", "operator+="},
    {"_ZN3c10mIERfRKNS_4HalfE", "operator-="},
    {"_ZN3c10mLERfRKNS_4HalfE", "operator*="},

    // Eigen::half
    {"_ZNK5Eigen4halfcvfEv", "operator float"},
    {"_ZNK5Eigen8internal14scalar_cast_opINS_4halfEfEclERKS2_", "cast_op"},
    {"_ZNK10tensorflow7functor11HalfToFloatclERKN5Eigen4halfE", "HalfToFloat"},
    {"_ZN10tensorflow12_GLOBAL__N_19GetScalarIfEET_RKNS_6TensorE", "getScalar<float>"},
    {"_ZN10tensorflow12_GLOBAL__N_111strict_castIfN5Eigen4halfEEENSt9enable_ifIXntsr3std7is_sameIT0_T_EE5valueES6_E4typeES5_", "strict_cast_half_to_float"},

    // bfloat
    // {"_ZNK3c108BFloat16cvfEv", "operator float"}, 
    // {"_ZN3c10mlENS_8BFloat16Ef", "operator*"},
    // {"_ZN3c10miENS_8BFloat16Ef", "operator-"},
    // {"_ZN3c10plENS_8BFloat16Ef", "operator+"},
    // {"_ZN3c10dvENS_8BFloat16Ef", "operator/"},

    // {"_ZN3c10mlEfNS_8BFloat16E", "operator*"},
    // {"_ZN3c10miEfNS_8BFloat16E", "operator-"},
    // {"_ZN3c10plEfNS_8BFloat16E", "operator+"},
    // {"_ZN3c10dvEfNS_8BFloat16E", "operator/"},

    // {"_ZN3c10pLERfRKNS_8BFloat16E", "operator+="},
    // {"_ZN3c10mIERfRKNS_8BFloat16E", "operator-="},
    // {"_ZN3c10mLERfRKNS_8BFloat16E", "operator*="},
  };
  void report(CallBase &I);
  void report(CallBase &I, User* user);
  void report(CallBase &I, std::vector<User*> users);

  // for capturing method call that implicitly converts type, for example, std::abs(weight_val)
  User* getUserFromDefUserChain(StoreInst *);
  User* getUserFromStore(StoreInst *storeInst);
  std::vector<User*> getUsers(Instruction &I);
  std::vector<User*> getUsersFromDefUserChain(StoreInst *storeInst);
  std::vector<User*> getUsersFromStore(StoreInst *storeInst);
  User* getFirstUser(Instruction &I);
  void printFuncInfo(const Function &F);

};

#endif // FUNCTION_INFO_PASS_H
