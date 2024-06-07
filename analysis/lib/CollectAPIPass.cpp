#include "FunctionInfo.h"

PreservedAnalyses CollectAPIPass::run(Module &M, ModuleAnalysisManager &) {
    std::vector<Type*> types;
    outs() << "potential converter API is:\n";
    for (auto &F : M) {
        // filter
        if (F.getName().contains("GLOBAL__N_")){
            // cuda global function
            continue;
        }

        types.clear();
        bool hasLow=false, hasHigh=false;
        // Print argument types
        for (auto &Arg : F.args()) {
            types.push_back(Arg.getType());
        }
        types.push_back(F.getReturnType());

        for(Type *Ty:types){
            // outs() << getTypeName(Ty) << "\n";
            for (auto low_type:strLowType){
                if (getTypeName(Ty).find(low_type) != std::string::npos){
                    hasLow = true;
                }
            }

            if(Ty->isFloatTy() || getTypeName(Ty).compare("float*")==0){
                hasHigh = true;
            }
        }
        if (hasLow && hasHigh){
            printFuncInfo(F);
            // outs() << "Function: " << F.getName() << "\n";
        }

    }

    return PreservedAnalyses::all();
}

std::string CollectAPIPass::getTypeName(Type *Ty){
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    Ty->print(rso);
    return rso.str();
}

void CollectAPIPass::printFuncInfo(Function &F){
    DISubprogram* DI = F.getSubprogram();
    if(!DI) {
        outs() << "Function " << F.getName() << " does not have a subprogram\n\n";
        return;
    }
    DIFile* DF = DI->getFile();

    if (!DF->getFilename().contains("c10") 
        // && !DF->getFilename().contains("tensorflow")
        && !DF->getFilename().contains("Eigen")) {
        // only care appraoch in c10 in Pytorch
        // only care approach in Eigen in Tensorflow
        // outs() << "Function " << F.getName() << " does not in interesting source code\n\n"; 
        return;
    }
    outs() << "Function name: " << DI->getName() <<"\n";
    outs() << "Linkage name: " << DI->getLinkageName() <<"\n";
    outs() << "Location: " << DF->getFilename() << ":" << DI->getLine() << "\n\n";
}
