#include "FunctionInfo.h"

PreservedAnalyses ScanAPIPass::run(Module &M, ModuleAnalysisManager &) {
    for (auto &F : M) {
        for (BasicBlock &bb : F) {
            for (Instruction &I : bb){
                if (isa<CallBase>(I)) {
                    CallBase &call_inst = cast<CallBase>(I);
                    Function *targetFunction = call_inst.getCalledFunction();
                    if(targetFunction == nullptr){
                        continue;
                    }
                    std::string funcName = targetFunction->getName().str();
                    for(const auto& n:strIncreaseAPI){
                        if(funcName.compare(n.first)==0){
                            // report(call_inst);


                            User* user = nullptr;
                            if(n.second.find_first_of("+-*/") != std::string::npos){
                                // for operator += -= *=, self is the first user
                                user = &cast<User>(I);
                                report(call_inst, user);
                            }
                            else{
                                auto users = getUsers(I);
                                report(call_inst, users);
                            }
                            
                        }
                    }
                }
                
            }
        }
        
    }
    return PreservedAnalyses::all();
}
std::string ScanAPIPass::getSourceLoc(Instruction &inst){
    std::string str;
    raw_string_ostream rawPath(str);
    unsigned line = 0;
    if (MDNode* instNd = inst.getMetadata("dbg")){
        auto instLoc = dyn_cast<DILocation>(instNd);
        rawPath << instLoc->getDirectory() << "/" << instLoc->getFilename();
        line = instLoc->getLine();
        rawPath << ":" << line;
    }
    return rawPath.str();
}
void ScanAPIPass::report(CallBase &inst){
    std::string inst_loc = getSourceLoc(inst);
    if (all_warnings_loc.count(inst_loc))
        return;
    all_warnings_loc[inst_loc] = 1;
    if(inst_loc.find("core/kernel")==std::string::npos){
        return;
    }
    outs() << "\n**************\n";
    outs() << "Variable at function: " << inst.getParent()->getParent()->getName().str() << "\n";
    outs() << "Increse at: " << inst_loc << "\n";
    outs() << "Inst is: " << inst << "\n";
    outs() << "Use API: " << strIncreaseAPI[inst.getCalledFunction()->getName().str()] << "\n";
    outs() << "**************\n\n";
}

void ScanAPIPass::report(CallBase &inst, std::vector<User*> users){
    std::string inst_loc = getSourceLoc(inst);
    if (all_warnings_loc.count(inst_loc))
        return;
    all_warnings_loc[inst_loc] = 1;
    if(inst_loc.find("native")==std::string::npos){
        return;
    }
    outs() << "\n**************\n";
    outs() << "Variable at function: " << inst.getParent()->getParent()->getName().str() << "\n";
    outs() << "Increse at: " << inst_loc << "\n";
    outs() << "Inst is: " << inst << "\n";
    outs() << "Use API: " << strIncreaseAPI[inst.getCalledFunction()->getName().str()] << "\n";
    if (std::distance(users.begin(),users.end()) == 0){
        outs() << "no appropriate user\n";
    }
    else{
        int num = 0;
        for(auto user:users){
            num++;
            outs() << "The " << num << "st user: "<< *user << "\n";
            outs() << "Location is: " << getSourceLoc(*cast<Instruction>(user)) << "\n\n";
            // if(isa<CallBase>(user)){
            //     printFuncInfo(*cast<CallBase>(user)->getCalledFunction());
            // }
        }

    }
    outs() << "**************\n\n";
}

void ScanAPIPass::report(CallBase &inst, User* user){
    std::string inst_loc = getSourceLoc(inst);
    if (all_warnings_loc.count(inst_loc))
        return;
    all_warnings_loc[inst_loc] = 1;
    if(inst_loc.find("native")==std::string::npos){
        return;
    }
    outs() << "\n**************\n";
    outs() << "Variable at function: " << inst.getParent()->getParent()->getName().str() << "\n";
    outs() << "Increse at: " << inst_loc << "\n";
    outs() << "Inst is: " << inst << "\n";
    outs() << "Use API: " << strIncreaseAPI[inst.getCalledFunction()->getName().str()] << "\n";
    if (user == nullptr){
        outs() << "no appropriate user\n";
    }
    else{
        outs() << "The 1st user: "<< *user << "\n";
        outs() << "Location is: " << getSourceLoc(*cast<Instruction>(user)) << "\n\n";
        // if(isa<CallBase>(user)){
        //     printFuncInfo(*cast<CallBase>(user)->getCalledFunction());
        // }
    }
    outs() << "**************\n\n";
}

User* ScanAPIPass::getFirstUser(Instruction &I){
    // if(std::distance(I.user_begin(),I.user_end()) != 1){
    //     outs() << "Instruction " << I << " contains more than one users\n";
    //     return nullptr;
    // }
    for (auto user: I.users()) {
        if (isa<StoreInst>(user)){
            return getUserFromDefUserChain(cast<StoreInst>(user));
        } else if(isa<PHINode>(user)){
            return getFirstUser(*cast<Instruction>(user));
        } else{
            return user;
        }
    }
}

std::vector<User*> ScanAPIPass::getUsers(Instruction &I){
    std::vector<User*> result;
    for (auto user: I.users()) {
        if (isa<StoreInst>(user)){
            auto tmp = getUsersFromDefUserChain(cast<StoreInst>(user));
            result.insert(result.end(),tmp.begin(),tmp.end());
        } else if(isa<PHINode>(user)){
            auto tmp = getUsers(*cast<Instruction>(user));
            result.insert(result.end(),tmp.begin(),tmp.end());
        } else{
            result.emplace_back(user);
        }
    }
    return result;
}

std::vector<User*> ScanAPIPass::getUsersFromDefUserChain(StoreInst *storeInst){
    std::vector<User*> result;
    if (visited_store.count(storeInst))
        return result;
    visited_store[storeInst] = 1;
    auto UsersOfStore = getUsersFromStore(storeInst);
    for(auto userOfStore: UsersOfStore){
        if(!isa<LoadInst>(userOfStore)){
            result.emplace_back(userOfStore);
        }
        // if loadInst
        for(auto user: userOfStore->users()){
            if(isa<StoreInst>(user)){
                auto tmp = getUsersFromDefUserChain(cast<StoreInst>(user));
                result.insert(result.end(),tmp.begin(),tmp.end()); 
            } else if(isa<PHINode>(user)){
                auto tmp = getUsers(*cast<Instruction>(user));
                result.insert(result.end(),tmp.begin(),tmp.end()); 
            } else{
                result.emplace_back(user);
            }
        }
    }
    return result;
}

User* ScanAPIPass::getUserFromDefUserChain(StoreInst *storeInst){
    User* UserOfStore = getUserFromStore(storeInst);
    while(UserOfStore){
        if (!isa<LoadInst>(UserOfStore)){
            return UserOfStore;
        }
        
        for (auto user: UserOfStore->users()) {
            if (isa<StoreInst>(user)){
                UserOfStore = getUserFromStore(cast<StoreInst>(user));
            } else if(isa<PHINode>(user)){
                return getFirstUser(*cast<Instruction>(user));
                // outs() << "Instruction " << *UserOfStore << "is phi node\n";
                // return nullptr;
            } else {
                return user;
            } 
        }
    }
    return nullptr;
}

std::vector<User*> ScanAPIPass::getUsersFromStore(StoreInst *storeInst){
    std::vector<User*> result;
    Value *pointerOperand = storeInst->getPointerOperand();
    // if(std::distance(pointerOperand->use_begin(),pointerOperand->use_end()) != 2){
    //     return nullptr;
    // }
    for (auto user: pointerOperand->users()) {
        // most are loadInst, sometimes call (pointer as argument)
        if(!isa<StoreInst>(user)){
            result.emplace_back(user);
        }
        
    }
    return result;
}

User* ScanAPIPass::getUserFromStore(StoreInst *storeInst){
    Value *pointerOperand = storeInst->getPointerOperand();
    // if(std::distance(pointerOperand->use_begin(),pointerOperand->use_end()) != 2){
    //     return nullptr;
    // }
    for (auto user: pointerOperand->users()) {
        // most are loadInst, sometimes call (pointer as argument)
        if(!isa<StoreInst>(user)){
            return user;
        }
        
    }
    return nullptr;
}

void ScanAPIPass::printFuncInfo(const Function &F){
    DISubprogram* DI = F.getSubprogram();
    if(!DI) {
        outs() << "Function name: " << F.getName() << "\n\n";
        return;
    }
    DIFile* DF = DI->getFile();
    
    outs() << "Function name: " << DI->getName() <<"\n";
    outs() << "Linkage name: " << DI->getLinkageName() <<"\n";
    outs() << "Location: " << DF->getFilename() << ":" << DI->getLine() << "\n\n";
}



