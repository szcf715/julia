// This file is a part of Julia. License is MIT: https://julialang.org/license

// Function multi-versioning
#define DEBUG_TYPE "julia_multiversioning"
#undef DEBUG

// LLVM pass to clone function for different archs

#include "llvm-version.h"
#include "support/dtypes.h"

#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include "fix_llvm_assert.h"

#include "julia.h"
#include "julia_internal.h"
#include "processor.h"

#include <unordered_map>
#include <vector>

using namespace llvm;

extern std::pair<MDNode*,MDNode*> tbaa_make_child(const char *name, MDNode *parent=nullptr,
                                                  bool isConstant=false);

namespace {

struct MultiVersioning: public ModulePass {
    struct CloneCtx {
        struct Target {
            uint32_t flags;
            ValueToValueMapTy VMap;
            std::vector<std::pair<const Function*,Function*>> funcs;
        };
        bool has_veccall;
        std::vector<Target> targets;
        CloneCtx(size_t ntargets)
            : has_veccall(false),
              targets(ntargets)
        {
        }
    };
    static char ID;
    MultiVersioning()
        : ModulePass(ID)
    {}

private:
    bool runOnModule(Module &M) override;
    void getAnalysisUsage(AnalysisUsage &AU) const override
    {
        AU.addRequired<LoopInfoWrapperPass>();
        AU.addRequired<CallGraphWrapperPass>();
        // AU.addPreserved<LoopInfoWrapperPass>();
    }
    void handleFunction(Function &F, CloneCtx &clone);
    void expandCloning(ValueToValueMapTy &VMap);
};

static bool isVectorFunction(FunctionType *ty)
{
    if (ty->getReturnType()->isVectorTy())
        return true;
    for (auto arg: ty->params()) {
        if (arg->isVectorTy()) {
            return true;
        }
    }
    return false;
}

void MultiVersioning::handleFunction(Function &F, CloneCtx &clone)
{
    if (F.empty())
        return;
    bool hasLoop = !getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo().empty();
    bool vecF = isVectorFunction(F.getFunctionType());
    if (vecF)
        clone.has_veccall = true;
    std::vector<CloneCtx::Target*> math_targets;
    std::vector<CloneCtx::Target*> simd_targets;
    std::vector<CloneCtx::Target*> math_simd_targets;
    for (auto &ele: clone.targets) {
        if (ele.flags & JL_TARGET_CLONE_ALL ||
            (ele.flags & JL_TARGET_CLONE_LOOP && hasLoop)) {
            ele.VMap[&F] = &F;
        }
        else if (ele.flags & JL_TARGET_CLONE_SIMD) {
            if (vecF) {
                ele.VMap[&F] = &F;
            }
            else if (ele.flags & JL_TARGET_CLONE_MATH) {
                math_simd_targets.push_back(&ele);
            }
            else {
                simd_targets.push_back(&ele);
            }
        }
        else if (ele.flags & JL_TARGET_CLONE_MATH) {
            math_targets.push_back(&ele);
        }
    }
    bool done = false;
    auto update = [&] {
        done = clone.has_veccall;
        done = done && math_targets.empty() && simd_targets.empty() && math_simd_targets.empty();
    };
    auto found_math = [&] {
        for (auto target: math_simd_targets) {
            target->VMap[&F] = &F;
        }
        math_simd_targets.clear();
        for (auto target: math_targets) {
            target->VMap[&F] = &F;
        }
        math_targets.clear();
        update();
    };
    auto found_simd = [&] (bool call) {
        if (call)
            clone.has_veccall = true;
        for (auto target: math_simd_targets) {
            target->VMap[&F] = &F;
        }
        math_simd_targets.clear();
        for (auto target: simd_targets) {
            target->VMap[&F] = &F;
        }
        simd_targets.clear();
        update();
    };

    for (auto &bb: F) {
        for (auto &I: bb) {
            if (auto call = dyn_cast<CallInst>(&I)) {
                if (isVectorFunction(call->getFunctionType())) {
                    found_simd(true);
                }
                if (auto callee = call->getCalledFunction()) {
                    auto name = callee->getName();
                    if (name.startswith("llvm.muladd.") || name.startswith("llvm.fma.")) {
                        found_math();
                    }
                }
            }
            else if (auto store = dyn_cast<StoreInst>(&I)) {
                if (store->getValueOperand()->getType()->isVectorTy()) {
                    found_simd(false);
                }
            }
            else if (I.getType()->isVectorTy()) {
                found_simd(false);
            }
            if (auto mathOp = dyn_cast<FPMathOperator>(&I)) {
                if (mathOp->getFastMathFlags().any()) {
                    found_math();
                }
            }
            if (done) {
                return;
            }
        }
    }
}

static void addFeatures(Function *F, const std::string &name, const std::string &features)
{
    auto attr = F->getFnAttribute("target-features");
    if (attr.isStringAttribute()) {
        std::string new_features = attr.getValueAsString();
        new_features += ",";
        new_features += features;
        F->addFnAttr("target-features", new_features);
    }
    else {
        F->addFnAttr("target-features", features);
    }
    F->addFnAttr("target-cpu", name);
}

// Reduce dispatch by expand the cloning set to functions that are directly called by
// and calling cloned functions.
void MultiVersioning::expandCloning(ValueToValueMapTy &VMap)
{
    std::set<const Function*> sets[2];
    auto &graph = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    for (auto v: VMap)
        sets[0].insert(cast<Function>(v.first));
    auto *cur_set = &sets[0];
    auto *next_set = &sets[1];
    while (!cur_set->empty()) {
        for (auto f: *cur_set) {
            auto node = graph[f];
            for (const auto &I: *node) {
                auto child_node = I.second;
                auto child_f = child_node->getFunction();
                if (!child_f)
                    continue;
                if (VMap.find(child_f) != VMap.end())
                    continue;
                bool calling_clone = false;
                for (const auto &I2: *child_node) {
                    auto child_f2 = I2.second->getFunction();
                    if (!child_f2)
                        continue;
                    if (VMap.find(child_f2) != VMap.end()) {
                        calling_clone = true;
                        break;
                    }
                }
                if (!calling_clone)
                    continue;
                next_set->insert(child_f);
                VMap[child_f] = child_f;
            }
        }
        std::swap(cur_set, next_set);
        next_set->clear();
    }
}

static Constant *stripCast(Constant *v)
{
    if (auto c = cast<ConstantExpr>(v)) {
        auto opcode = c->getOpcode();
        bool iscast = opcode == Instruction::Trunc || opcode == Instruction::ZExt ||
            opcode == Instruction::SExt || opcode == Instruction::BitCast ||
            opcode == Instruction::PtrToInt || opcode == Instruction::IntToPtr ||
            opcode == Instruction::AddrSpaceCast;
        if (iscast) {
            return stripCast(c->getOperand(0));
        }
    }
    return v;
}

static std::pair<Function*,ConstantExpr*> getFunction(ConstantExpr *v, Function *base_func)
{
    assert(v->hasOneUse());
    v = cast<ConstantExpr>(stripCast(v));
    assert(v->hasOneUse());
    assert(v->getOpcode() == Instruction::Sub);
    ConstantExpr *f = cast<ConstantExpr>(v->getOperand(0));
    assert(f->hasOneUse());
    assert(f->getOpcode() == Instruction::PtrToInt);
    Constant *base = v->getOperand(1);
    (void)base;
    assert(stripCast(base) == base_func);
    return std::make_pair(cast<Function>(f->getOperand(0)), f);
}

void checkUses(const Constant *parent, Constant *fvar_use, Constant *fbase, bool samebits=true)
{
    for (auto *user: parent->users()) {
        if (isa<Instruction>(user))
            continue;
        if (fvar_use && fvar_use == user)
            continue;
        if (auto gv = dyn_cast<GlobalVariable>(user)) {
            if (!samebits || gv->isConstant())
                user->dump();
            continue;
        }
        if (user == fbase)
            continue;
        if (auto expr = dyn_cast<ConstantExpr>(user)) {
            bool samebits2 = samebits;
            if (samebits2) {
                auto opcode = expr->getOpcode();
                samebits2 = opcode == Instruction::BitCast || opcode == Instruction::PtrToInt ||
                    opcode == Instruction::IntToPtr || opcode == Instruction::AddrSpaceCast;
            }
            checkUses(expr, fvar_use, fbase, samebits2);
            continue;
        }
        else if (auto aggr = dyn_cast<ConstantAggregate>(user)) {
            checkUses(aggr, fvar_use, fbase, samebits);
            continue;
        }
        user->dump();
    }
}

bool MultiVersioning::runOnModule(Module &M)
{
    // MDNode *tbaa_const = tbaa_make_child("jtbaa_const", nullptr, true).first;
    GlobalVariable *fvars = M.getGlobalVariable("jl_sysimg_fvars_offsets");
    auto fbase = cast<GlobalAlias>(M.getNamedValue("jl_sysimg_fvars_base"));
    // auto gbase = cast<GlobalAlias>(M.getNamedValue("jl_sysimg_gvars_base"));
    // Makes sure this only runs during sysimg generation
    assert(fvars && fvars->hasInitializer());
    LLVMContext &ctx = M.getContext();
    auto T_size = (sizeof(size_t) == 8 ? Type::getInt64Ty(ctx) : Type::getInt32Ty(ctx));

    auto clone_targets = jl_get_llvm_clone_targets(JL_LLVM_VERSION);
    size_t ntargets = clone_targets.size();
    CloneCtx clone(ntargets);
    clone.targets[0].flags = 0;
    // The first target does not needs cloning.
    for (size_t i = 1; i < ntargets; i++)
        clone.targets[i].flags = clone_targets[i].flags;
    // First decide the set of functions that needs to be cloned.
    // The heuristic here is really simple but should cover many of the useful cases.
    size_t nfunc = 0;
    for (auto &F: M) {
        if (!F.empty())
            nfunc++;
        handleFunction(F, clone);
    }
    // Expand the clone set
    for (size_t i = 1; i < ntargets; i++) {
        auto &t = clone.targets[i];
        if (t.flags & JL_TARGET_CLONE_ALL)
            continue;
        expandCloning(t.VMap);
    }
    // Fill in old->new mapping. We need to do this before cloning the function so that
    // the intra target calls are automatically fixed up on cloning.
    for (size_t i = 1; i < ntargets; i++) {
        auto &t = clone.targets[i];
        auto suffix = ".clone_" + std::to_string(i);
        for (auto v: t.VMap) {
            auto orig_f = cast<Function>(v.second);
            Function *new_f = Function::Create(orig_f->getFunctionType(), orig_f->getLinkage(),
                                               orig_f->getName() + suffix, &M);
            new_f->copyAttributesFrom(orig_f);
            v.second = new_f;
        }
    }
    // Actually clone the functions and add feature strings.
    std::unordered_map<const Function*,std::pair<size_t,Constant*>> idx_map;
    for (size_t i = 1; i < ntargets; i++) {
        auto &t = clone.targets[i];
        // Copy the function list out first since we'll mutate the map and put many other
        // stuff in it.
        for (auto v: t.VMap) {
            auto orig_f = cast<Function>(v.first);
            auto new_f = cast<Function>(v.second);
            t.funcs.push_back(std::make_pair(orig_f, new_f));
            // The assignment is probably not necessary
            idx_map[orig_f] = std::make_pair(0, (Constant*)0);
        }
        for (auto v: t.funcs) {
            auto orig_f = v.first;
            auto new_f = v.second;
            Function::arg_iterator DestI = new_f->arg_begin();
            for (Function::const_arg_iterator J = orig_f->arg_begin();
                 J != orig_f->arg_end(); ++J) {
                DestI->setName(J->getName());
                t.VMap[&*J] = &*DestI++;
            }
            SmallVector<ReturnInst*,8> Returns;
            CloneFunctionInto(new_f, orig_f, t.VMap, false, Returns);
            addFeatures(new_f, clone_targets[i].cpu_name, clone_targets[i].cpu_features);
        }
    }

    // At this point, all the calls between cloned functions should have been rewritten
    // to correct direct calls. It's time to handle all other uses of the cloned functions.
    // The cases we can currently handle are:
    //
    // 1. Use in functions
    // 2. fvar offsets
    // 3. global variable initializer (without offset)

    // Collect function index and corresponding use.
    // This is needed to detect fvar related uses since it's otherwise hard to do.
    auto base_func = cast<Function>(fbase->getAliasee());
    auto *fary = cast<ConstantArray>(fvars->getInitializer());
    size_t nf = fary->getNumOperands() - 1; // The fist element is a size
    auto base_int = ConstantExpr::getPtrToInt(base_func, T_size);
    assert(base_int->hasNUses(nf - 1));
    std::vector<Function*> func_map(nf);
    idx_map[base_func] = std::make_pair(1, base_int);
    func_map[0] = base_func;
    for (size_t i = 2; i < nf + 1; i++) {
        auto ele = getFunction(cast<ConstantExpr>(fary->getOperand(i)), base_func);
        assert(ele.first && ele.second);
        auto it = idx_map.find(ele.first);
        if (it != idx_map.end()) {
            assert(!it->second.second);
            it->second = std::make_pair(i, ele.second);
        }
        func_map[i - 1] = ele.first;
    }

    // Now recursively walk the use list of each cloned functions and fix up the references
    //
    // 1. We walk the use list until we hit either a `GlobalValue` or an `Instruction`.
    //    Other than the use in fvars including the base pointer and the offsets
    //    (these uses will be checked directly)
    //    these are the two kinds of uses that are supported.
    //
    //     1. For `Instruction`,
    //        all `ConstantExpr` and `ConstantAggregate`s in the use chain are supported.
    //        If the use in the function needs a runtime dispatch (see below),
    //        it will be converted to corresponding `Instruction` versions initialized by
    //        a constant load from the corresponding GOT slot.
    //
    //     2. For `GlobalValue`, only non-constant `GlobalVariable` initializers are supported.
    //        Also, only `ConstantAggregate` and `ConstantExpr` that does not change the bits
    //        of the pointer (`PtrToInt`, `IntToPtr`, `BitCast`, `AddrSpaceCast`) are supported.
    //        The initializer of the global variables will be replaced with `null`.
    //        The address (offset) of the global variable will be added to the additional GOT
    //        offset list so that it can be initialized on loading.
    //        `ConstantAggregate` in global variable appears on ARM and AArch64 due to
    //        `GlobalMerge` pass.
    //
    //     Unsupported use will abort the compilation.
    //
    // 2. A `Instruction` use of a cloned function may need a runtime dispatch through a GOT slot.
    //    The rules to decide if such a slot is necessary for a function/callsite are,
    //
    //     1. A function needs a GOT slot if a cloned version is called from an uncloned
    //        function for at least one target.
    //     2. A use in a cloned function never need fixing.
    //     3. A use in an uncloned function is only needed if at least one (non-default) target
    //        do not have this function cloned.
    //
    // 3. If the function is in fvar and needs a GOT,
    //    a target needs the GOT slot to be initialized iff at least one caller
    //    of the function is not cloned.
    //
    //     For other functions, the GOT slot is always initialized.
    //
    for (auto it: idx_map) {
        auto orig_f = it.first;
        checkUses(orig_f, it.second.second, fbase);
    }


    // 1. Handle the use in functions. For each cloned functions (entry in idx_map)
    //    it needs to have a GOT if one of it's uncloned caller may call both cloned
    //    and uncloned versions.

    // Now go through each cloned function and allocate GOT slot/record relocation if necessary

    for (auto &t: clone.targets) {
        jl_safe_printf("%zu / %zu\n", t.funcs.size(), nfunc);
    }

    // TODO
    return true;

    // std::vector<Constant*> ptrs;
    // std::vector<Constant*> idxs;
    // auto T_void = Type::getVoidTy(ctx);
    // auto T_pvoidfunc = FunctionType::get(T_void, false)->getPointerTo();
    // for (auto I: idx_map) {
    //     auto oldF = I.first;
    //     auto idx = I.second;
    //     auto newF = cast<Function>(VMap[oldF]);
    //     ptrs.push_back(ConstantExpr::getBitCast(newF, T_pvoidfunc));
    //     auto offset = ConstantInt::get(T_size, idx);
    //     idxs.push_back(offset);
    //     for (auto user: oldF->users()) {
    //         auto inst = dyn_cast<Instruction>(user);
    //         if (!inst)
    //             continue;
    //         auto encloseF = inst->getParent()->getParent();
    //         if (VMap.find(encloseF) != VMap.end())
    //             continue;
    //         Value *slot = ConstantExpr::getBitCast(fvars, T_pvoidfunc->getPointerTo());
    //         slot = GetElementPtrInst::Create(T_pvoidfunc, slot, {offset}, "", inst);
    //         Instruction *ptr = new LoadInst(slot, "", inst);
    //         ptr->setMetadata(llvm::LLVMContext::MD_tbaa, tbaa_const);
    //         ptr = new BitCastInst(ptr, oldF->getType(), "", inst);
    //         inst->replaceUsesOfWith(oldF, ptr);
    //     }
    // }
}

char MultiVersioning::ID = 0;
static RegisterPass<MultiVersioning> X("JuliaMultiVersioning", "JuliaMultiVersioning Pass",
                                       false /* Only looks at CFG */,
                                       false /* Analysis Pass */);

}

Pass *createMultiVersioningPass()
{
    return new MultiVersioning();
}
