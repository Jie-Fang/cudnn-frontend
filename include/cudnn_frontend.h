#pragma once

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_Heuristics.h"
#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_EngineConfig.h"
#include "cudnn_frontend_EngineFallbackList.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_Filters.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_VariantPack.h"

namespace cudnn_frontend {
    using Tensor = Tensor_v8;
    using TensorBuilder = TensorBuilder_v8;
    using ConvDesc = ConvDesc_v8;
    using ConvDescBuilder = ConvDescBuilder_v8;
    using Operation = Operation_v8;
    using OperationBuilder = OperationBuilder_v8;
    using OperationGraph = OperationGraph_v8;
    using OperationGraphBuilder = OperationGraphBuilder_v8;
    using EngineHeuristicsBuilder = EngineHeuristicsBuilder_v8;
    using EngineHeuristics = EngineHeuristics_v8;
    using EngineBuilder = EngineBuilder_v8;
    using Engine = Engine_v8;    
    using EngineConfig = EngineConfig_v8;
    using EngineConfigBuilder = EngineConfigBuilder_v8;
    using ExecutionPlan = ExecutionPlan_v8;
    using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;
    using VariantPack = VariantPack_v8;
    using VariantPackBuilder = VariantPackBuilder_v8;
    using EngineFallbackList = EngineFallbackList_v8;
    using EngineFallbackListBuilder = EngineFallbackListBuilder_v8;
}
