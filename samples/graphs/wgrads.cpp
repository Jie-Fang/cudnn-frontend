TEST_CASE("Wgrad Graph", "[wgrad][graph][scale-bias-relu-wgrad][ConvBNwgrad]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("wgrad");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::HALF)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor("image").set_dim({4, 64, 16, 16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    auto S = graph.tensor(fe::graph::Tensor("scale").set_dim({1, 64, 1, 1}));
    S->generateStrides(CUDNN_TENSOR_NHWC);

    auto scale_options = fe::graph::Pointwise("scale").set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output = graph.pointwise(X, S, scale_options);
    scale_output->set_is_virtual(true);

    auto B = graph.tensor(fe::graph::Tensor("bias").set_dim({1, 64, 1, 1}));
    B->generateStrides(CUDNN_TENSOR_NHWC);
    auto bias_options = fe::graph::Pointwise("bias").set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output = graph.pointwise(scale_output, B, bias_options);
    bias_output->set_is_virtual(true);
    
    auto relu_options = fe::graph::Pointwise("relu").set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto relu_output = graph.pointwise(bias_output, relu_options);
    relu_output->set_is_virtual(true);

    auto DY = graph.tensor(fe::graph::Tensor("grad").set_dim({4, 64, 16, 16}));
    DY->generateStrides(CUDNN_TENSOR_NHWC);
    auto wgrad_options = fe::graph::Conv_wgrad("wgrad").set_padding({1,1}).set_stride({1,1}).set_dilation({1,1});
    auto DW = graph.conv_wgrad(DY, relu_output, wgrad_options);

    #if (CUDNN_VERSION < 8800)
        SKIP("ConvBNwgrad requires cudnn 8.8 and up");
    #endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("ConvBNwgrad requires hopper and above architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(fe::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*64*16*16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> dy_tensor(4*64*16*16, false);
    Surface<half> dw_tensor(64*64*3*3, false);

    auto workspace = graph.tensor(fe::graph::Tensor("workspace"));
    Surface<int8_t> workspace_tensor(graph.get_workspace_size(), false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {X, x_tensor.devPtr}
        , {S, s_tensor.devPtr}
        , {B, b_tensor.devPtr}
        , {DY, dy_tensor.devPtr}
        , {DW, dw_tensor.devPtr}
        , {workspace, workspace_tensor.devPtr}
    };
    REQUIRE(fe::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}