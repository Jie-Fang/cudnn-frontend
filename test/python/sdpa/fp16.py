import cudnn
import pytest
import torch
from looseversion import LooseVersion

from .fp16_ref import compute_ref
from .helpers import (
    convert_to_cudnn_type,
    exact_equal,
    approx_equal,
    alloc_tensor,
    convert_ragged_to_uniform,
    create_container_and_page_table,
    time_execution,
    profile_execution,
)

# fmt: off

def exec_sdpa(cfg, request, cudnn_handle):
    # Do not run any test when --dryrun option is provided.

    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    # ============================
    # Basic parameter check.
    # ============================

    if not all((x > 0 and type(x) == int) for x in (cfg.batches, cfg.d_qk, cfg.d_v, cfg.s_q, cfg.s_kv, cfg.h_q, cfg.h_k, cfg.h_v)):
       assert False, "tensor dimensions must be integer and positive"

    assert cfg.shape_q == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_qk), f"wrong shape_q={cfg.shape_q}"
    assert cfg.shape_k == (cfg.batches, cfg.h_k, cfg.s_kv, cfg.d_qk), f"wrong shape_k={cfg.shape_k}"
    assert cfg.shape_v == (cfg.batches, cfg.h_v, cfg.s_kv, cfg.d_v), f"wrong shape_v={cfg.shape_v}"
    assert cfg.shape_o == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_v), f"wrong shape_o={cfg.shape_o}"

    if not cfg.is_infer:
        assert cfg.is_paged == False and cfg.block_size == None, "paged attention not allowed in backward pass"

    if cfg.is_ragged:
        assert cfg.is_padding == True, "is_ragged=True and is_padding=False not allowed"

    assert isinstance(cfg.seq_len_q, (list, tuple)), "input 'seq_len_q' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_q) == cfg.batches, f"wrong 'seq_len_q' length"
    else:
        assert len(cfg.seq_len_q) == 0, f"wrong 'seq_len_q' length, expecting 0"

    assert isinstance(cfg.seq_len_kv, (list, tuple)), "input 'seq_len_kv' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_kv) == cfg.batches, f"wrong 'seq_len_kv' length, expecting {cfg.batches}"
    else:
        assert len(cfg.seq_len_kv) == 0, f"wrong 'seq_len_kv' length, expecting 0"

    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_q), f"wrong seq_len_q={cfg.seq_len_q}"
    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_kv), f"wrong seq_len_kv={cfg.seq_len_kv}"

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        print("@@@@ Overall result: WAIVED, test_mhas_v2.py supports cudnn 9.10.0 or higher.")
        pytest.skip("test_mhas_v2.py requires cudnn 9.10.0 or higher")

    if cudnn_version < "9.13.1" and cfg.implementation == cudnn.attention_implementation.UNIFIED:
        print("@@@@ Overall result: WAIVED, unified SDPA implementation requires cudnn 9.13.1 or higher.")
        pytest.skip("unified SDPA implementation requires cudnn 9.13.1 or higher")

    if cfg.s_q == cfg.s_kv == 1:
        print("@@@@ Overall result: WAIVED, skipping known issue of s_q == s_kv == 1.")
        pytest.skip("skipping known issue of s_q == s_kv == 1")

    rng_data_gen = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)

    (q_gpu, _, _) = alloc_tensor(cfg.shape_q, cfg.data_type, strides=cfg.stride_q, rng=rng_data_gen, mean=-0.5, std=1.0)
    (k_gpu, _, _) = alloc_tensor(cfg.shape_k, cfg.data_type, strides=cfg.stride_k, rng=rng_data_gen, mean=-0.5, std=1.0)
    (v_gpu, _, _) = alloc_tensor(cfg.shape_v, cfg.data_type, strides=cfg.stride_v, rng=rng_data_gen, mean=-0.5, std=1.0)
    (bias_gpu, _, _) = (alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type, rng=rng_data_gen, mean=0.0, std=1.0) if cfg.is_bias else (None, None, None))

    TILE_M = 128
    TILE_N = 128
    block_mask_gpu = torch.randint(0, 256, (cfg.batches, cfg.h_q, (cfg.s_q + TILE_M - 1) // TILE_M, ((cfg.s_kv + TILE_N - 1) // TILE_N + 7) // 8), dtype=torch.uint8, device="cuda")

    if not cfg.is_infer:
        (dQ_gpu, dQ_sep, dQ_raw) = alloc_tensor(cfg.shape_q, cfg.data_type, strides=cfg.stride_q)
        (dK_gpu, dK_sep, dK_raw) = alloc_tensor(cfg.shape_k, cfg.data_type, strides=cfg.stride_k)
        (dV_gpu, dV_sep, dV_raw) = alloc_tensor(cfg.shape_v, cfg.data_type, strides=cfg.stride_v)
        (dBias_gpu, dBias_sep, dBias_raw) = (alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type) if cfg.is_bias else (None, None, None))
        (dO_gpu, dO_sep, dO_raw) = alloc_tensor(cfg.shape_o, cfg.data_type, strides=cfg.stride_o, rng=rng_data_gen, mean=0.0, std=0.1)

    # Sequence lengths for gpu, must be a four dimensional tensor.
    seq_len_q_gpu = seq_len_kv_gpu = None
    if len(cfg.seq_len_q) > 0:
        seq_len_q_gpu = torch.tensor(cfg.seq_len_q, dtype=torch.int32, device="cuda")
        seq_len_q_gpu = seq_len_q_gpu[:, None, None, None]  # batches x 1 x 1 x 1
    if len(cfg.seq_len_kv) > 0:
        seq_len_kv_gpu = torch.tensor(cfg.seq_len_kv, dtype=torch.int32, device="cuda")
        seq_len_kv_gpu = seq_len_kv_gpu[:, None, None, None]  # batches x 1 x 1 x 1

    # maxT = next_multiple_of_64(sum(seq_len))
    max_t_q = ((torch.sum(seq_len_q_gpu).item() + 63) // 64) * 64 if cfg.is_ragged else None
    max_t_kv = ((torch.sum(seq_len_kv_gpu).item() + 63) // 64) * 64 if cfg.is_ragged else None

    if cfg.is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = torch.zeros((cfg.batches, cfg.h_q, cfg.s_q, cfg.s_kv), dtype=torch.float32, device="cuda") if cfg.is_dropout else None

    if cfg.is_ragged:
        def prefix_sum(t):
            return torch.cat((torch.zeros(1, 1, 1, 1, dtype=t.dtype, device=t.device), torch.cumsum(t, dim=0)))
        q_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * cfg.h_q * cfg.d_qk).to(torch.int64)
        k_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * cfg.h_k * cfg.d_qk).to(torch.int64)
        v_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * cfg.h_v * cfg.d_v).to(torch.int64)
        o_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * cfg.h_q * cfg.d_v).to(torch.int64)

    (o_gpu, o_sep, o_raw) = alloc_tensor(cfg.shape_o, cfg.data_type, strides=cfg.stride_o)
    (stats_gpu, stats_sep, stats_raw) = (alloc_tensor((cfg.batches, cfg.h_q, cfg.s_q, 1), torch.float32) if not cfg.is_infer else (None, None, None))

    container_k_gpu  = None
    container_v_gpu  = None
    page_table_k_gpu = None
    page_table_v_gpu = None

    if cfg.is_paged:
        container_k_gpu, page_table_k_gpu = create_container_and_page_table(k_gpu, cfg.block_size)
        container_v_gpu, page_table_v_gpu = create_container_and_page_table(v_gpu, cfg.block_size)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Forward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(cfg.data_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu) if not cfg.is_paged else graph.tensor_like(container_k_gpu)
    v = graph.tensor_like(v_gpu) if not cfg.is_paged else graph.tensor_like(container_v_gpu)

    page_table_k = graph.tensor_like(page_table_k_gpu) if cfg.is_paged else None
    page_table_v = graph.tensor_like(page_table_v_gpu) if cfg.is_paged else None

    bias = graph.tensor_like(bias_gpu) if cfg.is_bias else None
    block_mask = graph.tensor_like(block_mask_gpu) if cfg.is_block_mask else None

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if cfg.is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if cfg.is_padding else None

    if cfg.is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (cfg.dropout_prob, seed, offset)

    rng_dump = graph.tensor_like(rng_dump_gpu) if cfg.is_dropout else None

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if cfg.is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if cfg.is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if cfg.is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if cfg.is_ragged else None

    if cfg.is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    attn_scale = 0.125

    o, stats = graph.sdpa(
        name="sdpa_forward",
        q=q,
        k=k,
        v=v,
        generate_stats=not cfg.is_infer,
        attn_scale=attn_scale,
        bias=bias,
        block_mask=block_mask,
        use_alibi_mask=cfg.is_alibi,
        use_padding_mask=cfg.is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=cfg.left_bound,
        diagonal_band_right_bound=cfg.right_bound,
        diagonal_alignment=cfg.diag_align,
        dropout=dropout_tuple if cfg.is_dropout else None,
        rng_dump=rng_dump,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=cfg.s_kv if cfg.is_paged else None,
        implementation=cfg.implementation,
    )

    o.set_output(True).set_dim(cfg.shape_o).set_stride(cfg.stride_o)
    if cfg.is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if cfg.is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph. {e}")
        pytest.skip("not supported forward graph")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during forward graph validate. {e}")
        pytest.fail("unexpected exception during forward graph validate", pytrace=False)

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph after validate. {e}")
        pytest.skip("not supported forward graph after validate")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after forward validate. {e}")
        pytest.fail("unexpected exception after forward validate", pytrace=False)

    variant_pack = {
        q: q_gpu,
        k: k_gpu if not cfg.is_paged else container_k_gpu,
        v: v_gpu if not cfg.is_paged else container_v_gpu,
        bias: bias_gpu,
        block_mask: block_mask_gpu if cfg.is_block_mask else None,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if cfg.is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if cfg.is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if cfg.is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if cfg.is_ragged else None,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
        page_table_k: page_table_k_gpu,
        page_table_v: page_table_v_gpu
    }

    if cfg.is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    # Allocate workspace for the forward call.
    (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

    if request.config.getoption("--perf"):
        forward_times_ms = time_execution(graph.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ Forward graph.execute avg_time_ms={forward_times_ms.mean().item():.3f}")
        profile_execution(graph.execute, variant_pack, workspace, cudnn_handle)

    # Execute forward cuDNN graph
    graph.execute(variant_pack, workspace, cudnn_handle)
    torch.cuda.synchronize()

    if ws_sep is not None and not torch.all(ws_sep==-1).item():
        print("@@@@ Overall result: FAILED, forward workspace overwritten outside its boundaries.")
        print(ws_sep)
        pytest.fail("forward workspace overwritten outside boundaries", pytrace=False)

    diffs = request.config.getoption("--diffs") or 10

    if not cfg.is_infer:
        if cudnn_version < "8.9.6" and cfg.is_padding:
            # zero out padded region of the output and stats
            for i, m in enumerate(seq_len_q_gpu):
                o_gpu[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

        stream = torch.cuda.current_stream().cuda_stream  #2
        cudnn.set_stream(handle=cudnn_handle, stream=stream)
        sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]

        # Backward cuDNN graph
        graph = cudnn.pygraph(
            io_data_type=convert_to_cudnn_type(cfg.data_type),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            sm_version = sm_version
        )

        q = graph.tensor_like(q_gpu)
        k = graph.tensor_like(k_gpu)
        v = graph.tensor_like(v_gpu)
        o = graph.tensor_like(o_gpu)
        dO = graph.tensor_like(dO_gpu)
        stats = graph.tensor_like(stats_gpu)

        bias = graph.tensor_like(bias_gpu) if cfg.is_bias else None
        dBias = (graph.tensor_like(dBias_gpu).set_stride((cfg.h_q * cfg.s_q * cfg.s_kv, cfg.s_q * cfg.s_kv, cfg.s_kv, 1)) if cfg.is_bias else None)

        seq_len_q = graph.tensor_like(seq_len_q_gpu) if cfg.is_padding else None
        seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if cfg.is_padding else None

        if cfg.is_dropout:
            seed = graph.tensor_like(seed_gpu)
            offset = graph.tensor_like(offset_gpu)
            dropout_tuple = (cfg.dropout_prob, seed, offset)

        q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if cfg.is_ragged else None
        k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if cfg.is_ragged else None
        v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if cfg.is_ragged else None
        o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if cfg.is_ragged else None

        if cfg.is_ragged:
            q.set_ragged_offset(q_ragged_offset)
            k.set_ragged_offset(k_ragged_offset)
            v.set_ragged_offset(v_ragged_offset)
            o.set_ragged_offset(o_ragged_offset)
            dO.set_ragged_offset(o_ragged_offset)

        dQ, dK, dV = graph.sdpa_backward(
            name="sdpa_backward",
            q=q,
            k=k,
            v=v,
            o=o,
            dO=dO,
            stats=stats,
            attn_scale=attn_scale,
            bias=bias,
            dBias=dBias,
            use_alibi_mask=cfg.is_alibi,
            use_padding_mask=cfg.is_padding,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_total_seq_len_q=max_t_q,
            max_total_seq_len_kv=max_t_kv,
            diagonal_band_left_bound=cfg.left_bound,
            diagonal_band_right_bound=cfg.right_bound,
            diagonal_alignment=cfg.diag_align,
            dropout=dropout_tuple if cfg.is_dropout else None,
            use_deterministic_algorithm=cfg.is_determin,
        )

        dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
        dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
        dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
        if cfg.is_ragged:
            dQ.set_ragged_offset(q_ragged_offset)
            dK.set_ragged_offset(k_ragged_offset)
            dV.set_ragged_offset(v_ragged_offset)

        try:
            graph.validate()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph. {e}")
            pytest.skip("not supported backward graph")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during backward graph validate. {e}")
            pytest.fail("unexpected exception during backward graph validate", pytrace=False)

        try:
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph after validate. {e}")
            pytest.skip("not supported backward graph after validate")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after backward validate. {e}")
            pytest.fail("unexpected exception after backward validate", pytrace=False)

        variant_pack = {
            q: q_gpu,
            k: k_gpu,
            v: v_gpu,
            o: o_gpu,
            dO: dO_gpu,
            stats: stats_gpu,
            dQ: dQ_gpu,
            dK: dK_gpu,
            dV: dV_gpu,
            bias: bias_gpu,
            dBias: dBias_gpu,
            seq_len_q: seq_len_q_gpu,
            seq_len_kv: seq_len_kv_gpu,
            q_ragged_offset: q_ragged_offset_gpu if cfg.is_ragged else None,
            k_ragged_offset: k_ragged_offset_gpu if cfg.is_ragged else None,
            v_ragged_offset: v_ragged_offset_gpu if cfg.is_ragged else None,
            o_ragged_offset: o_ragged_offset_gpu if cfg.is_ragged else None,
        }

        if cfg.is_dropout:
            variant_pack[seed] = seed_gpu
            variant_pack[offset] = offset_gpu

        # Allocate workspace for the backward call.
        (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

        if request.config.getoption("--perf"):
            backward_times_ms = time_execution(graph.execute, variant_pack, workspace, cudnn_handle)
            print(f"@@@@ Backward graph.execute avg_time_ms={backward_times_ms.mean().item():.3f}")
            profile_execution(graph.execute, variant_pack, workspace, cudnn_handle)

        # Execute backward cuDNN graph
        graph.execute(variant_pack, workspace, cudnn_handle)
        torch.cuda.synchronize()

        if ws_sep is not None and not torch.all(ws_sep==-1).item():
            print("@@@@ Overall result: FAILED, backward workspace overwritten outside its boundaries.")
            print(ws_sep)
            pytest.fail("backward workspace overwritten outside boundaries", pytrace=False)

        # create fresh output tensors and rerun the backward graph
        # For deterministic algorithm, the grads should bitwise match the original grads
        if cfg.is_determin:
            dQ_gpu_rerun = dQ_gpu.clone().detach()
            dK_gpu_rerun = dK_gpu.clone().detach()
            dV_gpu_rerun = dV_gpu.clone().detach()
            
            dQ_gpu = torch.fill_(dQ_gpu, float("nan"))
            dK_gpu = torch.fill_(dK_gpu, float("nan"))
            dV_gpu = torch.fill_(dV_gpu, float("nan"))
            graph.execute(variant_pack, workspace, cudnn_handle)
            torch.cuda.synchronize()
            if ws_sep is not None and not torch.all(ws_sep==-1).item():
                print("@@@@ Overall result: FAILED, backward workspace overwritten outside its boundaries.")
                print(ws_sep)
                pytest.fail("backward workspace overwritten outside boundaries", pytrace=False)
            
            determin_err_count = 0
            determin_err_count += exact_equal(dQ_gpu, dQ_gpu_rerun, tag="dQ_determin", disp_elems=diffs)
            determin_err_count += exact_equal(dK_gpu, dK_gpu_rerun, tag="dK_determin", disp_elems=diffs)
            determin_err_count += exact_equal(dV_gpu, dV_gpu_rerun, tag="dV_determin", disp_elems=diffs)
            
            if determin_err_count != 0:
                print("@@@@ Overall result: FAILED, determinism check failed - outputs differ between runs.")
                pytest.fail("determinism check failed", pytrace=False)
            print("@@@@ Determinism check: PASSED, dQ, dK, dV bitwise match between runs.")

    bias_ref = None
    rng_dump_ref = None

    if not cfg.is_infer:
        # Using torch autograd reference in the backward pass.
        q_ref  = q_gpu.detach().float().requires_grad_()
        k_ref  = k_gpu.detach().float().requires_grad_()
        v_ref  = v_gpu.detach().float().requires_grad_()
        dO_ref = dO_gpu.detach().float()
        if cfg.is_ragged:
            dO_ref = convert_ragged_to_uniform(dO_ref, seq_len_q_gpu.detach())
        if cfg.is_bias:
            bias_ref = bias_gpu.detach().float().requires_grad_()
    else:
        # No autograd in the forward pass.
        q_ref  = q_gpu.detach().float()
        k_ref  = k_gpu.detach().float()
        v_ref  = v_gpu.detach().float()
        dO_ref = None
        if cfg.is_bias:
            bias_ref = bias_gpu.detach().float()

    if cfg.is_ragged:
        q_ref  = convert_ragged_to_uniform(q_ref, seq_len_q_gpu.detach())
        k_ref  = convert_ragged_to_uniform(k_ref, seq_len_kv_gpu.detach())
        v_ref  = convert_ragged_to_uniform(v_ref, seq_len_kv_gpu.detach())

    if cfg.is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    if cfg.is_dropout:
        rng_dump_ref = rng_dump_gpu.detach().float()

    # Compute forward reference output.
    ret = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref,
        block_mask=block_mask_gpu if cfg.is_block_mask else None,
        is_alibi=cfg.is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if cfg.is_padding else None,
        left_bound=cfg.left_bound,
        right_bound=cfg.right_bound,
        diag_align=cfg.diag_align,
        dropout_prob=cfg.dropout_prob,
        dropout_mask=rng_dump_ref,
        generate_stats=(cfg.is_infer == False),
    )

    if not cfg.is_infer:
        o_ref, stats_ref = ret
    else:
        o_ref = ret

    if cfg.is_ragged:
        o_gpu = convert_ragged_to_uniform(o_gpu, seq_len_q_gpu.detach())

    err_count = 0

    if cfg.is_padding:
        # zero out padded region of the output for comparison
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if cfg.is_infer == False:
                if cudnn_version < "9.14.0":
                    stats_ref[i, :, m:, :] = 0
                    stats_gpu[i, :, m:, :] = 0
                else:
                    stats_ref[i, :, m:, :] = -float("inf")

    err_count += approx_equal(o_gpu, o_ref, o_sep, o_raw, atol=2e-2, rtol=2e-2, tag="o", disp_elems=diffs)

    if not cfg.is_infer:
        err_count += approx_equal(stats_gpu, stats_ref, stats_sep, stats_raw, atol=2e-2, rtol=2e-2, tag="stats", disp_elems=diffs)

        inputs_ref = [q_ref, k_ref, v_ref]
        if cfg.is_bias:
            inputs_ref.append(bias_ref)

        [dQ_ref, dK_ref, dV_ref, *opt_refs] = list(
            torch.autograd.grad(outputs=o_ref, inputs=inputs_ref, grad_outputs=dO_ref)
        )

        if cfg.is_bias:
            dBias_ref = opt_refs.pop(0)

        if cfg.is_ragged:
            dQ_gpu = convert_ragged_to_uniform(dQ_gpu, seq_len_q_gpu.detach())
            dK_gpu = convert_ragged_to_uniform(dK_gpu, seq_len_kv_gpu.detach())
            dV_gpu = convert_ragged_to_uniform(dV_gpu, seq_len_kv_gpu.detach())

        if cfg.is_padding:
            # zero out padded region of the output for comparison
            for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
                dQ_ref[i, :, m:, :] = 0
                dQ_gpu[i, :, m:, :] = 0
                dK_ref[i, :, n:, :] = 0
                dK_gpu[i, :, n:, :] = 0
                dV_ref[i, :, n:, :] = 0
                dV_gpu[i, :, n:, :] = 0

        torch.cuda.synchronize()

        err_count += approx_equal(dQ_gpu, dQ_ref, dQ_sep, dQ_raw, atol=2e-2, rtol=2e-2, tag="dQ", disp_elems=diffs)
        err_count += approx_equal(dK_gpu, dK_ref, dK_sep, dK_raw, atol=2e-2 if cfg.data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dK", disp_elems=diffs)
        err_count += approx_equal(dV_gpu, dV_ref, dV_sep, dV_raw, atol=2e-2 if cfg.data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dV", disp_elems=diffs)
        if cfg.is_bias:
            err_count += approx_equal(dBias_gpu, dBias_ref, dBias_sep, dBias_raw, atol=2e-2, rtol=2e-2, tag="dBias", disp_elems=diffs)

    if err_count != 0:
        print("@@@@ Overall result: FAILED, disallowed mismatches")
        pytest.fail("disallowed mismatches", pytrace=False)
    else:
        print("@@@@ Overall result: PASSED, everything looks good!")
    
    del workspace
    del graph
    del variant_pack

    if cfg.is_paged:
        del container_k_gpu, container_v_gpu, page_table_k_gpu, page_table_v_gpu
    if cfg.is_ragged:
        del q_ragged_offset_gpu, k_ragged_offset_gpu, v_ragged_offset_gpu, o_ragged_offset_gpu
    if cfg.is_dropout:
        del seed_gpu, offset_gpu
        del rng_dump_gpu
        del rng_dump_ref
    if cfg.is_padding:
        del seq_len_q_gpu, seq_len_kv_gpu
        del seq_len_q_ref, seq_len_kv_ref

    del q_gpu, k_gpu, v_gpu, o_gpu
    if cfg.is_bias:
        del bias_gpu
    if not cfg.is_infer:
        del dQ_gpu, dK_gpu, dV_gpu, dO_gpu, stats_gpu
        if cfg.is_bias:
            del dBias_gpu

        del q_ref, k_ref, v_ref, dO_ref, o_ref, stats_ref
        if cfg.is_bias:
            del dBias_ref, bias_ref
        del dQ_ref, dK_ref, dV_ref
    else:
        del q_ref, k_ref, v_ref, o_ref
        if cfg.is_bias:
            del bias_ref

    del o_sep, o_raw
    if not cfg.is_infer:
        del dQ_sep, dQ_raw, dK_sep, dK_raw, dV_sep, dV_raw
        del stats_sep, stats_raw

    torch.cuda.empty_cache()
