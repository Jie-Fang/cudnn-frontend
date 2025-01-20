def generate_cutlass_args(flags) :
    # assuming format nchw from cudnn flags
    op = flags['R'][0]
    type_input = 'f16'
    type_output = 'f16'
    type_compute = 'f32'
    if op == 'gemm':
        if flags['Pin'][0] == 's':
          type_input = 'f32'
        if flags['Pin'][0] == 'h':
          type_input = 'f16'
        if flags['Pout'][0] == 's':
          type_output = 'f32'
        if flags['Pout'][0] == 'h':
          type_output = 'f16'
        if flags['P'][0] == 'h':
            type_compute = 'f16'
        if flags['P'][0] == 's':
            type_compute = 'f32'
        major_a = "column" if 'ta' in flags.keys() else "row"
        major_b = "column" if 'tb' in flags.keys() else "row"
        cutlass_profiler_args = ["--function=Gemm", "--problem-size::m=" + flags['m'][0], "--problem-size::n=" + flags['n'][0], "--problem-size::k=" + flags['k'][0], '--A=' + type_input + ":" + major_a, '--B=' + type_input + ":" + major_b, '--C=' + type_output, '--accumulator-type=' + type_compute]
        return cutlass_profiler_args
    elif op == 'conv' or op == 'dgrad' or op == 'wgrad':
        dims_act = flags['dimA'][0].split(',')
        dims_filt = flags['filtA'][0].split(',')
        if 'padA' in flags:
            dims_pad = flags['padA'][0].split(',')
            dims_pad_h = dims_pad[0]
            dims_pad_w = dims_pad[1]
        if 'pad_w' in flags:
            dims_pad_w = flags['pad_w'][0]
        if 'pad_h' in flags:
            dims_pad_h = flags['pad_h'][0]
        if 'u' in flags:
            dims_stride_w = flags['u'][0]
        if 'v' in flags:
            dims_stride_h = flags['v'][0]
        alpha = flags['A'][0]
        beta = flags['B'][0]
        formatOut = 'packed_nhwc'
        formatIn = 'packed_nhwc'
        filtFormat = 'packed_nhwc'
        if 'formatOut' in flags.keys():
            if flags['formatOut'] != 1:
                formatOut = 'packed_nchw'
        if 'formatIn' in flags.keys():
            if flags['formatIn'] != 1:
                formatIn = 'packed_nchw'
        if 'filtFormat' in flags.keys():
            if flags['filtFormat'] != 1:
                filtFormat = 'packed_nchw'
        conv_func = 'Conv2dFprop'
        if op == 'dgrad':
            conv_func = 'Conv2dDgrad'
        if op == 'wgrad':
            conv_func = 'Conv2dWgrad'
        cutlass_profiler_args = ["--function=" + conv_func, "--Output=" + type_output, "--accumulator-type=" + type_compute, "--Activation=" + type_input + ":" + formatIn, "--Filter=" + type_input + ":" + filtFormat, "--input-size::n=" + dims_act[0], "--input-size::c=" + dims_act[1], "--input-size::h=" + dims_act[2], "--input-size::w=" + dims_act[3], "--filter-size::k=" + dims_filt[0], "--filter-size::c=" + dims_filt[1], "--filter-size::r=" + dims_filt[2], "--filter-size::s=" + dims_filt[3], "--pad::top=" + dims_pad_h, "--pad::bottom=" + dims_pad_h, "--pad::left=" + dims_pad_w, "--pad::right=" + dims_pad_w, "--stride::h=" + dims_stride_h, "--stride::w=" + dims_stride_w, "--dilation::h=" + "1", "--dilation::w=" + "1", "--op-tile::m=16", "--op-tile::n=8", "--op-tile::k=8"]
        return cutlass_profiler_args
    else:
        return []

def cutlass_output_parse(layer, parsed_args, test_results):
    a = test_results.output.find("CSV")
    layer_file = parsed_args.layers_path.split('/')[-1]
    config_run = True if a != -1 else False
    shell_flags = generate_cutlass_args(layer.flags.flags_dict)
    shell = parsed_args.bin_path + "/" + parsed_args.bin_name + " " + " ".join(shell_flags)
    if test_results.flags['R'][0] == 'gemm' or test_results.flags['R'][0] == 'conv' or test_results.flags['R'][0] == 'dgrad' or test_results.flags['R'][0] == 'wgrad':
        if config_run == False:
            f = open("db_" + test_results.flags['R'][0] + "_config_no_output_" + layer_file + ".txt", "a+")
            f.write(layer.base_name + " | " + str(test_results.flags) + " | " + shell + "\n")
            f.close()
    if config_run == True:
      if test_results.flags['R'][0] == 'gemm' and config_run:
        f_not_verified = open("db_gemm_not_verified_" + layer_file + ".txt", "a+")
        f_incorrect = open("db_gemm_incorrect_" + layer_file + ".txt", "a+")
        f_good = open("db_gemm_good_" + layer_file + ".txt", "a+")
        f_other = open("db_gemm_other_" + layer_file + ".txt", "a+")
        perfs = []
        _results = test_results.output[a:].split('\n')[2:-1]
        for __result in _results[1:]:
          if __result.find("CUTLASS") != -1:
            disposition = __result.split(',')[5]
            is_incorrect = True if disposition == "incorrect" else False
            is_not_verified = True if disposition == "not_verified" else False
            is_passed = True if disposition == "passed" else False
            if is_not_verified == True:
              f_not_verified.write(layer.base_name + " | ")
              f_not_verified.write(str(test_results.flags) + " | " + shell + " | ")
              f_not_verified.write(__result)
              f_not_verified.write("\n")
            elif is_incorrect == True:
              f_incorrect.write(layer.base_name + " | ")
              f_incorrect.write(str(test_results.flags) + " | " + shell + " | ")
              f_incorrect.write(__result)
              f_incorrect.write("\n")
            elif is_passed == True:
              f_good.write(layer.base_name + " | ")
              f_good.write(str(test_results.flags) + " | " + shell + " | ")
              f_good.write(__result)
              f_good.write("\n")
            else:
              f_other.write(layer.base_name + " | ")
              f_other.write(str(test_results.flags) + " | " + shell + " | ")
              f_other.write(__result)
              f_other.write("\n")
        f_not_verified.close()
        f_incorrect.close()
        f_good.close()
        f_other.close()
      elif test_results.flags['R'][0] == 'conv' or test_results.flags['R'][0] == 'dgrad' or test_results.flags['R'][0] == 'wgrad':
        f_not_verified = open("db_" + test_results.flags['R'][0] + "_not_verified_" + layer_file + ".txt", "a+")
        f_incorrect = open("db_" + test_results.flags['R'][0] + "_incorrect_" + layer_file + ".txt", "a+")
        f_good = open("db_" + test_results.flags['R'][0] + "_good_" + layer_file + ".txt", "a+")
        f_other = open("db_" + test_results.flags['R'][0] + "_other_" + layer_file + ".txt", "a+")
        _results = test_results.output[a:].split('\n')[2:-1]
        for __result in _results[1:]:
          if __result.find("CUTLASS") != -1:
            disposition = __result.split(',')[5]
            is_incorrect = True if disposition == "incorrect" else False
            is_not_verified = True if disposition == "not_verified" else False
            is_passed = True if disposition == "passed" else False
            if is_not_verified == True:
              f_not_verified.write(layer.base_name + " | ")
              f_not_verified.write(str(test_results.flags) + " | " + shell + " | ")
              f_not_verified.write(__result)
              f_not_verified.write("\n")
            elif is_incorrect == True:
              f_incorrect.write(layer.base_name + " | ")
              f_incorrect.write(str(test_results.flags) + " | " + shell + " | ")
              f_incorrect.write(__result)
              f_incorrect.write("\n")
            elif is_passed:
              f_good.write(layer.base_name + " | ")
              f_good.write(str(test_results.flags) + " | " + shell + " | ")
              f_good.write(__result)
              f_good.write("\n")
            else:
              f_other.write(layer.base_name + " | ")
              f_other.write(str(test_results.flags) + " | " + shell + " | ")
              f_other.write(__result)
              f_other.write("\n")
        f_not_verified.close()
        f_incorrect.close()
        f_good.close()
        f_other.close()
