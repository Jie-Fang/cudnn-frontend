I would like to implement THD support for fp8 SDPA in cuDNN. 

Please read:
include/cudnn_frontend/node/scaled_dot_product_flash_attention.h
include/cudnn_frontend/node/sdpa_fp8_bwd.h

We already THD support for fp16, 
test/python/sdpa/fp16.py

We want it added here
test/python/sdpa/fp8.py

The backend library that this the actual fp8 kernel is generated is located in ../cudnn

Let me know if you need any more information!

Good luck!
