from nv_tensor_ir._mlir.dialects import nv_tensor_ir
import threading


class CompilerWithCaskContext:
    def __init__(self):
        self.cask_context = nv_tensor_ir.create_cask_context()
        self.compiler = nv_tensor_ir.Compiler(self.cask_context)

    def compile(self, module, compile_options):
        return self.compiler.compile(module, compile_options)


class CompilerWithKernelCache:
    def __init__(self):
        self.cached_shaders = {}
        self.cached_can_compile_results = {}
        # Need to make sure the lifetime of the cask context and compiler is longer
        # than the shaders to avoid segmentation fault when the cask context is released
        self.base_compiler = CompilerWithCaskContext()

        self.hit_cnt = 0
        self.miss_cnt = 0
        self.lock_kernel_cache = threading.Lock()

    def get_compute_capability(self):
        return self.base_compiler.cask_context.get_compute_capability()

    def compile(self, module, compile_options, use_cache=True):
        if use_cache is False:
            return self.base_compiler.compile(module, compile_options)
        else:
            return self.compile_with_cache(module, compile_options)

    def compile_with_cache(self, module, compile_options):
        cache_key = self.base_compiler.compiler.get_shader_cache_key(
            module, compile_options
        )
        # First check if the shader is in the cache
        with self.lock_kernel_cache:
            shader = self.cached_shaders.get(cache_key, None)
            if shader is not None:
                self.hit_cnt += 1
                return shader

        # If not, compile the shader
        shader = self.base_compiler.compile(module, compile_options)

        # Then add the shader to the cache
        with self.lock_kernel_cache:
            existing = self.cached_shaders.get(cache_key)
            if existing is None:
                self.cached_shaders[cache_key] = shader
                self.miss_cnt += 1
            else:
                # another thread beat us – discard dup compile, count as hit
                shader = existing
                self.hit_cnt += 1
        return shader

    def get_hit_rate(self):
        return self.hit_cnt / (self.hit_cnt + self.miss_cnt + 1e-9)

    def get_hit_cnt(self):
        return self.hit_cnt

    def get_miss_cnt(self):
        return self.miss_cnt

    def can_compile(self, module, compile_options):
        cache_key = self.base_compiler.compiler.get_shader_cache_key(
            module, compile_options
        )
        with self.lock_kernel_cache:
            can_compile = self.cached_can_compile_results.get(cache_key, None)
            if can_compile is not None:
                return can_compile
        can_compile = self.base_compiler.compiler.can_compile(module, compile_options)
        with self.lock_kernel_cache:
            self.cached_can_compile_results[cache_key] = can_compile
        return can_compile

    def print_stats(self):
        print(f"hit rate: {self.get_hit_rate()*100:.2f}%")
        print(f"hit cnt: {self.get_hit_cnt():,d}")
        print(f"miss cnt: {self.get_miss_cnt():,d}")
        print(f"total cnt: {self.get_hit_cnt() + self.get_miss_cnt():,d}")


class CompilerWithKernelCacheSingleton:
    _lock = threading.Lock()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CompilerWithKernelCache()
        return cls._instance
