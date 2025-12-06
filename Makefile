NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr

all: block_sparse_att streaming_att block_topk_sparse

block_sparse_att: block_sparse_att.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

streaming_att: streaming_att.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

block_topk_sparse: block_topk_sparse.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f block_sparse_att streaming_att block_topk_sparse

.PHONY: all clean
