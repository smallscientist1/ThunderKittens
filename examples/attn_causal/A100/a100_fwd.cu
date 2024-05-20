// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "../src/kittens.cuh"
#else
#include "../../../src/kittens.cuh"
#endif

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define NUM_WORKERS (8)
// #define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))
#define NUM_WORKERS_KV (4)

using namespace kittens;

template<ducks::rt::row_layout RT>
__device__ static inline void wg_make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            if(j < ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
        }
    }
}

// warp 间一定是 warp*1 排布
template<ducks::rt::row_layout RT>
__device__ static inline void warp_make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            if(j < ((warpid() ) * dst.height) + i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > ((warpid() ) * dst.height) + i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
        }
    }
}

using layout_q = kittens::ducks::st_layout::swizzle; 
using layout_k = kittens::ducks::st_layout::swizzle; 
using layout_v = kittens::ducks::st_layout::swizzle; 
using layout_o = kittens::ducks::st_layout::swizzle;

// shared tile 128, 128（must be square for causal）
constexpr int qo_height = 8; 
constexpr int kv_height = 8; // not used

// NUMWORKERS * 16 = qo_height(shared tile)
template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 2)
void fwd_attend_ker_dim(int N, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    auto block_start = blockIdx.x * (N * D);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    // st_bf<qo_height, D/kittens::TILE_DIM, layout_q>          (&q_smem)   [NUM_WARPGROUPS] = al.allocate<st_bf<qo_height, D/kittens::TILE_DIM, layout_q>,          NUM_WARPGROUPS>();
    st_bf<1, D/kittens::TILE_DIM, layout_k>          (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<1, D/kittens::TILE_DIM, layout_k>, 2,       NUM_WORKERS_KV>();
    st_bf<1, D/kittens::TILE_DIM, layout_v>          (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<1, D/kittens::TILE_DIM, layout_v>, 2,       NUM_WORKERS_KV>();

    int tic = 0, toc = 1;
 
    rt_bf<1, D/kittens::TILE_DIM> q_reg, k_reg, v_reg;
    // rt_fl<1, kv_height> att_block;
    // rt_bf<1, kv_height> att_block_mma;
    rt_fl<1, 1> att_block;
    rt_bf<1, 1> att_block_mma;
    rt_fl<1, D/kittens::TILE_DIM> o_prev;
    rt_fl<1, kv_height>::col_vec max_vec_last,  max_vec;
    rt_fl<1, kv_height>::col_vec norm_vec_last, norm_vec;

    int warpid      = kittens::warpid();
    // int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int qo_index    = (blockIdx.x * NUM_WORKERS) + warpid;

    int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    // __shared__ uint64_t qsmem_barrier, kvsmem_barrier;//, vsmem_barrier;

    // int q_phasebit = 0;
    // int kv_phasebit = 0;

    // if (threadIdx.x == 0) {
    //     tma::init_barrier<st_bf<qo_height, D/kittens::TILE_DIM, layout_q>, NUM_WARPGROUPS>(qsmem_barrier, 1);
    //     tma::init_barrier<st_bf<kv_height, D/kittens::TILE_DIM, layout_k>, NUM_WORKERS_KV*2>(kvsmem_barrier, 1); 
    // }

    // if (warpid == 0) {
    //     for (int wg = 0; wg < NUM_WORKERS/kittens::WARPGROUP_WARPS; wg++) { // load q
    //         int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + wg;
    //         tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
    //     }
    //     for (int w = 0; w < NUM_WORKERS_KV; w++) { // load k, v      
    //         int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
    //         tma::load_async((k_smem[tic][w]), tma_k, kvsmem_barrier, tile_idx); 
    //         tma::load_async((v_smem[tic][w]), tma_v, kvsmem_barrier, tile_idx); 
    //     }
    // }

    load(q_reg, _q + (warpid)*q_reg.num_elements, q_reg.cols);


    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    // __syncthreads();

    // tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    // q_phasebit ^= 1;

    // if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); }
    // else { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }
    if constexpr (D == 64) { mul(q_reg, q_reg, __float2bfloat16(0.125f)); }
    else { mul(q_reg, q_reg, __float2bfloat16(0.08838834764f)); }


    for (auto kv_idx = 0; kv_idx <= qo_index/NUM_WORKERS; kv_idx++, tic ^= 1, toc ^= 1) {
        // tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        // load from global, global no swizzle; thread mapping default
        load(k_smem[tic][warpid], _k + (kv_idx*NUM_WORKERS+warpid)*k_reg.num_elements, k_reg.cols);
        load(v_smem[tic][warpid], _v + (kv_idx*NUM_WORKERS+warpid)*v_reg.num_elements, v_reg.cols);
        // kv_phasebit ^= 1;

        __syncthreads();
        // if (warpid == 0) {
        //     if (kv_idx + 1 < kv_blocks) {
        //         tma::set_bytes(kvsmem_barrier, 2 * NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));
                
        //         for (int w = 0; w < NUM_WORKERS_KV; w++) {        
        //             int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
        //             tma::load_async((k_smem[toc][w]), tma_k, kvsmem_barrier, tile_idx); 
        //             tma::load_async((v_smem[toc][w]), tma_v, kvsmem_barrier, tile_idx);
        //         }
        //     }
        // }

        // warpgroup::mma_fence(att_block);
        // warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][0]);
        // warpgroup::mma_commit_group();
        for(int subtile = 0; subtile < NUM_WORKERS_KV; subtile++) { // 1
            load(k_reg, k_smem[tic][subtile]);
            zero(att_block);
            mma_ABt(att_block, q_reg, k_reg, att_block);

            copy(norm_vec_last, norm_vec);
            copy(max_vec_last,  max_vec);

            if (kv_idx == qo_index/NUM_WORKERS) {
                warp_make_causal(att_block, att_block, -INFINITY); 
            }

            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec);
            exp(att_block, att_block);

            sub(max_vec_last, max_vec_last, max_vec);
            exp(max_vec_last, max_vec_last);
            mul(norm_vec, norm_vec, max_vec_last);

            row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
            div_row(att_block, att_block, norm_vec);

            mul(norm_vec_last, norm_vec_last, max_vec_last);
            div(norm_vec_last, norm_vec_last, norm_vec);

            copy(att_block_mma, att_block); // convert to bf16 for mma

            load(v_reg, v_smem[tic][subtile]);
            // different smem layout for mma_AB vs mma_ABt to avoid load smem->reg smem bankconflicts
            rt_bf<1, D/kittens::TILE_DIM, ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg
            

            mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it
            mma_AB(o_prev, att_block_mma, v_reg_col, o_prev); // mfma onto o_prev with the local attention@V matmul.
        }
        // warpgroup::mma_fence(o_prev);
        // warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][0]);
        // warpgroup::mma_commit_group();
    }

    // auto (*o_smem) = reinterpret_cast<st_bf<qo_height, D/kittens::TILE_DIM, layout_o>(*)>(q_smem); // reuse q memory
    // warpgroup::store(o_smem[warpgroupid], o_prev); 
    // __syncthreads();
    
    // if (warpid % 4 == 0) { // store o
    //     int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
    //     tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
    //     tma::store_commit_group(); 
    // }

    // tma::store_async_wait();
    store(_o + (warpid)*o_prev.num_elements, o_prev, o_prev.cols);
}

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void attention_forward_causal(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);

    auto batch   = q.size(0);
    auto heads   = q.size(1);
    auto N       = q.size(2);
    auto D       = q.size(3);

    auto threads = NUM_WORKERS * kittens::WARP_THREADS;

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    // make sure sequence length is multiple of 128 for now
    TORCH_CHECK(N % (NUM_WORKERS * kittens::TILE_DIM) == 0, "Please pad sequence length to be multiple of 128");

    // make sure D = 64 or 128
    TORCH_CHECK(D == 64 | D == 128, "Currently, only D = 64 or 128 is supported");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

    if (D == 64) {
        CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, 4, layout_q>>(q_bf, (batch*heads*N)/(qo_height * 16));
        CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, 4, layout_k>>(k_bf, (batch*heads*N)/(kv_height * 16));
        CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, 4, layout_v>>(v_bf, (batch*heads*N)/(kv_height * 16));
        CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, 4, layout_o>>(o_bf, (batch*heads*N)/(qo_height * 16));

        unsigned long mem_size = 112000;
        cudaFuncSetAttribute(fwd_attend_ker_dim<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

        fwd_attend_ker_dim<64><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
    }
    else {
        CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, 8, layout_q>>(q_bf, (batch*heads*N)/(qo_height * 16));
        CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, 8, layout_k>>(k_bf, (batch*heads*N)/(kv_height * 16));
        CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<kv_height, 8, layout_v>>(v_bf, (batch*heads*N)/(kv_height * 16));
        CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<qo_height, 8, layout_o>>(o_bf, (batch*heads*N)/(qo_height * 16));

        unsigned long mem_size = 112000;
        cudaFuncSetAttribute(fwd_attend_ker_dim<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

        fwd_attend_ker_dim<128><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
    }
    
    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness_a100_fwd.impl"
#endif

