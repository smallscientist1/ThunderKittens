#include "../../../src/kittens.cuh"

 // this kernel is more of an example kernel to show some TK programming models, rather than a kernel we think you should put into production, though it is pretty fast!

#define NUM_WORKERS 8 // This kernel uses 8 workers in parallel per block, to help issue instructions more quickly.

using namespace kittens; // this kernel only handles headdim=64 for simplicity. Also n should be a multiple of 256 here.

// warp 间一定是 warp*1 排布
template<ducks::rt::row_layout RT>
__device__ static inline void warp_make_causal(RT &dst, const RT &src, int subtile_j, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            if(j+subtile_j < ((warpid() ) * dst.height) + i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j+subtile_j > ((warpid() ) * dst.height) + i) { // above the diagonal, zero
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

__global__ void attend_ker64(int n, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    auto warpid        = kittens::warpid();
    // auto block_start   = blockIdx.x*(n*64);
    auto block_start_q = blockIdx.y * (n * 64) + blockIdx.x * (NUM_WORKERS * kittens::TILE_DIM * 64);
    auto block_start_kv = blockIdx.y * (n * 64);
    // const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
    //       bf16 *_o = __o__ + block_start;
    const bf16 *_q = __q__ + block_start_q, *_k = __k__ + block_start_kv, *_v = __v__ + block_start_kv;
          bf16 *_o = __o__ + block_start_q;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    // K and V live in shared memory -- this is about all that will fit.
    st_bf_1x4<ducks::st_layout::swizzle> (&k_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::swizzle> (&v_smem)[NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::swizzle>, NUM_WORKERS>();

    // Initialize all of the register tiles.
    rt_bf_1x4<> q_reg, k_reg, v_reg; // v_reg need to be swapped into col_l
    rt_fl_1x1<> att_block;
    rt_bf_1x1<> att_block_mma;
    rt_fl_1x4<> o_reg;
    rt_fl_1x1<>::col_vec max_vec_last, max_vec; // these are column vectors for the attention block
    rt_fl_1x1<>::col_vec norm_vec_last, norm_vec; // these are column vectors for the attention block
    
    // int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);
    int kv_blocks = n / (q_reg.rows*NUM_WORKERS);
    int kv_blocks_causal = blockIdx.x;

    // for(auto q_blk = 0; q_blk < qo_blocks; q_blk++) {

        // each warp loads its own Q tile of 16x64, and then multiplies by 1/sqrt(d)
        // load(q_reg, _q + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        load(q_reg, _q + (warpid)*q_reg.num_elements, q_reg.cols);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        // zero flash attention L, M, and O registers.
        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        // iterate over k, v for these q's that have been loaded
        for(auto kv_idx = 0; kv_idx <= kv_blocks_causal; kv_idx++) {

            // each warp loads its own chunk of k, v into shared memory
            load(v_smem[warpid], _v + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            load(k_smem[warpid], _k + (kv_idx*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
            __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

            // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
            for(int subtile = 0; subtile < NUM_WORKERS; subtile++) {

                load(k_reg, k_smem[subtile]); // load k from shared into registers

                zero(att_block); // zero 16x16 attention tile
                mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T

                copy(norm_vec_last, norm_vec);
                copy(max_vec_last,  max_vec);

                if (kv_idx == kv_blocks_causal) {
                    // 或者对warp进行判断然后continue?
                warp_make_causal(att_block, att_block, subtile, -INFINITY); 
                }

                row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
                sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
                exp(att_block, att_block); // exponentiate the block in-place.

                sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
                exp(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
                mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.

                row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
                div_row(att_block, att_block, norm_vec); // now the attention block is correctly normalized

                mul(norm_vec_last, norm_vec_last, max_vec_last); // normalize the previous norm vec according to the new max
                div(norm_vec_last, norm_vec_last, norm_vec); // normalize the previous norm vec according to the new norm

                copy(att_block_mma, att_block); // convert to bf16 for mma_AB

                load(v_reg, v_smem[subtile]); // load v from shared into registers.
                rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

                mul_row(o_reg, o_reg, norm_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
                mma_AB(o_reg, att_block_mma, v_reg_col, o_reg); // mfma onto o_reg with the local attention@V matmul.
            }
            __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk
        }

        // store(_o + (q_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
        store(_o + (warpid)*q_reg.num_elements, o_reg, q_reg.cols); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    // }
}

#include "harness.impl"