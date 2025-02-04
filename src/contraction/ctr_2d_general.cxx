/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "ctr_2d_general.h"
#include "../tensor/untyped_tensor.h"
#include "../mapping/mapping.h"
#include "../shared/util.h"
#include "../shared/offload.h"
#include <climits>

namespace CTF_int {

  int  ctr_2d_gen_build(int                        is_used,
                        CommData                   global_comm,
                        int                        i,
                        int *                      virt_dim,
                        int64_t &                  cg_edge_len,
                        int &                      total_iter,
                        tensor *                   A,
                        int                        i_A,
                        CommData *&                cg_cdt_A,
                        int64_t &                  cg_ctr_lda_A,
                        int64_t &                  cg_ctr_sub_lda_A,
                        bool &                     cg_move_A,
                        int64_t *                  blk_len_A,
                        int64_t &                  blk_sz_A,
                        int64_t const *            virt_blk_len_A,
                        int &                      load_phase_A,
                        tensor *                   B,
                        int                        i_B,
                        CommData *&                cg_cdt_B,
                        int64_t &                  cg_ctr_lda_B,
                        int64_t &                  cg_ctr_sub_lda_B,
                        bool &                     cg_move_B,
                        int64_t *                  blk_len_B,
                        int64_t &                  blk_sz_B,
                        int64_t const *            virt_blk_len_B,
                        int &                      load_phase_B,
                        tensor *                   C,
                        int                        i_C,
                        CommData *&                cg_cdt_C,
                        int64_t &                  cg_ctr_lda_C,
                        int64_t &                  cg_ctr_sub_lda_C,
                        bool &                     cg_move_C,
                        int64_t *                  blk_len_C,
                        int64_t &                  blk_sz_C,
                        int64_t const *            virt_blk_len_C,
                        int &                      load_phase_C){
    mapping * map;
    int j;
    int64_t nstep = 1;
    if (comp_dim_map(&C->edge_map[i_C], &B->edge_map[i_B])){
      map = &B->edge_map[i_B];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
      }
      return 0;
    } else {
      if (B->edge_map[i_B].type == VIRTUAL_MAP &&
        C->edge_map[i_C].type == VIRTUAL_MAP){
        virt_dim[i] = B->edge_map[i_B].np;
        return 0;
      } else {
        cg_edge_len = 1;
        if (B->edge_map[i_B].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, B->edge_map[i_B].calc_phase());
          cg_cdt_B = &B->topo->dim_comm[B->edge_map[i_B].cdt];
          /*if (is_used && cg_cdt_B.alive == 0)
            cg_cdt_B.activate(global_comm.cm);*/
          nstep = B->edge_map[i_B].calc_phase();
          cg_move_B = 1;
        } else
          cg_move_B = 0;
        if (C->edge_map[i_C].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, C->edge_map[i_C].calc_phase());
          cg_cdt_C = &C->topo->dim_comm[C->edge_map[i_C].cdt];
          /*if (is_used && cg_cdt_C.alive == 0)
            cg_cdt_C.activate(global_comm.cm);*/
          nstep = MAX(nstep, C->edge_map[i_C].calc_phase());
          cg_move_C = 1;
        } else
          cg_move_C = 0;
        cg_ctr_lda_A = 1;
        cg_ctr_sub_lda_A = 0;
        cg_move_A = 0;
  
  
        /* Adjust the block lengths, since this algorithm will cut
           the block into smaller ones of the min block length */
        /* Determine the LDA of this dimension, based on virtualization */
        cg_ctr_lda_B  = 1;
        if (B->edge_map[i_B].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_B= blk_sz_B*B->edge_map[i_B].np/cg_edge_len;
        else
          cg_ctr_sub_lda_B= blk_sz_B/cg_edge_len;
        for (j=i_B+1; j<B->order; j++) {
          cg_ctr_sub_lda_B = (cg_ctr_sub_lda_B *
                virt_blk_len_B[j]) / blk_len_B[j];
          cg_ctr_lda_B = (cg_ctr_lda_B*blk_len_B[j])
                /virt_blk_len_B[j];
        }
        cg_ctr_lda_C  = 1;
        if (C->edge_map[i_C].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_C= blk_sz_C*C->edge_map[i_C].np/cg_edge_len;
        else
          cg_ctr_sub_lda_C= blk_sz_C/cg_edge_len;
        for (j=i_C+1; j<C->order; j++) {
          cg_ctr_sub_lda_C = (cg_ctr_sub_lda_C *
                virt_blk_len_C[j]) / blk_len_C[j];
          cg_ctr_lda_C = (cg_ctr_lda_C*blk_len_C[j])
                /virt_blk_len_C[j];
        }
        if (B->edge_map[i_B].type != PHYSICAL_MAP){
          if (blk_sz_B / nstep == 0) 
            printf("blk_len_B[%d] = %ld, nstep = %ld blk_sz_B = %ld\n",i_B,blk_len_B[i_B],nstep,blk_sz_B);
          blk_sz_B  = blk_sz_B / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] / nstep;
        } else {
          if (blk_sz_B  * B->edge_map[i_B].np/ nstep == 0) 
            printf("blk_len_B[%d] = %ld  B->edge_map[%d].np = %d, nstep = %ld blk_sz_B = %ld\n",i_B,blk_len_B[i_B],i_B,B->edge_map[i_B].np,nstep,blk_sz_B);
          blk_sz_B  = blk_sz_B * B->edge_map[i_B].np / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] * B->edge_map[i_B].np / nstep;
        }
        if (C->edge_map[i_C].type != PHYSICAL_MAP){
          if (blk_sz_C / nstep == 0) 
            printf("blk_len_C[%d] = %ld, nstep = %ld blk_sz_C = %ld\n",i_C,blk_len_C[i_C],nstep,blk_sz_C);
          blk_sz_C  = blk_sz_C / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] / nstep;
        } else {
          if (blk_sz_C  * C->edge_map[i_C].np/ nstep == 0) 
            printf("blk_len_C[%d] = %ld  C->edge_map[%d].np = %d, nstep = %ld blk_sz_C = %ld\n",i_C,blk_len_C[i_C],i_C,C->edge_map[i_C].np,nstep,blk_sz_C);
          blk_sz_C  = blk_sz_C * C->edge_map[i_C].np / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] * C->edge_map[i_C].np / nstep;
        }
  
        if (B->edge_map[i_B].has_child){
          ASSERT(B->edge_map[i_B].child->type == VIRTUAL_MAP);
          virt_dim[i] = B->edge_map[i_B].np*B->edge_map[i_B].child->np/nstep;
        }
        if (C->edge_map[i_C].has_child) {
          ASSERT(C->edge_map[i_C].child->type == VIRTUAL_MAP);
          virt_dim[i] = C->edge_map[i_C].np*C->edge_map[i_C].child->np/nstep;
        }
        if (C->edge_map[i_C].type == VIRTUAL_MAP){
          virt_dim[i] = C->edge_map[i_C].np/nstep;
        }
        if (B->edge_map[i_B].type == VIRTUAL_MAP)
          virt_dim[i] = B->edge_map[i_B].np/nstep;
  #ifdef OFFLOAD
        total_iter *= nstep;
        if (cg_ctr_sub_lda_A == 0)
          load_phase_A *= nstep;
        else 
          load_phase_A  = 1;
        if (cg_ctr_sub_lda_B == 0)   
          load_phase_B *= nstep;
        else 
          load_phase_B  = 1;
        if (cg_ctr_sub_lda_C == 0) 
          load_phase_C *= nstep;
        else 
          load_phase_C  = 1;
  #endif
      }
    } 
    return 1;
  }



  ctr_2d_general::~ctr_2d_general() {
    /*if (move_A) cdt_A->deactivate();
    if (move_B) cdt_B->deactivate();
    if (move_C) cdt_C->deactivate();*/
    if (rec_ctr != NULL)
      delete rec_ctr;
  }

  ctr_2d_general::ctr_2d_general(ctr * other) : ctr(other) {
    ctr_2d_general * o = (ctr_2d_general*)other;
    rec_ctr = o->rec_ctr->clone();
    edge_len      = o->edge_len;
    ctr_lda_A     = o->ctr_lda_A;
    ctr_sub_lda_A = o->ctr_sub_lda_A;
    cdt_A         = o->cdt_A;
    move_A        = o->move_A;
    ctr_lda_B     = o->ctr_lda_B;
    ctr_sub_lda_B = o->ctr_sub_lda_B;
    cdt_B         = o->cdt_B;
    move_B        = o->move_B;
    ctr_lda_C     = o->ctr_lda_C;
    ctr_sub_lda_C = o->ctr_sub_lda_C;
    cdt_C         = o->cdt_C;
    move_C        = o->move_C;
#ifdef OFFLOAD
    alloc_host_buf = o->alloc_host_buf;
#endif
  }

  void ctr_2d_general::print() {
    printf("ctr_2d_general: edge_len = %ld\n", edge_len);
    printf("move_A = %d, ctr_lda_A = %ld, ctr_sub_lda_A = %ld",
            move_A, ctr_lda_A, ctr_sub_lda_A);
    if (move_A) printf(", cdt_A length = %d",cdt_A->np);
    printf("\n");
    printf("move_B = %d, ctr_lda_B = %ld, ctr_sub_lda_B = %ld",
            move_B, ctr_lda_B, ctr_sub_lda_B);
    if (move_B) printf(", cdt_B length = %d",cdt_B->np);
    printf("\n");
    printf("move_C = %d, ctr_lda_C = %ld, ctr_sub_lda_C = %ld",
            move_C, ctr_lda_C, ctr_sub_lda_C);
    if (move_C) printf(", cdt_C length = %d",cdt_C->np);
    printf("\n");
#ifdef OFFLOAD
    if (alloc_host_buf)
      printf("alloc_host_buf is true\n");
    else
      printf("alloc_host_buf is false\n");
#endif
    rec_ctr->print();
  }

  ctr * ctr_2d_general::clone() {
    return new ctr_2d_general(this);
  }

  void ctr_2d_general::find_bsizes(int64_t & b_A,
                                   int64_t & b_B,
                                   int64_t & b_C,
                                   int64_t & s_A,
                                   int64_t & s_B,
                                   int64_t & s_C,
                                   int64_t & aux_size){
    b_A = 0, b_B = 0, b_C = 0;
    s_A = ctr_sub_lda_A*ctr_lda_A;
    s_B = ctr_sub_lda_B*ctr_lda_B;
    s_C = ctr_lda_C*ctr_sub_lda_C;
    if (move_A){
      b_A = edge_len/cdt_A->np;
    } 
    if (move_B){
      b_B = edge_len/cdt_B->np;
    }
    if (move_C){
      b_C = edge_len/cdt_C->np;
    }

    aux_size = MAX(move_A*sr_A->el_size*s_A, MAX(move_B*sr_B->el_size*s_B, move_C*sr_C->el_size*s_C));
  }

  double ctr_2d_general::est_time_fp(int nlyr) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    double est_comm_time = 0.0;
    if (move_A)
      est_comm_time += cdt_A->estimate_bcast_time(sr_A->el_size*s_A);
    if (move_B)
      est_comm_time += cdt_B->estimate_bcast_time(sr_B->el_size*s_B);
    if (move_C)
      est_comm_time += cdt_C->estimate_red_time(sr_C->el_size*s_C, sr_C->addmop());
    return (est_comm_time*(double)edge_len)/MIN(nlyr,edge_len);
  }

  double ctr_2d_general::est_time_rec(int nlyr) {
    return rec_ctr->est_time_rec(1)*(double)edge_len/MIN(nlyr,edge_len) + est_time_fp(nlyr);
  }


  double ctr_2d_general::est_internode_collective_comm_vol(int nlyr) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    double sz = 0.0;
    if (move_A)
      sz += (sr_A->el_size*s_A) * (cdt_A->num_nodes - 1);// ((cdt_A->np / cdt_A->intra_node_np) - 1);
    if (move_B)
      sz += (sr_B->el_size*s_B) * (cdt_B->num_nodes - 1);// ((cdt_B->np / cdt_B->intra_node_np) - 1);
    if (move_C)
      sz += (sr_C->el_size*s_C) * (cdt_C->num_nodes - 1); //((cdt_C->np / cdt_C->intra_node_np) - 1);
    return (sz*(double)edge_len)/MIN(nlyr,edge_len);
  }

  double ctr_2d_general::est_internode_comm_vol_rec(int nlyr) {
    return rec_ctr->est_internode_comm_vol_rec(1)*(double)edge_len/MIN(nlyr,edge_len) + est_internode_collective_comm_vol(nlyr);
  }



  int64_t ctr_2d_general::mem_fp() {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    return sr_A->el_size*s_A+sr_B->el_size*s_B+sr_C->el_size*s_C+aux_size;
  }

  int64_t ctr_2d_general::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void ctr_2d_general::run(char * A, char * B, char * C){
    int owner_A, owner_B, owner_C, ret;
    int64_t ib;
    char * buf_A, * buf_B, * buf_C; 
    char * op_A, * op_B, * op_C; 
    int rank_A, rank_B, rank_C;
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    if (move_A) rank_A = cdt_A->rank;
    else rank_A = -1;
    if (move_B) rank_B = cdt_B->rank;
    else rank_B = -1;
    if (move_C) rank_C = cdt_C->rank;
    else rank_C = -1;
    
    TAU_FSTART(ctr_2d_general);

    /* Must move at most two tensors */
    ASSERT(!(move_A && move_B && move_C));
    
    rec_ctr->beta         = this->beta;

    int iidx_lyr, inum_lyr;
    if (edge_len >= num_lyr && edge_len % num_lyr == 0){
      inum_lyr         = num_lyr;
      iidx_lyr         = idx_lyr;
      rec_ctr->num_lyr = 1;
      rec_ctr->idx_lyr = 0;
    } else if (edge_len < num_lyr && num_lyr % edge_len == 0){
      inum_lyr         = edge_len;
      iidx_lyr         = idx_lyr%edge_len;
      rec_ctr->num_lyr = num_lyr/edge_len;
      rec_ctr->idx_lyr = idx_lyr/edge_len;
    } else {
      rec_ctr->num_lyr = num_lyr;
      rec_ctr->idx_lyr = idx_lyr;
      inum_lyr         = 1;
      iidx_lyr         = 0;
    }

    
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    
#ifdef OFFLOAD
    if (alloc_host_buf){
      if (s_A > 0) host_pinned_alloc((void**)&buf_A, s_A*sr_A->el_size);
      if (s_B > 0) host_pinned_alloc((void**)&buf_B, s_B*sr_B->el_size);
      if (s_C > 0) host_pinned_alloc((void**)&buf_C, s_C*sr_C->el_size);
    }
#else
    if (0){
    }
#endif
    else {
      ret = CTF_int::alloc_ptr(s_A*sr_A->el_size, (void**)&buf_A);
      ASSERT(ret==0);
      ret = CTF_int::alloc_ptr(s_B*sr_B->el_size, (void**)&buf_B);
      ASSERT(ret==0);
      ret = CTF_int::alloc_ptr(s_C*sr_C->el_size, (void**)&buf_C);
      ASSERT(ret==0);
    }
    //ret = CTF_int::alloc_ptr(aux_size, (void**)&buf_aux);
    //ASSERT(ret==0);

    //for (ib=this->idx_lyr; ib<edge_len; ib+=this->num_lyr){
#ifdef MICROBENCH
    for (ib=iidx_lyr; ib<edge_len; ib+=edge_len)
#else
    for (ib=iidx_lyr; ib<edge_len; ib+=inum_lyr)
#endif
    {
      if (move_A){
        owner_A   = ib % cdt_A->np;
        if (rank_A == owner_A){
          if (b_A == 1){
            op_A = A;
          } else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A, 
                       A+sr_A->el_size*(ib/cdt_A->np)*ctr_sub_lda_A, ctr_sub_lda_A*b_A, 
                       op_A, ctr_sub_lda_A);
          }
        } else
          op_A = buf_A;
        cdt_A->bcast(op_A, s_A, sr_A->mdtype(), owner_A);
      } else {
        if (ctr_sub_lda_A == 0)
          op_A = A;
        else {
          if (ctr_lda_A == 1)
            op_A = A+sr_A->el_size*ib*ctr_sub_lda_A;
          else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A,
                       A+sr_A->el_size*ib*ctr_sub_lda_A, ctr_sub_lda_A*edge_len, 
                       buf_A, ctr_sub_lda_A);
          }      
        }
      }
      if (move_B){
        owner_B   = ib % cdt_B->np;
        if (rank_B == owner_B){
          if (b_B == 1){
            op_B = B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       B+sr_B->el_size*(ib/cdt_B->np)*ctr_sub_lda_B, ctr_sub_lda_B*b_B, 
                       op_B, ctr_sub_lda_B);
          }
        } else 
          op_B = buf_B;
//        printf("c_B = %ld, s_B = %ld, d_B = %ld, b_B = %ld\n", c_B, s_B,db, b_B);
        cdt_B->bcast(op_B, s_B, sr_B->mdtype(), owner_B);
      } else {
        if (ctr_sub_lda_B == 0)
          op_B = B;
        else {
          if (ctr_lda_B == 1){
            op_B = B+sr_B->el_size*ib*ctr_sub_lda_B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       B+sr_B->el_size*ib*ctr_sub_lda_B, ctr_sub_lda_B*edge_len, 
                       buf_B, ctr_sub_lda_B);
          }      
        }
      }
      if (move_C){
        op_C = buf_C;
        //sr_C->set(op_C, sr_C->addid(), s_C);
        //rec_ctr->beta = sr_C->mulid();
        rec_ctr->beta = sr_C->addid();
      } else {
        if (ctr_sub_lda_C == 0)
          op_C = C;
        else {
          if (ctr_lda_C == 1) 
            op_C = C+sr_C->el_size*ib*ctr_sub_lda_C;
          else {
            op_C = buf_C;
            //sr_C->set(op_C, sr_C->addid(), s_C);
            //rec_ctr->beta = sr_C->mulid();
            rec_ctr->beta = sr_C->addid();
          }
        }
      } 


      rec_ctr->run(op_A, op_B, op_C);

      /*for (int i=0; i<ctr_sub_lda_C*ctr_lda_C; i++){
        printf("[%d] P%d op_C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)op_C)[i]);
      }*/
      if (move_C){
        /* FIXME: Wont work for single precsion */
        owner_C   = ib % cdt_C->np;
        if (cdt_C->rank == owner_C)
          cdt_C->red(MPI_IN_PLACE, op_C, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        else
          cdt_C->red(op_C, op_C, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        if (rank_C == owner_C){
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     op_C, ctr_sub_lda_C, sr_C->mulid(),
                     C+sr_C->el_size*(ib/cdt_C->np)*ctr_sub_lda_C, 
                     ctr_sub_lda_C*b_C, this->beta);
        }
      } else {
        if (ctr_lda_C != 1 && ctr_sub_lda_C != 0)
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     buf_C, ctr_sub_lda_C, sr_C->mulid(), 
                     C+sr_C->el_size*ib*ctr_sub_lda_C, 
                     ctr_sub_lda_C*edge_len, this->beta);
        if (ctr_sub_lda_C == 0)
          rec_ctr->beta = sr_C->mulid();
      }
/*      for (int i=0; i<ctr_sub_lda_C*ctr_lda_C*edge_len; i++){
        printf("[%d] P%d C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)C)[i]);
      }*/
    }
    /* FIXME: reuse that */
#ifdef OFFLOAD
    if (alloc_host_buf){
      if (s_A > 0) host_pinned_free(buf_A);
      if (s_B > 0) host_pinned_free(buf_B);
      if (s_C > 0) host_pinned_free(buf_C);
    }
#else
    if (0){
    }
#endif
    else {
      CTF_int::cdealloc(buf_A);
      CTF_int::cdealloc(buf_B);
      CTF_int::cdealloc(buf_C);
    }
    TAU_FSTOP(ctr_2d_general);
  }

  void ctr_2d_general::blockComm( int const * rgrid, char *A, char *B, char *C
                                , size_t sizeA, size_t sizeB, size_t sizeC
                                , CommData glb_comm, std::vector<int> &swap
  ){
    int rank = glb_comm.rank;
    int np = glb_comm.np;
    int src, dst;
    // we have to determine the partners
    if (! swap.size() ) {
      ipair nr(getNumNodes(glb_comm.cm));
      // rGrid is the rankGrid of the given tensor topology
      CommGrid grid({rgrid[0], rgrid[1]}, nr.first);
      ipair nGrid = grid.nGrid;
      ipair iGrid = grid.iGrid;
      // rr is the key/color pair for the original rank distribution of dim_comm[0]
      std::vector<ipair> rr(np);
      std::vector< std::pair<ipair, int> > perm(np);
      for (int r(0); r < np; r++) rr[r] = { r % rgrid[0], r / rgrid[0] };
      // the desired distribution are nGrid[0] x nGrid[1] blocks with the some color
      for (int r(0); r < np; r++){
        // the color is the jth column and kth row in the nodeGrid
        int clr = (rr[r].second/iGrid.second)*nGrid.first + rr[r].first/iGrid.first;
        int key = (rr[r].second%iGrid.second)*iGrid.first + rr[r].first%iGrid.first;
       // we have to swap color and key that we can use std::sort
        perm[r] = { { clr, key }, r};
      }
      std::sort(perm.begin(), perm.end());
      for (auto p: perm) swap.push_back(p.second);

      src = swap[rank];
      auto it( std::find(swap.begin(), swap.end(), rank) );
      dst = std::distance(swap.begin(), it);
    }
    else {
      dst = swap[rank];
      auto it( std::find(swap.begin(), swap.end(), rank) );
      src = std::distance(swap.begin(), it);
    }

    MPI_Barrier(glb_comm.cm);
    MPI_Status s;
    MPI_Sendrecv_replace(&cdt_A->color, 1, MPI_INT, dst, 0, src, 0, glb_comm.cm, &s);
    MPI_Sendrecv_replace(&cdt_B->color, 1, MPI_INT, dst, 0, src, 0, glb_comm.cm, &s);
    MPI_Sendrecv_replace(&cdt_A->rank,  1, MPI_INT, dst, 0, src, 0, glb_comm.cm, &s);
    MPI_Sendrecv_replace(&cdt_B->rank,  1, MPI_INT, dst, 0, src, 0, glb_comm.cm, &s);

    MPI_Barrier(glb_comm.cm);

    size_t el(std::max(sizeA, sizeB));
    el = std::max(el, sizeC);
    char *buf = new char[el*sr_A->el_size];
    // Do the A job
    MPI_Request sreq, rreq;
    MPI_Irecv(buf, sizeA, sr_A->mdtype(), src, 0, glb_comm.cm, &rreq);
    MPI_Isend(A,   sizeA, sr_A->mdtype(), dst, 0, glb_comm.cm, &sreq);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
    memcpy(A, buf, sizeA*sr_A->el_size);

    // Do the B job
    MPI_Irecv(buf, sizeB, sr_A->mdtype(), src, 0, glb_comm.cm, &rreq);
    MPI_Isend(B,   sizeB, sr_A->mdtype(), dst, 0, glb_comm.cm, &sreq);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
    memcpy(B, buf, sizeB*sr_A->el_size);

    // Do the B job
    MPI_Irecv(buf, sizeC, sr_A->mdtype(), src, 0, glb_comm.cm, &rreq);
    MPI_Isend(C,   sizeC, sr_A->mdtype(), dst, 0, glb_comm.cm, &sreq);
    MPI_Wait(&rreq, MPI_STATUS_IGNORE);
    MPI_Wait(&sreq, MPI_STATUS_IGNORE);
    memcpy(C, buf, sizeC*sr_A->el_size);
    MPI_Barrier(glb_comm.cm);
  }

  ipair ctr_2d_general::getNumNodes(MPI_Comm comm){
    int rank, np;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);

    std::vector<std::string> nodeList(np);
    char nodeName[MPI_MAX_PROCESSOR_NAME];
    char nodeNames[np*MPI_MAX_PROCESSOR_NAME];
    std::vector<int> nameLengths(np);
    std::vector<int> off(np);
    int nameLength;
    MPI_Get_processor_name(nodeName, &nameLength);
    MPI_Allgather(
      &nameLength, 1, MPI_INT, nameLengths.data(), 1, MPI_INT, comm
    );
    for (int i(1); i < np; i++) off[i] = off[i-1] + nameLengths[i-1];
    MPI_Allgatherv(
      nodeName, nameLengths[rank], MPI_BYTE, nodeNames,
      nameLengths.data(), off.data(), MPI_BYTE, comm
    );
    for (int i(0); i < np; i++) {
      std::string s(&nodeNames[off[i]], nameLengths[i]);
      nodeList[i] = s;
    }
    std::sort(nodeList.begin(), nodeList.end());
    std::vector<std::string>::iterator it(
      std::unique(nodeList.begin(), nodeList.end())
    );
    size_t nNodes(std::distance(nodeList.begin(), it));
    return {nNodes, np/nNodes};
  }

}

