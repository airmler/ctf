/*Copyright (c) 2022, Andreas Irmler, all rights reserved.*/

#include "../tensor/untyped_tensor.h"
#include "bstensor.h"
#include <cstring>
#ifdef _OPENMP
#include "omp.h"
#endif


#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

namespace CTF {


  template<typename dtype>
  bsTensor<dtype>::bsTensor(int                       order_,
                            int64_t const *           len_,
                            int const *               sym_,
                            std::vector<ivec>         nonZero_,
                            World *                   world_,
                            char const *              name_,
                            bool                      profile_,
                            CTF_int::algstrct const & sr_) {
//    this->sr = sr_;
    this->order = order_;
    this->lens = (int64_t*)CTF_int::alloc(order*sizeof(int64_t));
    memcpy(this->lens, len_, order*sizeof(int64_t));
//    this->world = *world_;
    if (name_ != NULL) {
      this->name = (char*)CTF_int::alloc(strlen(name_)+1);
      strcpy(this->name, name_);
    } else {
      this->name = (char*)CTF_int::alloc(7*sizeof(char));
      for (int i=0; i<4; i++){
        this->name[i] = 'A'+(world_->glob_wrld_rng()%26);
      }
      this->name[4] = '0'+(order_/10);
      this->name[5] = '0'+(order_%10);
      this->name[6] = '\0';
    }
    nBlocks = nonZero_.size();
    if (nBlocks <= 0) { printf("Tensor %s", this->name); IASSERT(nBlocks>0);};
    int i(0);
    for (auto nz: nonZero_){
      IASSERT(nz.size()==order_);
      ivec r(nz);
      r.push_back(i++);
      nonZeroCondition.push_back(r);
      tensors.push_back(
        new CTF_int::tensor(&sr_, order_, len_, sym_, world_, 1, this->name, profile_)
      );
    }
    IASSERT(sizeof(dtype)==sr_.el_size);
  }


  template<typename dtype>
  bsTensor<dtype>::bsTensor(bsTensor<dtype> const & A)
  {
    order = A->order;
    lens = A->lens;
    name = A->name;
    nonZeroCondition = A->nonZeroCondition;
    nBlocks = A->nBlocks;
    tensors = A->tensors;
  }


  template<typename dtype>
  bsTensor<dtype>::~bsTensor(){
    for (auto &t: tensors) {
//      delete t;
    }
    CTF_int::cdealloc(lens);
    CTF_int::cdealloc(name);
  }



  template<typename dtype>
  void bsTensor<dtype>::read(int64_t         npair,
                             int64_t const * global_idx,
                             dtype *         data,
                             int64_t const   block,
                             CTF_int::algstrct const &sr)
  {
    IASSERT(block>=0); //not implemented yet
    IASSERT(block< this->nBlocks);
    this->tensors[block]->read(
      npair, sr.mulid(), sr.addid(),
      (const int64_t *)global_idx, (char *) data
    );
  }


  template<typename dtype>
  void bsTensor<dtype>::read_all(dtype *         data, int64_t const block)
  {
//    IASSERT(block>=0); //not implemented yet
    IASSERT(block< this->nBlocks);
    int64_t npair;
    printf("We have %ld blocks\n", this->nBlocks);
    for (auto nz: this->nonZeroCondition) {
      for (auto a: nz) printf("%d ", a);
      printf("\n");
    }
    this->tensors[0]->allread(&npair, (char *) data, false);
  }



  template<typename dtype>
  void bsTensor<dtype>::write(int64_t         npair,
                              int64_t const * global_idx,
                              dtype const   * data,
                              int64_t const   block,
                              CTF_int::algstrct const & sr)
  {

    IASSERT(block< this->nBlocks);
    if (block < 0){
      // in this case it is assumed that all data (npair elements)
      // WATCH OUT: it is assumed that the global_idx is the same
      //            for all different blocks!!!
      // it is assumed that global_idx
      size_t i(0);
      for (auto &t: this->tensors)
        t->write( npair, sr.mulid(), sr.addid()
                , (const int64_t *) global_idx, (const char *) &data[(i++)*npair]
                );
    } else{
      this->tensors[block]->write(
        npair, sr.mulid(), sr.addid(),
        (const int64_t *) global_idx,(const char*) data
      );
    }
  }


  template<typename dtype>
  char const * bsTensor<dtype>::get_name() const {
    return name;
  }

  template<typename dtype>
  void bsTensor<dtype>::checkDublicate(std::string t){
    std::sort(t.begin(), t.end());
    auto it = std::unique(t.begin(), t.end());
    if (it != t.end()) assert(0);
  }

  template<typename dtype>
  std::function<int(const ivec &, const ivec &)>
  bsTensor<dtype>::compare(const ivec p)
  {
    return [p] (const ivec &a, const ivec &b) -> int {
      size_t n(a.size());
      ivec c(n), d(n);
      for (size_t i(0); i < n - 1; i++){
        c[i] = a[p[i]]; d[i] = b[p[i]];
      }
      return c < d;
    };
  }


  template<typename dtype>
  std::function<int(const ivec &)>
  bsTensor<dtype>::find(const ivec c, const ivec p, const size_t n)
  {
    return [c, p, n] (const ivec &a) -> int {
      for (size_t i(0); i < n; i++)
        if ( a[p[i]] != c[p[i]]) return false;
      return true;
    };
  }


  template<typename dtype>
  size_t bsTensor<dtype>::orderToN(ivec o){
    size_t p(0);
    size_t f(1);
    for (size_t i(0); i < o.size(); i++){
      p += o[i] * f;
      f *= o.size();
    }
    return p;
  }



  template<typename dtype>
  void bsTensor<dtype>::sum(dtype                  alpha,
                            bsTensor<dtype>      & A,
                            char const *           cidx_A,
                            dtype                  beta,
                            char const *           cidx_B,
                            bool                   verbose) {
// allow the left side to be larger than the right side
// A["ijab"] = epsi["i"] means: write for every jab the value of epsi for the given "i"
// the same is true for the nonZero blocks. Every
    if (this->order != A.order) {
      printf("Problems with tensor order of tensors %s %s", this->name, A.name);
      IASSERT(this->order == A.order);
    }
    int nBlocks = this->nBlocks;
    ivec idA(nBlocks);
    IASSERT(this->nBlocks == A.nBlocks);
    checkDublicate(cidx_A);
    checkDublicate(cidx_B);
    std::string idxA(cidx_A);
    std::string idxB(cidx_B);
    std::vector<int> idx(order+1);
    for (int i(0); i < order; i++){
      auto p = std::find(std::begin(idxA), std::end(idxA), idxB[i]);
      idx[i] = std::distance(std::begin(idxA), p);
    }
    idx[order] = order;

    auto nzA = A.nonZeroCondition;
    auto p = orderToN(idx);
    std::sort(nzA.begin(), nzA.end(), compare(idx));

    for (int i(0); i < this->nBlocks; i++){
      idA[i] = nzA[i][order];
      if (verbose){
        printf("bs %d: ", i);
        for (int j(0); j < order; j++) printf("%d ", this->nonZeroCondition[i][j]);
        printf("| %d -> ", this->nonZeroCondition[i][order]);
        for (int j(0); j < order; j++) printf("%d ", nzA[i][j]);
        printf(" | %d\n", idA[i]);
      }
    }

    for (int i(0); i < this->nBlocks; i++){
      CTF_int::summation sum
        = CTF_int::summation(
            A.tensors[idA[i]], cidx_A, (char*)&alpha,
            this->tensors[i], cidx_B, (char*)&beta
          );
      sum.execute();
    }
  }

  template<typename dtype>
  void bsTensor<dtype>::sum(dtype                  alpha,
                            bsTensor<dtype>      & A,
                            char const *           cidx_A,
                            dtype                  beta,
                            char const *           cidx_B,
                            Univar_Function<dtype> fseq,
                            bool                   verbose){
    IASSERT(this->order == A.order);
    int nBlocks = this->nBlocks;
    ivec idA(nBlocks);
    IASSERT(this->nBlocks == A.nBlocks);
    checkDublicate(cidx_A);
    checkDublicate(cidx_B);
    std::string idxA(cidx_A);
    std::string idxB(cidx_B);
    std::vector<int> idx(order+1);
    for (int i(0); i < order; i++){
      auto p = std::find(std::begin(idxA), std::end(idxA), idxB[i]);
      idx[i] = std::distance(std::begin(idxA), p);
    }
    idx[order] = order;

    auto nzA = A.nonZeroCondition;
    auto p = orderToN(idx);
    std::sort(nzA.begin(), nzA.end(), compare(idx));

    for (int i(0); i < this->nBlocks; i++){
      idA[i] = nzA[i][order];
      if (verbose){
        printf("bs %d: ", i);
        for (int j(0); j < order; j++) printf("%d ", this->nonZeroCondition[i][j]);
        printf("| %d -> ", this->nonZeroCondition[i][order]);
        for (int j(0); j < order; j++) printf("%d ", nzA[i][j]);
        printf(" | %d\n", idA[i]);
      }
    }

    for (int i(0); i < this->nBlocks; i++){
      CTF_int::summation sum
        = CTF_int::summation(
            A.tensors[idA[i]], cidx_A, (char*)&alpha,
            this->tensors[i], cidx_B, (char*)&beta,&fseq
          );
      sum.execute();
    }
  }




  template<typename dtype>
  void bsTensor<dtype>::sum(dtype             alpha,
                            bsTensor<dtype> & A,
                            char const *      cidx_A,
                            dtype             beta,
                            char const *      cidx_B,
                            std::vector<ivec> nonZeroA,
                            std::vector<ivec> nonZeroB,
                            bool              verbose) {
    IASSERT(this->order == A.order);
    int nBlocks = nonZeroA.size();
    IASSERT(nonZeroA.size() == nonZeroB.size());
    int *nzIdxA = new int[nBlocks];
    int *nzIdxB = new int[nBlocks];
    checkDublicate(cidx_A);
    checkDublicate(cidx_B);

    for (int64_t i(0); i < nonZeroA.size(); i++){
      auto p =
        std::find( A.nonZeroCondition.begin()
                 , A.nonZeroCondition.end()
                 , nonZeroA[i]
                 );
      nzIdxA[i] = std::distance(A.nonZeroCondition.begin(), p);
      IASSERT(nzIdxA[i] < A.nonZeroCondition.size());

      p = std::find( this->nonZeroCondition.begin()
                   , this->nonZeroCondition.end()
                   , nonZeroB[i]
                   );
      nzIdxB[i] = std::distance(this->nonZeroCondition.begin(), p);
      IASSERT(nzIdxB[i] < this->nonZeroCondition.size());
      if (verbose){
        printf("bs %ld: ", i);
        for (int j(0); j < order; j++) printf("%d ", nonZeroA[i][j]);
        printf("| %d -> ", nzIdxA[i]);
        for (int j(0); j < order; j++) printf("%d ", nonZeroB[i][j]);
        printf(" | %d\n", nzIdxB[i]);
      }

   }

    for (int64_t i(0); i < nonZeroA.size(); i++){
      CTF_int::summation sum
        = CTF_int::summation(
            A.tensors[nzIdxA[i]], cidx_A, (char*)&alpha,
            this->tensors[nzIdxB[i]], cidx_B, (char*)&beta
          );

      sum.execute();
    }
    free(nzIdxA);
    free(nzIdxB);
  };

  template<typename dtype>
  void bsTensor<dtype>::sum(dtype                  alpha,
                            bsTensor<dtype>  &     A,
                            char const *           cidx_A,
                            dtype                  beta,
                            char const *           cidx_B,
                            std::vector<ivec>      nonZeroA,
                            std::vector<ivec>      nonZeroB,
                            Univar_Function<dtype> fseq,
                            bool                   verbose) {
    IASSERT(this->order == A.order);
    int nBlocks = nonZeroA.size();
    IASSERT(nonZeroA.size() == nonZeroB.size());
    int *nzIdxA = new int[nBlocks];
    int *nzIdxB = new int[nBlocks];

    checkDublicate(cidx_A);
    checkDublicate(cidx_B);



    for (int64_t i(0); i < nonZeroA.size(); i++){
      auto p =
        std::find( A->nonZeroCondition.begin()
                 , A->nonZeroCondition.end()
                 , nonZeroA[i]
                 );
      nzIdxA[i] = std::distance(A->nonZeroCondition.begin(), p);
      IASSERT(nzIdxA[i] < A.nonZeroCondition.size());

      p = std::find( this->nonZeroCondition.begin()
                   , this->nonZeroCondition.end()
                   , nonZeroB[i]
                   );
      nzIdxB[i] = std::distance(this->nonZeroCondition.begin(), p);
      IASSERT(nzIdxB[i] < this->nonZeroCondition.size());
      if (verbose){
        printf("bs %ld: ", i);
        for (int j(0); j < order; j++) printf("%d ", nonZeroA[i][j]);
        printf("| %d -> ", nzIdxA[i]);
        for (int j(0); j < order; j++) printf("%d ", nonZeroB[i][j]);
        printf(" | %d\n", nzIdxB[i]);
      }

   }

   for (int64_t i(0); i < nonZeroA.size(); i++){
     CTF_int::summation sum
       = CTF_int::summation(
           A.tensors[nzIdxA[i]], cidx_A, (char*)&alpha,
           this->tensors[nzIdxB[i]], cidx_B, (char*)&beta, fseq
         );

     sum.execute();
   }
    free(nzIdxA);
    free(nzIdxB);
  };


  template<typename dtype>
  void bsTensor<dtype>::contract(dtype             alpha,
                                 bsTensor<dtype>   & A,
                                 char const *      cidx_A,
                                 bsTensor<dtype>   & B,
                                 char const *      cidx_B,
                                 dtype             beta,
                                 char const *      cidx_C,
                                 bool              verbose) {

    checkDublicate(cidx_A);
    checkDublicate(cidx_B);
    checkDublicate(cidx_C);
    ivec idxA(A.order);
    ivec idxB(B.order);
    std::vector< std::pair<int, int> > ca, cb, ab;
    int aa(0), bb(0);
    for (int i(0); i < this->order; i++){
      auto c(cidx_C[i]);
      for (int j(0); j < A.order; j++)
        if ( cidx_A[j] == c) { idxA[aa++] = j; ca.push_back({i,j}); }
      for (int j(0); j < B.order; j++)
        if ( cidx_B[j] == c) { idxB[bb++] = j; cb.push_back({i,j}); }
    }

    for (int i(0); i < this->order; i++) printf("%c",cidx_C[i]);
    printf(" = ");
    for (int i(0); i < A.order; i++) printf("%c",cidx_A[i]);
    printf(" x ");
    for (int i(0); i < B.order; i++) printf("%c",cidx_B[i]);
    printf("\n");

    // this is the number of indices which appear on the lhs
    int lhsA(aa), lhsB(bb);
    int rhs(A.order - aa);
    IASSERT(rhs == B.order-bb);
    IASSERT(aa+bb == this->order);

    for (int i(0); i < A.order; i++){
      auto a(cidx_A[i]);
      for (int j(0); j < B.order; j++)
      if ( a == cidx_B[j]) {
        ab.push_back({i,j});
        idxA[aa++] = i;
        idxB[bb++] = j;
      }
    }

    IASSERT(aa == A.order);
    IASSERT(bb == B.order);
/*
    printf("A: ");
    for (int i(0); i < A.order; i++)
      printf("%d ", idxA[i]);
    printf("\nB: ");
    for (int i(0); i < B.order; i++)
      printf("%d ", idxB[i]);
    printf("\n\n");
*/

    auto nzA = A.nonZeroCondition;
    std::sort(nzA.begin(), nzA.end(), compare(idxA));
/*
    for (auto n: nzA){
      for (size_t p(0); p < A.order; p++) printf("%d ", n[p]);
      printf("\n");
    }

    printf("========\n");
*/
    auto nzB = B.nonZeroCondition;
    std::sort(nzB.begin(), nzB.end(), compare(idxB));
    auto nzC = this->nonZeroCondition;
    std::sort(nzC.begin(), nzC.end());


    std::vector< std::array<int,3> > tasks;
    for (size_t n(0); n < nzC.size(); n++){
      auto beginA = std::distance( nzA.begin()
                                 , std::find_if( nzA.begin()
                                               , nzA.end()
                                               , find(nzC[n], idxA, lhsA)
                                               )
                                 );
      auto endA   = std::distance( nzA.begin()
                                 , std::find_if_not( nzA.begin() + beginA
                                                   , nzA.end()
                                                   , find(nzC[n], idxA, lhsA)
                                                   )
                                 );

      auto beginB = std::distance( nzB.begin()
                                 , std::find_if( nzB.begin()
                                               , nzB.end()
                                               , find(nzC[n], idxB, lhsB)
                                               )
                                 );
      auto endB   = std::distance( nzB.begin()
                                 , std::find_if_not( nzB.begin() + beginB
                                                   , nzB.end()
                                                   , find(nzC[n], idxB, lhsB)
                                                   )
                                 );
      auto els(endA-beginA);
      IASSERT(els == endB - beginB);

      for (size_t p(0); p < this->order; p++) printf("%d", nzC[n][p]);
      if (verbose) printf("(%2d)", nzC[n][this->order]);
      if (verbose) printf(" = ");
      for (size_t i(0); i < els; i++){
        if (verbose){
          if (i) printf(" + ");
          for (size_t p(0); p < A.order; p++) printf("%d", nzA[beginA+i][p]);
          printf("(%2d)", nzA[beginA+i][A.order]);
          printf(" x ");
          for (size_t p(0); p < B.order; p++) printf("%d", nzB[beginB+i][p]);
          printf("(%2d)", nzB[beginB+i][B.order]);
        }
        tasks.push_back(
          {nzC[n][this->order], nzA[beginA+i][A.order], nzB[beginB+i][B.order]}
        );
      }
      if (verbose) printf("\n");

    }
    printf("=====\n");
    for (auto t: tasks){
      CTF_int::contraction ctr
        = CTF_int::contraction( A.tensors[t[1]], cidx_A
                              , B.tensors[t[2]], cidx_B, (char*)&alpha
                              , this->tensors[t[0]], cidx_C, (char*)&beta
                              );
      ctr.execute();
    }

  };



  template<typename dtype>
  void bsTensor<dtype>::slice(int64_t const *        offsets,
                              int64_t const *        ends,
                              dtype                  beta,
                              bsTensor<dtype> const &A,
                              int64_t const *        offsets_A,
                              int64_t const *        ends_A,
                              dtype                  alpha){

    // make sure that the tensors
    IASSERT(this->nBlocks == A.nBlocks);
    for (int i(0); i < this->nBlocks; i++)
      IASSERT( this->nonZeroCondition[i] == A.nonZeroCondition[i] );



    for (int i(0); i < this->nBlocks; i++){
      this->tensors[i]->slice(
        offsets, ends, (char*)&beta, A.tensors[i], offsets_A, ends_A, (char*)&alpha);
    }
  }
//
//  template<typename dtype>
//  void Tensor<dtype>::contract(dtype            alpha,
//                               CTF_int::tensor& A,
//                               const char *     idx_A,
//                               CTF_int::tensor& B,
//                               const char *     idx_B,
//                               dtype            beta,
//                               const char *     idx_C){
//    CTF_int::contraction ctr
//      = CTF_int::contraction(&A, idx_A, &B, idx_B, (char*)&alpha, this, idx_C, (char*)&beta);
//    ctr.execute();
//  }


}




