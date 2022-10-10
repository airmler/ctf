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
  bsTensor<dtype>::bsTensor(int                        order_,
                            std::vector<int64_t> const len_,
                            std::vector<int> const     sym_,
                            std::vector<ivec> const    nonZero_,
                            World *                    world_,
                            std::string const          name_,
                            bool                       profile_,
                            CTF_int::algstrct const &  sr_) {
    assert(order_ == len_.size());
    this->init(order_, len_, sym_, nonZero_, world_, name_, profile_, sr_);
  }

  template<typename dtype>
  bsTensor<dtype>::bsTensor(int                        order_,
                            std::vector<int64_t> const len_,
                            std::vector<ivec> const    nonZero_,
                            World *                    world_,
                            std::string const          name_,
                            bool                       profile_,
                            CTF_int::algstrct const &  sr_) {
    assert(order_ == len_.size());
    std::vector<int> sym(order_, NS);
    this->init(order_, len_, sym.data(), nonZero_, world_, name_, profile_, sr_);
  }

  template<typename dtype>
  bsTensor<dtype>::bsTensor(bsTensor<dtype> const & A)
  {
    order = A.order;
    lens = A.lens;
    name = A.name;
    nonZeroCondition = A.nonZeroCondition;
    nBlocks = A.nBlocks;
    //TODO
    tensors = A.tensors;
  }


  template<typename dtype>
  bsTensor<dtype>::~bsTensor(){
    for (auto &t: tensors) delete t;
  }


  template<typename dtype>
  void bsTensor<dtype>::init(int order_,
                             std::vector<int64_t> const len_,
                             int const *                sym_,
                             std::vector<ivec>          nonZero_,
                             World *                    world_,
                             std::string const          name_,
                             bool                       profile_,
                             CTF_int::algstrct const &  sr_) {
    IASSERT(sizeof(dtype)==sr_.el_size);
    this->order = order_;
    lens = len_;
    if (name_.size()) {
      this->name = name_;
    } else {
      this->name = "";
      for (int i=0; i<4; i++){
        this->name += 'A'+(world_->glob_wrld_rng()%26);
      }
      this->name += '0'+(order_/10);
      this->name += '0'+(order_%10);
      this->name[6] = '\0';
    }
    nBlocks = nonZero_.size();
    if (nBlocks <= 0) {
      printf("Tensor %s", this->name.c_str());
      IASSERT(nBlocks>0);
    }
    int i(0);
    for (auto nz: nonZero_){
      if (nz.size() != order_) printf("order %d != nz.size() %ld\n"
                                     , order_, nz.size());
      IASSERT(nz.size()==order_);
      ivec r(nz);
      r.push_back(i++);
      nonZeroCondition.push_back(r);
      tensors.push_back(
        new CTF_int::tensor(&sr_, order_, len_.data(), sym_, world_, 1, this->name.c_str(), profile_)
      );
    }
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
//  TODO
//    IASSERT(block>=0); //not implemented yet
    IASSERT(block< this->nBlocks);
    int64_t npair;
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
      // WATCH OUT: global_idx is the same for all different blocks!!!
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
  void bsTensor<dtype>::read_dense_from_file(MPI_File & file, int64_t block){
    IASSERT(block < this->nBlocks);
    int64_t elements = std::accumulate( lens.begin()
                                      , lens.end()
                                      , 1L
                                      , std::multiplies<int64_t>());
    if (block < 0){
      int64_t offset(0L);
      for (auto &t: this->tensors) {
        t->read_dense_from_file(file, offset*sizeof(dtype));
        offset += elements;
      }
    }
    else
      this->tensors[block]->read_dense_from_file(file, block*elements*sizeof(dtype));
  }

  template<typename dtype>
  void bsTensor<dtype>::write_dense_to_file(MPI_File & file, int64_t block){
    IASSERT(block < this->nBlocks);
    int64_t elements = std::accumulate( lens.begin()
                                      , lens.end()
                                      , 1L
                                      , std::multiplies<int64_t>());
    if (block < 0){
      int64_t offset(0L);
      for (auto &t: this->tensors) {
        t->write_dense_to_file(file, offset*sizeof(dtype));
        offset += elements;
      }
    }
    else
      this->tensors[block]->write_dense_to_file(file, block*elements*sizeof(dtype));
  }


  template<typename dtype>
  std::string const bsTensor<dtype>::get_name() const {
    return name;
  }

  template<typename dtype>
  void bsTensor<dtype>::checkDublicate(std::string t){
    std::sort(t.begin(), t.end());
    auto it = std::unique(t.begin(), t.end());
    if (it != t.end()) IASSERT(0);
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
  bsTensor<dtype>::find(const ivec c, const std::vector< std::pair<int,int> > ca)
  {
    return [c, ca] (const ivec &a) -> int {
      for (size_t i(0); i < ca.size(); i++)
      for (auto &x: ca)  if ( a[x.second] != c[x.first]) return false;
      return true;
    };
  }


  template<typename dtype>
  void bsTensor<dtype>::sum(dtype                  alpha,
                            bsTensor<dtype>      & A,
                            char const *           cidx_A,
                            dtype                  beta,
                            char const *           cidx_B,
                            bool                   verbose) {
// sum can be either a permutation
// a) B["Gia"] = A["Gai"]
// or an expression like
// b) B["ijab"] = A["i"]
// in both cases we have to find the correct non-zero block summation
    // TODO: we have to resolve this assert. if the right side has an higher
    //       order, it implies a sum over the additional index
    IASSERT(this->order >= A.order);
    IASSERT(this->nBlocks >= A.nBlocks);
    checkDublicate(cidx_A);  checkDublicate(cidx_B);
    std::string idxA(cidx_A), idxB(cidx_B);
   // These are the nonZeroIndices for A
    // We initialize these indices as the nonZeroCondition of B
    auto nzA(this->nonZeroCondition);   auto nzB(this->nonZeroCondition);
    std::sort( nzB.begin(), nzB.end());

    // idx maps the indices on the left with the incides on the right
    std::vector<int> idx;
    for (int i(0); i < A.order; i++){
      auto it = std::find( std::begin(idxB), std::end(idxB), idxA[i]);
      if (it == std::end(idxB)) continue;
      idx.push_back( std::distance( std::begin(idxB), it ) );
    }

    if (this->order == A.order){
      IASSERT(this->nBlocks == A.nBlocks);
      // i.e.: B["Gia"] = A["Gai"] -> idx: 0,2,1
      // we sort nzA according to the indices-sequence in idx
      if (verbose && !this->world.rank) {
        for (auto i: idx) printf("%d ", i);
        std::cout << std::endl;
      }
      std::sort(nzA.begin(), nzA.end(), compare(idx));
    }
    else {
    // Right now: this else means left hand side has more indices than the
    //            right hand side
    // we remove the columns in nzA which do not appear on the right side
    // example: "ia" = "i"
    // {0,0} = {0}, {1,0} = {1}, {0,1} = {0}, {1,1} = {1}
      std::vector<int> iidx(this->order);
      std::vector<int> toRemove;
      std::iota(iidx.begin(), iidx.end(), 0);
      std::set_difference( iidx.begin(), iidx.end()
                         , idx.begin(), idx.end()
                         , std::back_inserter(toRemove)
                         );
      if (verbose && !this->world.rank) printf("\n\n");
      if (verbose && !this->world.rank) for (auto t: toRemove) printf("%d ", t);
      if (verbose && !this->world.rank) printf("\n\n");
      std::sort(nzA.begin(), nzA.end());
      for (auto &n: nzA){
        for (auto t: toRemove) n[t] = -1;
        n.erase( std::remove( n.begin(), n.end(), -1), n.end() );
        if (verbose && !this->world.rank) {
          for (auto nn: n) printf("%d ", nn);
          printf("\n");
        }
        // the last element in n is the index of the nonZeroCondition
        // this vector should appear in the list of the nonZeroConditions of A
        bool replaced(false);
        for (auto a: A.nonZeroCondition){
          if ( std::equal( a.begin(), a.end() - 1, n.begin()) ){
            n.back() = a.back();
            replaced = true;
          }
        }
        IASSERT(replaced);
      }
    }

    if (verbose && !this->world.rank) {
      printf( "%s[%s] <- %s[%s]\n"
            , this->name.c_str(), cidx_B, A.name.c_str(), cidx_A);
      for (int i(0); i < this->nBlocks; i++){
        printf("bs %d: ", i);
        for (int j(0); j < this->order; j++)
          printf("%d ", nzB[i][j]);
        printf("| %d <- ", nzB[i][this->order]);
        for (int j(0); j < A.order; j++) printf("%d ", nzA[i][j]);
        printf(" | %d\n", nzA[i][A.order]);
      }
    }


    for (int i(0); i < this->nBlocks; i++){
      CTF_int::summation sum
        = CTF_int::summation(
            A.tensors[nzA[i][A.order]], cidx_A, (char*)&alpha,
            this->tensors[nzB[i][this->order]], cidx_B, (char*)&beta
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
                            Univar_Function<dtype>& fseq,
                            bool                   verbose){
    // sum can be either a permutation
    // a) B["Gia"] = A["Gai"]
    // or an expression like
    // b) B["ijab"] = A["i"]
    // in both cases we have to find the correct non-zero block summation
    // TODO: we have to resolve this assert. if the right side has an higher
    //       order, it implies a sum over the additional index
    IASSERT(this->order >= A.order);
    IASSERT(this->nBlocks >= A.nBlocks);
    checkDublicate(cidx_A);  checkDublicate(cidx_B);
    std::string idxA(cidx_A), idxB(cidx_B);
   // These are the nonZeroIndices for A
    // We initialize these indices as the nonZeroCondition of B
    auto nzA(this->nonZeroCondition);   auto nzB(this->nonZeroCondition);
    std::sort( nzB.begin(), nzB.end());

    // idx maps the indices on the left with the incides on the right
    std::vector<int> idx;
    for (int i(0); i < A.order; i++){
      auto it = std::find( std::begin(idxB), std::end(idxB), idxA[i]);
      if (it == std::end(idxB)) continue;
      idx.push_back( std::distance( std::begin(idxB), it ) );
    }

    if (this->order == A.order){
      IASSERT(this->nBlocks == A.nBlocks);
      // i.e.: B["Gia"] = A["Gai"] -> idx: 0,2,1
      // we sort nzA according to the indices-sequence in idx
      if (verbose && !this->world.rank) {
        for (auto i: idx) printf("%d ", i);
        std::cout << std::endl;
      }
      std::sort(nzA.begin(), nzA.end(), compare(idx));
    }
    else {
    // Right now: this else means left hand side has more indices than the
    //            right hand side
    // we remove the columns in nzA which do not appear on the right side
    // example: "ia" = "i"
    // {0,0} = {0}, {1,0} = {1}, {0,1} = {0}, {1,1} = {1}
      std::vector<int> iidx(this->order);
      std::vector<int> toRemove;
      std::iota(iidx.begin(), iidx.end(), 0);
      std::set_difference( iidx.begin(), iidx.end()
                         , idx.begin(), idx.end()
                         , std::back_inserter(toRemove)
                         );
      if (verbose && !this->world.rank) printf("\n\n");
      if (verbose && !this->world.rank) for (auto t: toRemove) printf("%d ", t);
      if (verbose && !this->world.rank) printf("\n\n");
      std::sort(nzA.begin(), nzA.end());
      for (auto &n: nzA){
        for (auto t: toRemove) n[t] = -1;
        n.erase( std::remove( n.begin(), n.end(), -1), n.end() );
        if (verbose && !this->world.rank) {
          for (auto nn: n) printf("%d ", nn);
          printf("\n");
        }
        // the last element in n is the index of the nonZeroCondition
        // this vector should appear in the list of the nonZeroConditions of A
        bool replaced(false);
        for (auto a: A.nonZeroCondition){
          if ( std::equal( a.begin(), a.end() - 1, n.begin()) ){
            n.back() = a.back();
            replaced = true;
          }
        }
        IASSERT(replaced);
      }
    }

    if (verbose && !this->world.rank) {
      printf( "%s[%s] <- %s[%s]\n"
            , this->name.c_str(), cidx_B, A.name.c_str(), cidx_A);
      for (int i(0); i < this->nBlocks; i++){
        printf("bs %d: ", i);
        for (int j(0); j < this->order; j++)
          printf("%d ", nzB[i][j]);
        printf("| %d <- ", nzB[i][this->order]);
        for (int j(0); j < A.order; j++) printf("%d ", nzA[i][j]);
        printf(" | %d\n", nzA[i][A.order]);
      }
    }


    for (int i(0); i < this->nBlocks; i++){
      CTF_int::summation sum
        = CTF_int::summation(
            A.tensors[nzA[i][A.order]], cidx_A, (char*)&alpha,
            this->tensors[nzB[i][this->order]], cidx_B, (char*)&beta, &fseq
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
    IASSERT(nonZeroA.size() == nonZeroB.size());
    std::vector<int> nzIdxA(nonZeroA.size());
    std::vector<int> nzIdxB(nonZeroB.size());
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
      if (verbose && !this->world.rank){
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
  };

  template<typename dtype>
  void bsTensor<dtype>::sum(dtype                  alpha,
                            bsTensor<dtype>  &     A,
                            char const *           cidx_A,
                            dtype                  beta,
                            char const *           cidx_B,
                            std::vector<ivec>      nonZeroA,
                            std::vector<ivec>      nonZeroB,
                            Univar_Function<dtype> &fseq,
                            bool                   verbose) {
    IASSERT(this->order == A.order);
    IASSERT(nonZeroA.size() == nonZeroB.size());
    std::vector<int> nzIdxA(nonZeroA.size());
    std::vector<int> nzIdxB(nonZeroB.size());

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
      if (verbose && !this->world.rank){
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
            this->tensors[nzIdxB[i]], cidx_B, (char*)&beta, &fseq
          );

      sum.execute();
    }
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
    // we cannot handle dublicate indices: C["ij"] = A["ik"] B["kkj"]
    // TODO: We want to allow the following contractions:
    // C["ijl"] = A["ikl"] * B["kjl"]
    // which implies a contraction over k and a pointwise multiplication of l
    checkDublicate(cidx_A); checkDublicate(cidx_B); checkDublicate(cidx_C);

    if (verbose && !this->world.rank)
      printf( "%s[%s] = %s[%s] x %s[%s]:\n"
            , this->name.c_str(), cidx_C, A.name.c_str(), cidx_A, B.name.c_str(), cidx_B);

    // Workflow: - idxA && idxB are std::vector<int> which determine the way nonZeroConditions
    //             of A && B are sorted. E.g.: C["ijab"] = A["ikca"] B["bckj"], A is sorted such
    //             that ia also the slowest indices (for B it would be jb).
    //           - ca,cb,ab are the indices which match between C && A (and the other tensors,
    //             respectively)

    ivec idxA, idxB;
    std::vector< std::pair<int, int> > ca, cb, ab;
    // Find the indices of A && B which appear on C
    for (int i(0); i < this->order; i++){
      for (int j(0); j < A.order; j++)
        if ( cidx_A[j] == cidx_C[i]) { idxA.push_back(j); ca.push_back({i,j}); }
      for (int j(0); j < B.order; j++)
        if ( cidx_B[j] == cidx_C[i]) { idxB.push_back(j); cb.push_back({i,j}); }
    }

    for (int i(0); i < A.order; i++)
    for (int j(0); j < B.order; j++)
    if ( cidx_A[i] == cidx_B[j]) {
      ab.push_back({i,j});
      idxA.push_back(i);
      idxB.push_back(j);
    }

    // We sort nzA, nzB such that the contraction indices are the fastest.
    auto nzA = A.nonZeroCondition;
    std::sort(nzA.begin(), nzA.end(), compare(idxA));
    auto nzB = B.nonZeroCondition;
    std::sort(nzB.begin(), nzB.end(), compare(idxB));
    auto nzC = this->nonZeroCondition;
    std::sort(nzC.begin(), nzC.end());

    // We write all tasks in the following object
    std::vector< std::array<int,3> > tasks;

    // We loop over all nonZeroConditions in C and look for
    // a given nonZeroCondition of C for all nonZeroConditions of A && B
    // which fulfill the overall nonZeroConditions
    // Eg. C[abij] = A[akic] * B[kbcj] for C[1001] lead to the following
    // conditions for A: A[1001] && A[1100], and B: B[0011] && B[1001]
    // As we have sorted A && B, the corresponding tuples appear consecutive
    // in the list
    for (size_t n(0); n < nzC.size(); n++){
      auto beginA = std::distance( nzA.begin()
                                 , std::find_if( nzA.begin()
                                               , nzA.end()
                                               , find(nzC[n], ca)
                                               )
                                 );
      auto endA   = std::distance( nzA.begin()
                                 , std::find_if_not( nzA.begin() + beginA
                                                   , nzA.end()
                                                   , find(nzC[n], ca)
                                                   )
                                 );

      auto beginB = std::distance( nzB.begin()
                                 , std::find_if( nzB.begin()
                                               , nzB.end()
                                               , find(nzC[n], cb)
                                               )
                                 );
      auto endB   = std::distance( nzB.begin()
                                 , std::find_if_not( nzB.begin() + beginB
                                                   , nzB.end()
                                                   , find(nzC[n], cb)
                                                   )
                                 );

      auto els(endA-beginA);
      if (els != endB - beginB){
        for (int i(0); i < endA - beginA; i++)
        for (int j(0); j < endB - beginB; j++){
          bool m(true);
          for (auto &x: ab)
            if (nzA[beginA+i][x.first] != nzB[beginB+j][x.second]) m = false;
          if (!m) continue;
          tasks.push_back(
            {nzC[n][this->order], nzA[beginA+i][A.order], nzB[beginB+j][B.order]}
          );
        }
      } else {
        for (size_t i(0); i < els; i++){
// if we require the following, V = G * G would not work properly!
//          bool m(true);
//          for (auto x: ab)
//            if (nzA[beginA+i][x.first] != nzB[beginB+i][x.second]) m = false;
//          if (!m) continue;
          tasks.push_back(
            {nzC[n][this->order], nzA[beginA+i][A.order], nzB[beginB+i][B.order]}
          );
        }
      }


      if (verbose && !this->world.rank) {
        for (size_t p(0); p < this->order; p++) printf(" %d", nzC[n][p]);
        printf(" (%*d)", 1 + (int) log10(this->nBlocks), nzC[n][this->order]);
        printf(" = ");
        for (size_t i(0); i < els; i++){
            if (i) printf(" + ");
            for (size_t p(0); p < A.order; p++) printf(" %d", nzA[beginA+i][p]);
            printf(" (%*d)", 1+(int) log10(A.nBlocks), nzA[beginA+i][A.order]);
            printf(" x ");
            for (size_t p(0); p < B.order; p++) printf(" %d", nzB[beginB+i][p]);
            printf(" (%*d)", 1+(int) log10(B.nBlocks), nzB[beginB+i][B.order]);
        }
        printf("\n");
      }

    }
    if (verbose && !this->world.rank) printf("this many tasks %ld\n",tasks.size());
    if (verbose && !this->world.rank) printf("--\n");
    // We have to be careful here: if beta is the zero element the final
    // result would only be the result of last contraction
    // (for this non-zero element)
    // thats why we zero the tensor C, then setting beta to one!!
    auto sr = this->tensors[0]->sr;
    if (beta == (dtype) 0) {
      for (auto t: this->tensors) sr->set(t->data, sr->addid(), t->size);
      beta = (dtype) 1;
    }
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
  void bsTensor<dtype>::slice(std::vector<int64_t> const offsets,
                              std::vector<int64_t> const ends,
                              dtype                  beta,
                              bsTensor<dtype> const &A,
                              std::vector<int64_t> const offsets_A,
                              std::vector<int64_t> const ends_A,
                              dtype                  alpha){

    // make sure that the tensors have the same nonZeroConditions
    IASSERT(this->nBlocks == A.nBlocks);
    for (int i(0); i < this->nBlocks; i++)
      IASSERT( this->nonZeroCondition[i] == A.nonZeroCondition[i] );

    for (int i(0); i < this->nBlocks; i++)
      this->tensors[i]->slice( offsets.data(), ends.data()
                             , (char*)&beta, A.tensors[i]
                             , offsets_A.data(), ends_A.data()
                             , (char*)&alpha);

  }
}
