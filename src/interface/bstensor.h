#ifndef __BSTENSOR_H__
#define __BSTENSOR_H__

#include "../tensor/algstrct.h"
#include "../tensor/untyped_tensor.h"
#include "set.h"



using ivec = std::vector<int>;

namespace CTF {
  /**
   * \brief the block sparse tensor which holds a vector of CTF tenors
   */
  template <typename dtype=double>
  class bsTensor {
    public:

      /**
       * \brief defines tensor filled with zeros on the default algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      bsTensor(int                        order,
               std::vector<int64_t> const len,
               std::vector<int> const     sym,
               std::vector<ivec> const    nonZero,
               World *                    wrld=get_universe(),
               char const *               name=NULL,
               bool                       profile=0,
               CTF_int::algstrct const &  sr=Ring<dtype>());

      /**
       * \brief defines tensor filled with zeros on the default algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] wrld a world for the tensor to live in
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      bsTensor(int                        order,
               std::vector<int64_t> const len,
               std::vector<ivec> const    nonZero,
               World *                    wrld=get_universe(),
               char const *               name=NULL,
               bool                       profile=0,
               CTF_int::algstrct const &  sr=Ring<dtype>());



      /**
       * \brief copies a tensor, copying the data of A
       * \param[in] A tensor to copy
       */
      bsTensor(bsTensor<dtype> const & A);


      void init(int                       order,
                std::vector<int64_t>      len,
                int const *               sym,
                std::vector<ivec>         nonZero,
                World *                   wrld=get_universe(),
                char const *              name=NULL,
                bool                      profile=0,
                CTF_int::algstrct const & sr=Ring<dtype>()
               );


      /**
       * \brief  Gives the values associated with any set of indices.
       * \param[in] npair number of values to fetch
       * \param[in,out] pairs a prealloced pointer to key-value pairs
       */
      void read(int64_t         npair,
                int64_t const * global_idx,
                dtype         * data,
                int64_t const   block=-1,
                CTF_int::algstrct const & sr=Ring<dtype>()
               );


      void read_all(dtype *         data, int64_t const block=-1);



      /**
       * \brief writes in values associated with any set of indices
       * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major.
       * \param[in] npair number of values to write into tensor
       * \param[in] global_idx global index within tensor of value to write
       * \param[in] data values to  write to the indices
       */
      void write(int64_t         npair,
                 int64_t const * global_idx,
                 dtype const   * data,
                 int64_t const   block=-1,
                 CTF_int::algstrct const & sr=Ring<dtype>()
                );


      void read_dense_from_file(MPI_File & file, int64_t const block = -1);

      void slice(std::vector<int64_t> const offsets,
                 std::vector<int64_t> const ends,
                 dtype           beta,
                 bsTensor<dtype> const &A,
                 std::vector<int64_t> const offsets_A,
                 std::vector<int64_t> const ends_A,
                 dtype           alpha );


      /**
       * \brief sums B[idx_B] = beta*B[idx_B] + fseq(alpha*A[idx_A])
       * \param[in] alpha A scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
       * \param[in] beta B scaling factor
       * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
       * \param[in] fseq sequential operation to execute, default is multiply-add
       */
      void sum(dtype                  alpha,
               bsTensor<dtype> &      A,
               char const *           idx_A,
               dtype                  beta,
               char const *           idx_B,
               Univar_Function<dtype> fseq,
               bool                   verbose=false);

      /**
       * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
       * \param[in] alpha A scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
       * \param[in] beta B scaling factor
       * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
       */

      void sum(dtype             alpha,
               bsTensor<dtype> & A,
               char const *      idx_A,
               dtype             beta,
               char const *      idx_B,
               bool              verbose=false);


      /**
       * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
       * \param[in] alpha A scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
       * \param[in] beta B scaling factor
       * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
       */

      void sum(dtype             alpha,
               bsTensor<dtype> & A,
               char const *      idx_A,
               dtype             beta,
               char const *      idx_B,
               std::vector<ivec> nonZeroA,
               std::vector<ivec> nonZeroB,
               bool              verbose=false);


      void sum(dtype             alpha,
               bsTensor<dtype> & A,
               char const *      idx_A,
               dtype             beta,
               char const *      idx_B,
               std::vector<ivec> nonZeroA,
               std::vector<ivec> nonZeroB,
               Univar_Function<dtype> fseq,
               bool              verbose=false);


     /**
       * \brief contracts C[idx_C] = beta*C[idx_C] + alpha*A[idx_A]*B[idx_B]
       * \param[in] alpha A*B scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
       * \param[in] B second operand tensor
       * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
       * \param[in] beta C scaling factor
       * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
       */
      void contract(dtype             alpha,
                    bsTensor<dtype>   & A,
                    char const *      idx_A,
                    bsTensor<dtype>   & B,
                    char const *      idx_B,
                    dtype             beta,
                    char const *      idx_C,
                    bool              verbose=false);


      /**
       * \brief get the tensor name
       * \return tensor name
       */

      char const * get_name() const;

      void checkDublicate(std::string t);
      size_t orderToN(ivec o);
      std::function<int(const ivec &, const ivec &)> compare(const ivec p);
      std::function<int(const ivec &)> find(const ivec, const std::vector< std::pair<int,int> >);


      /**
       * \brief frees CTF tensor
       */
      ~bsTensor();


      std::vector<CTF_int::tensor *> tensors;
      std::vector<ivec> nonZeroCondition; 
      int64_t nBlocks;
      std::vector<int64_t> lens;
      int order;
      char * name;
      CTF::World world;
  };
  /**
   * @}
   */
}

#include "bstensor.cxx"
 
#endif

