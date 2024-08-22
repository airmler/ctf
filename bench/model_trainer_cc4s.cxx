/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{
  * \addtogroup model_trainer
  * @{
  * \brief Executes a set of different contractions on different processor counts to train model parameters
  */

#include <sys/types.h>
#include <sys/stat.h>
#include <ctf.hpp>
#define TEST_SUITE
#include "../examples/ccsd.cxx"
#include "../examples/sparse_mp3.cxx"
#undef TEST_SUITE
using namespace CTF;

namespace CTF_int{
  void update_all_models(MPI_Comm comm);
}

struct Ccsd_dimensions {
  int64_t No;
  int64_t Nv;
  int64_t Nx;
  int64_t Ng;
};

Ccsd_dimensions get_ccsd_dimensions(double mem_per_core, int64_t nvfac, World &dw) {
  int np;
  MPI_Comm_size(dw.comm, &np);
  int64_t No(10);
  while ( No*No*No*No*nvfac*nvfac*8./np/1024/1024 < mem_per_core) No++;
  return Ccsd_dimensions({No, No*nvfac, No, (int64_t) No*nvfac*2.5});
}

void ph1_contraction(int64_t No, int64_t Nv, World &dw) {
  int64_t vvoo[] = {Nv, Nv, No, No};
  int syms[] = {NS, NS, NS, NS};
  CTF::Tensor< double > T(4, vvoo, syms, dw, "T");
  CTF::Tensor< double > V(4, vvoo, syms, dw, "V");
  CTF::Tensor< double > R(4, vvoo, syms, dw, "R");
  V.fill_random(0, 1);
  T.fill_random(0, 1);
  R["abij"] = T["acik"] * V["cbkj"];
}

void ph2_contraction(int64_t No, int64_t Nv, World &dw) {
  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t ovvo[] = {No, Nv, Nv, No};
  int syms[] = {NS, NS, NS, NS};
  CTF::Tensor< double > T(4, vvoo, syms, dw, "T");
  CTF::Tensor< double > V(4, ovvo, syms, dw, "V");
  CTF::Tensor< double > R(4, vvoo, syms, dw, "R");
  V.fill_random(0, 1);
  T.fill_random(0, 1);
  R["abij"] = T["acik"] * V["kbcj"];
}

void ggv_contraction(int64_t Nv, int64_t Nx, int64_t Ng, World &dw) {
  int64_t gxv[] = {Ng, Nx, Nv};
  int64_t vvxx[] = {Nv, Nv, Nx, Nx};
  int syms[] = {NS, NS, NS, NS};
  CTF::Tensor< double > G(3,  gxv, syms, dw, "B");
  CTF::Tensor< double > V(4, vvxx, syms, dw, "C");
  G.fill_random(0, 1);
  V["cdxy"] = G["Gxc"] * G["Gyd"];
}

void rvt_contraction(int64_t No, int64_t Nv, int64_t Nx, World &dw) {
  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t xxoo[] = {Nx, Nx, No, No};
  int64_t vvxx[] = {Nv, Nv, Nx, Nx};
  int syms[] = {NS, NS, NS, NS};
  CTF::Tensor< double > T(4, vvoo, syms, dw, "T");
  CTF::Tensor< double > V(4, vvxx, syms, dw, "V");
  CTF::Tensor< double > R(4, xxoo, syms, dw, "R");
  V.fill_random(0, 1);
  T.fill_random(0, 1);
  R["abij"] = V["xyab"] * T["xyij"];
}

void train_ccsd(World & dw, double mem_per_core, int64_t nvfac, int c_id){
  auto dim = get_ccsd_dimensions(mem_per_core, nvfac, dw);
  if (c_id & 1) ph1_contraction(dim.No, dim.Nv, dw);
  if (c_id & 2) ph2_contraction(dim.No, dim.Nv, dw);
  if (c_id & 4) ggv_contraction(dim.Nv, dim.Nx, dim.Ng, dw);
  if (c_id & 8) rvt_contraction(dim.No, dim.Nv, dim.Nx, dw);
}



void train_all(std::string dump_path, int num_iterations, int rounds, int ppn){
  World dw("hallo", 0, ppn);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  for (int i=0; i<num_iterations; i++){
    if (rank == 0){
      printf("Starting iteration %d/%d\n", i+1,num_iterations);
    }
    for (int j(0); j < rounds; j++) {
      train_ccsd(dw, 10.,  8, 15);
      train_ccsd(dw, 10., 12, 15);
      train_ccsd(dw, 25.,  8, 15);
      train_ccsd(dw, 25., 12, 15);
      train_ccsd(dw, 25., 16, 15);
      CTF_int::update_all_models(dw.comm);
      if (rank == 0) printf("Completed training round %d/%d\n", j+1, rounds);
    }
  }


//  CTF_int::write_all_models(coeff_file);
  if (rank == 0) CTF_int::print_all_models();

  if (dump_path.size()) CTF_int::dump_touched_models(dump_path);

}

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  std::string dump_path("./data");
  int iterations(3), rounds(3), ppn(0);
  if (getCmdOption(input_str, input_str+in_num, "-write")){
    dump_path = getCmdOption(input_str, input_str+in_num, "-write");
  }
  if (getCmdOption(input_str, input_str+in_num, "-ppn")){
    ppn = atoi(getCmdOption(input_str, input_str+in_num, "-ppn"));
  }

  struct stat info;
  if (!rank) {
    if(!stat( dump_path.c_str(), &info ) != 0 ) {
      printf( "Warning: dumping data into existing directory %s.\n", dump_path.c_str() );
    } else {
      mkdir(dump_path.c_str(), 0777);
    }
    printf("we train\n");
  }
  train_all(dump_path, iterations, rounds, ppn);


  MPI_Finalize();
  return 0;
}

/**
 * @}
 * @}
 */
