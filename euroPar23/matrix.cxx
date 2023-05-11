#include <mpi.h>
#include <chrono>
#include <thread>
#include <vector>
#include <ctf.hpp>

void matrix(int64_t i, int64_t j, int64_t k, int rank, int np)
{ 
  int64_t ii[] = {i, j};
  int64_t ij[] = {i, k};
  int64_t ji[] = {k, j};
  int syms[] = {NS, NS};

  CTF::World dw;
  
  CTF::Tensor<double> A(2, ij, syms, dw, "A");
  CTF::Tensor<double> B(2, ji, syms, dw, "B");
  CTF::Tensor<double> C(2, ii, syms, dw, "C");
  
  A.fill_random(0, 1);
  B.fill_random(0, 1);
    
  double begin;
  std::vector<double> time;

  for (int it(0); it < 5; it++) {
    MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();
    C["ij"] = A["ik"] * B["kj"];
    MPI_Barrier(MPI_COMM_WORLD);
    time.push_back(MPI_Wtime() - begin);
  }

  double flops(2.0*(i*j*k));
  double mem(8.0*(i*j+i*k+k*j)/1024.0/1024./1024.0);
  if (rank ==0)
    printf( "m * n * k: %ld %ld %ld, %d cores, mem: %f, Time: %f %f %f %f %f\n"
          , i, j, k, np, mem*1024.0/np, time[0], time[1], time[2], time[3], time[4]
          );
}

int main(int argc, char ** argv){
  int rank, np;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    int64_t ii(0), jj(0), kk(0), ppn(1);
    bool dryRun(false);
    int m(-1);

    for (int i(1); i < argc; i++){
      std::string arg(argv[i]);
      //NOTE: this crashes if you dont keep the correct order!!
      if (arg == "-i") ii = std::atoi(argv[++i]);
      if (arg == "-j") jj = std::atoi(argv[++i]);
      if (arg == "-k") kk = std::atoi(argv[++i]);
    }
    if (ii > 0 && jj > 0 && kk > 0) matrix(ii, jj, kk, rank, np);
    else
      if (rank==0)
        printf("gimme right format: ./exe -i 14 -j 12 -k 142\n");
  }
  MPI_Finalize();
  return 0;

}

