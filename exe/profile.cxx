#include <ctf.hpp>
#include <mpi.h>
#include "../src/interface/contraction_collector.h"
#include <chrono>
#include <thread>

CTF::World worldCreator(bool isDry, int np){
  if (!isDry) return CTF::World();
  //return CTF::World(MPI_COMM_WORLD, 72, np);
  return CTF::World("low", np);
}


void phContraction(int64_t No, int64_t Nv, int rank, int np, bool dryRun) {
  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t voov[] = {Nv, No, No, Nv};
  int64_t vvvv[] = {Nv, Nv, Nv, Nv};
  int syms[] = {NS, NS, NS, NS};

  int64_t storage(Nv*Nv*No*No*8*3/1024/1024);

  double start, end, est;
  {
    CTF::World dw = worldCreator(dryRun, np);
    CTF::Contraction_collector cc;
    CTF::Flop_counter count;

    CTF::Tensor<double> A(4, voov, syms, dw, "A");
    CTF::Tensor<double> B(4, vvoo, syms, dw, "B");
    CTF::Tensor<double> C(4, vvoo, syms, dw, "C");
    A.fill_random(0, 1);
    B.fill_random(0, 1);

    MPI_Barrier(dw.comm);
    start = MPI_Wtime();
    C["abij"] = A["akic"] * B["cbkj"];
    C["abij"] = A["akic"] * B["cbkj"];
    C["abij"] = A["akic"] * B["cbkj"];
    C["abij"] = A["bkjc"] * B["acki"];
    C["abij"] = A["akjc"] * B["bcki"];
  
    MPI_Barrier(dw.comm);
    end = MPI_Wtime();

    if (!rank) cc.analyze(1);
  }
  double flops(No*Nv*No*Nv*No*Nv*2);
  if (rank==0) printf("#0 cores %d mem %f, est: %f, ph-time: %f, flops/s/core: %f\n"
         , np, double(storage)/1024./np, est, end-start, flops/(end-start)/1e9/np);

}


int main(int argc, char ** argv){
  int rank, np;


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    int64_t No(-1), Nv(-1);
    bool dryRun(false);

    for (int i(1); i < argc; i++){
      std::string arg(argv[i]);
      if (arg == "-No") No = std::atoi(argv[++i]);
      if (arg == "-Nv") Nv = std::atoi(argv[++i]);
      if (arg == "-d")  { dryRun = true; np = std::atoi(argv[++i]); }
    }

    if (No > 0 && Nv > 0)
      phContraction(No, Nv, rank, np, dryRun);
    else
      if (rank==0) printf("usage: ./exe -No 12 -Nv 24 (-d 72)\n");
  }
  MPI_Finalize();
  return 0;

}

