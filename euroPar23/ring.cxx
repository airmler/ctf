#include <ctf.hpp>
#include <mpi.h>
#include <chrono>
#include <thread>
#include <vector>

void ring(int64_t No, int64_t Nv, int rank, int np)
{
  int64_t voov[] = {Nv, No, No, Nv};
  int64_t vvoo[] = {Nv, Nv, No, No};
  int syms[] = {NS, NS, NS, NS};

  CTF::World dw;

  CTF::Tensor<double> R(4, vvoo, syms, dw, "R");
  CTF::Tensor<double> V(4, voov, syms, dw, "V");
  CTF::Tensor<double> T(4, vvoo, syms, dw, "T");

  T.fill_random(0, 1);
  V.fill_random(0, 1);

  double begin;
  std::vector<double> time;


  for (int it(0); it < 5; it++) {
    MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();
    R["abij"] = V["akic"] * T["cbkj"];
    MPI_Barrier(MPI_COMM_WORLD);
    time.push_back(MPI_Wtime() - begin);
  }

  double flops(2.0*(No*No*Nv*Nv*No*Nv));
  double mem(8.0*3.0*(No*No*Nv*Nv)/1024.0/1024./1024.0);
  if (rank ==0)
    printf( "No Nv: %ld %ld , %d cores, mem: %f, Time: %f %f %f %f %f\n"
          , No, Nv, np, mem*1024.0/np, time[0], time[1], time[2], time[3], time[4]);
}

int main(int argc, char ** argv){
  int rank, np;


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    int64_t No(0), Nv(0), ppn(1);
    bool dryRun(false);
    //NOTE: this crashes if you dont keep the correct order!!
    for (int i(1); i < argc; i++){
      std::string arg(argv[i]);
      if (arg == "-No") No = std::atoi(argv[++i]);
      if (arg == "-Nv") Nv = std::atoi(argv[++i]);
    }
    if (No > 0 && Nv > 0 ) ring(No, Nv, rank, np);
    else
      if (rank==0)
        printf("gimme right format: ./exe -No 14 -Nv 120 (-d 10) -ppn 1\n");
  }
  MPI_Finalize();
  return 0;

}

