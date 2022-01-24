#include <ctf.hpp>
#include <mpi.h>
#include <chrono>
#include <thread>
#include <numeric>

using ivec = std::vector<int>;

CTF::World worldkacker(bool isDry, int np){
  if (!isDry) return CTF::World();
  return CTF::World("LIFE", np);
}



void transposeG(int64_t No, int64_t Nv, int64_t Ng){

  int64_t Gia[] = {Ng, No, Nv};
  int64_t Gai[] = {Ng, Nv, No};
  int syms[] = {NS, NS, NS};
  CTF::World dw(MPI_COMM_WORLD, 0, NULL);

  std::vector< ivec > a(9);
  for (auto &aa: a) aa.resize(3);
  a[0] = {0, 0, 0};
  a[1] = {1, 0, 1};
  a[2] = {2, 0, 2};
  a[3] = {2, 1, 0};
  a[4] = {0, 1, 1};
  a[5] = {1, 1, 2};
  a[6] = {1, 2, 0};
  a[7] = {2, 2, 1};
  a[8] = {2, 2, 2};
 
  CTF::bsTensor<double> G(3, Gia, syms, a, &dw, "C");
  CTF::bsTensor<double> F(3, Gai, syms, a, &dw, "D");
 
  int64_t el(Ng*No*Nv);
  std::vector<double> data(el);
  std::vector<int64_t> idx(el);
  for (int64_t ii(0); ii < el; ii++){
    data[ii] = 7.3*ii;
    idx[ii] = ii;
  }
  size_t n(0);
  if (!dw.rank) n = el;
  G.write(n, idx.data(), data.data(), 0);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 9.1*ii;
  G.write(n, idx.data(), data.data(), 1);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 3.6*ii;
  G.write(n, idx.data(), data.data(), 2);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 1.2*ii;
  G.write(n, idx.data(), data.data(), 3);




  for (int64_t ii(0); ii < el; ii++) data[ii] = 0.0;
  F.sum(1.0, G, "Gai", 1.0, "Gia", G.nonZeroCondition, F.nonZeroCondition);
  //F.sum(1.0, G, "Gai", 1.0, "Gia", true);
 
  F.read(el, idx.data(), data.data(), 0);
  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);
  printf("=======\n");
  F.read(el, idx.data(), data.data(), 1);
  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);

}


void transposeT(int64_t No, int64_t Nv){

  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t oovv[] = {No, No, Nv, Nv};
  int syms[] = {NS, NS, NS, NS};
  CTF::World dw(MPI_COMM_WORLD, 0, NULL);

  std::vector< ivec > a(8);
  for (auto &aa: a) aa.resize(4);
  a[0] = {0, 0, 0, 0};
  a[1] = {0, 0, 1, 1};
  a[2] = {0, 1, 0, 1};
  a[3] = {0, 1, 1, 0};
  a[4] = {1, 0, 0, 1};
  a[5] = {1, 0, 1, 0};
  a[6] = {1, 1, 0, 0};
  a[7] = {1, 1, 1, 1};
  CTF::bsTensor<double> C(4, vvoo, syms, a, &dw, "C");
  CTF::bsTensor<double> D(4, oovv, syms, a, &dw, "D");
 
  int64_t el(No*No*Nv*Nv);
  std::vector<double> data(el);
  std::vector<int64_t> idx(el);
  for (int64_t ii(0); ii < el; ii++){
    data[ii] = 7.3*ii;
    idx[ii] = ii;
  }
  size_t n(0);
  if (!dw.rank) n = el;
  C.write(n, idx.data(), data.data(), 0);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 9.1*ii;
  C.write(n, idx.data(), data.data(), 1);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 3.6*ii;
  C.write(n, idx.data(), data.data(), 2);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 1.2*ii;
  C.write(n, idx.data(), data.data(), 3);




  for (int64_t ii(0); ii < el; ii++) data[ii] = 0.0;
  //D.sum(1.0, C, "abij", 1.0, "ijab", C.nonZeroCondition, D.nonZeroCondition);
  D.sum(1.0, C, "abij", 1.0, "ijab", true);
 
  D.read(el, idx.data(), data.data(), 0);
  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);
  printf("=======\n");
  D.read(el, idx.data(), data.data(), 1);
  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);


}

void contractT(int64_t No, int64_t Nv){

  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t voov[] = {Nv, No, No, Nv};
  int64_t oovv[] = {No, No, Nv, Nv};
  int syms[] = {NS, NS, NS, NS};
  CTF::World dw(MPI_COMM_WORLD, 0, NULL);

  std::vector< ivec > a(27);
  for (auto &aa: a) aa.resize(4);
  a[ 0] = {0, 0, 0, 0};
  a[ 1] = {1, 0, 0, 1};
  a[ 2] = {2, 0, 0, 2};
  a[ 3] = {0, 1, 0, 1};
  a[ 4] = {1, 1, 0, 2};
  a[ 5] = {2, 1, 0, 0};
  a[ 6] = {0, 2, 0, 2};
  a[ 7] = {1, 2, 0, 0};
  a[ 8] = {2, 2, 0, 1};
  a[ 9] = {0, 0, 1, 2};
  a[10] = {1, 0, 1, 0};
  a[11] = {2, 0, 1, 1};
  a[12] = {0, 1, 1, 0};
  a[13] = {1, 1, 1, 1};
  a[14] = {2, 1, 1, 2};
  a[15] = {0, 2, 1, 1};
  a[16] = {1, 2, 1, 2};
  a[17] = {2, 2, 1, 0};
  a[18] = {0, 0, 2, 1};
  a[19] = {1, 0, 2, 2};
  a[20] = {2, 0, 2, 0};
  a[21] = {0, 1, 2, 2};
  a[22] = {1, 1, 2, 0};
  a[23] = {2, 1, 2, 1};
  a[24] = {0, 2, 2, 0};
  a[25] = {1, 2, 2, 1};
  a[26] = {2, 2, 2, 2};
 
  CTF::bsTensor<double> T(4, vvoo, syms, a, &dw, "T");
  CTF::bsTensor<double> V(4, voov, syms, a, &dw, "V");
  CTF::bsTensor<double> R(4, vvoo, syms, a, &dw, "R");
 
  int64_t el(No*No*Nv*Nv);
  std::vector<double> data(el);
  std::vector<int64_t> idx(el);
  for (int64_t ii(0); ii < el; ii++){
    data[ii] = 7.3*ii;
    idx[ii] = ii;
  }
  size_t n(0);
  if (!dw.rank) n = el;
  V.write(n, idx.data(), data.data(), 0);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 9.1*ii;
  V.write(n, idx.data(), data.data(), 1);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 3.6*ii;
  V.write(n, idx.data(), data.data(), 2);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 1.2*ii;
  V.write(n, idx.data(), data.data(), 3);

  for (int64_t ii(0); ii < el; ii++) data[ii] = 0.0;
//  T.sum(1.0, V, "abij", 1.0, "ijab", false);
 
  R.contract(1.0, V, "akic", T, "cbkj",  0.0, "abij", true);

  R.read(el, idx.data(), data.data(), 0);
//  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);
//  printf("=======\n");
//  R.read(el, idx.data(), data.data(), 1);
//  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);


}

void contractKappa(int64_t No, int64_t Nv){
  int64_t oo[] = { No, No};
  int64_t vvoo[] = {Nv, Nv, No, No};
  int64_t oovv[] = {No, No, Nv, Nv};
  int syms[] = {NS, NS, NS, NS};
  CTF::World dw(MPI_COMM_WORLD, 0, NULL);

  std::vector< ivec > a(27);
  for (auto &aa: a) aa.resize(4);
//  a[0] = {0, 0, 0, 0};
//  a[1] = {0, 0, 1, 1};
//  a[2] = {0, 1, 0, 1};
//  a[3] = {0, 1, 1, 0};
//  a[4] = {1, 0, 0, 1};
//  a[5] = {1, 0, 1, 0};
//  a[6] = {1, 1, 0, 0};
//  a[7] = {1, 1, 1, 1};
  a[ 0] = {0, 0, 0, 0};
  a[ 1] = {1, 0, 0, 1};
  a[ 2] = {2, 0, 0, 2};
  a[ 3] = {0, 1, 0, 1};
  a[ 4] = {1, 1, 0, 2};
  a[ 5] = {2, 1, 0, 0};
  a[ 6] = {0, 2, 0, 2};
  a[ 7] = {1, 2, 0, 0};
  a[ 8] = {2, 2, 0, 1};
  a[ 9] = {0, 0, 1, 2};
  a[10] = {1, 0, 1, 0};
  a[11] = {2, 0, 1, 1};
  a[12] = {0, 1, 1, 0};
  a[13] = {1, 1, 1, 1};
  a[14] = {2, 1, 1, 2};
  a[15] = {0, 2, 1, 1};
  a[16] = {1, 2, 1, 2};
  a[17] = {2, 2, 1, 0};
  a[18] = {0, 0, 2, 1};
  a[19] = {1, 0, 2, 2};
  a[20] = {2, 0, 2, 0};
  a[21] = {0, 1, 2, 2};
  a[22] = {1, 1, 2, 0};
  a[23] = {2, 1, 2, 1};
  a[24] = {0, 2, 2, 0};
  a[25] = {1, 2, 2, 1};
  a[26] = {2, 2, 2, 2};
 
  std::vector<ivec> o(3);
  for (auto &i: o) i.resize(2);
  o[0] = {0, 0};
  o[1] = {1, 1};
  o[2] = {2, 2};
  CTF::bsTensor<double> T(4, vvoo, syms, a, &dw, "T");
  CTF::bsTensor<double> W(4, oovv, syms, a, &dw, "W");
  CTF::bsTensor<double> K(2, oo, syms, o, &dw, "K");
 
  int64_t el(No*No*Nv*Nv);
  std::vector<double> data(el);
  std::vector<int64_t> idx(el);
  for (int64_t ii(0); ii < el; ii++){
    data[ii] = 7.3*ii;
    idx[ii] = ii;
  }
  size_t n(0);
  if (!dw.rank) n = el;
  W.write(n, idx.data(), data.data(), 0);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 9.1*ii;
  W.write(n, idx.data(), data.data(), 1);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 3.6*ii;
  W.write(n, idx.data(), data.data(), 2);
	for (int64_t ii(0); ii < el; ii++) data[ii] = 1.2*ii;
  W.write(n, idx.data(), data.data(), 3);

  for (int64_t ii(0); ii < el; ii++) data[ii] = 0.0;
  T.sum(1.0, W, "abij", 1.0, "ijab", false);
 
  K.contract(1.0, W, "klcd", T, "cdil", 0.0, "ki", true);

  K.read(el, idx.data(), data.data(), 0);
//  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);
//  printf("=======\n");
//  R.read(el, idx.data(), data.data(), 1);
//  for (int64_t ii(0); ii < el; ii++) printf("%f ", data[ii]);
}


/*
void matrix(int64_t No, int64_t Nv, int rank, int np, bool dryRun) {
  int64_t vvoo[] = {Nv, Nv, No, No};
  int syms[] = {NS, NS, NS, NS};

  CTF::World dw = worldkacker(dryRun, np);

  CTF::Tensor<double> R(4, vvoo, syms, dw, "A");
  CTF::Tensor<double> T(4, vvoo, syms, dw, "B");
  CTF::Tensor<double> V(4, vvoo, syms, dw, "C");

  if (!dryRun){
    int64_t n(No*No*Nv*Nv);
    double *dataA = new double[n];
    double *dataB = new double[n];
    double *dataC = new double[n];
    for (int64_t u(0UL); u < n; u++) dataA[u] = u+1;
    for (int64_t u(0UL); u < n; u++) dataB[u] = u+3;
    size_t el(0);
    if (!rank) el = n;
    std::vector<int64_t> index(el);
    if (!rank) std::iota(index.begin(), index.end(), 0);
    A.write(el, index.data(), dataA);
    if (!rank) el = n;
    index.resize(el);
    if (!rank) std::iota(index.begin(), index.end(), 0);
  }

  C["ij"] = A["ik"] * B["kj"];


}
*/

int main(int argc, char ** argv){

  int rank, np;
  int64_t No(-1), Nv(-1), Ng(-1);
  bool dryRun(false);
  for (int a(1); a < argc; a++){
    std::string arg(argv[a]);
    if (arg == "-No") No = std::atoi(argv[++a]);
    if (arg == "-Nv") Nv = std::atoi(argv[++a]);
    if (arg == "-Ng") Ng = std::atoi(argv[++a]);
    if (arg == "-d")  { dryRun = true; np = std::atoi(argv[++a]); }
  }
  if ( No < 0 || Nv < 0 ) {
    printf("gimme right format -No 14 -Nv 12 (-d 10)\n");
    return 1;
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (Ng > 0)
    transposeG(No, Nv, Ng);
  else
//    transposeT(No, Nv);
    contractT(No, Nv);
//    contractKappa(No, Nv);

  MPI_Finalize();
  return 0;
}

