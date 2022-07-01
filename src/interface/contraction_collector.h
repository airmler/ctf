#ifndef __CONTRACTION_COLL_H__
#define __CONTRACTION_COLL_H__

#include "../contraction/contraction.h"

namespace CTF {
   class Contraction_collector{

    public:
      Contraction_collector();
      ~Contraction_collector();
      void analyze(int verbosity);

  };


  struct Contraction {
    CTF_int::contraction_signature sig;
    int64_t counter;
    double time;
    double estimate;
  };


}


#endif

