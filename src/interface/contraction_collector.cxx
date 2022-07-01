/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "contraction_collector.h"
#include "../contraction/contraction.h"
#include "../mapping/mapping.h"

namespace CTF {
  Contraction_collector::Contraction_collector(){};
  Contraction_collector::~Contraction_collector(){};
 
  void Contraction_collector::analyze(int verbosity){
  // we want the result in a vector
    std::vector<Contraction> contractions;
    for (auto c: CTF_int::ctr_sig_map)
      contractions.push_back(
        {c.first, c.second.counter, c.second.time, c.second.timeEstimate}
      );
    
    auto totalEstTime = std::accumulate( contractions.begin(), contractions.end()
                                       , 0.0, [](double &v, Contraction &c)
                                                { return v + c.estimate * c.counter;}
                                       );
    auto totalTime = std::accumulate(contractions.begin(), contractions.end()
                                    , 0.0, [](const double &v, Contraction &c)
                                             { return v + c.time; }
                                    );
    bool dryRun(totalTime < 1e-8);
    if (!dryRun)
      printf("Total ctf contraction times %f s; estimate %f s\n"
            , totalTime, totalEstTime);
    else
      printf("Dryrun: Total ctf contraction time estimate %f s\n"
            , totalEstTime);

    if (verbosity)
    for (auto &c: contractions){
      if (!dryRun)
        printf( "Contraction carried out %ld times; estimate %f s\n"
              , c.counter, c.estimate); 
      else 
        printf( "Contraction carried out %ld times; each took %f s, estimate %f s\n"
              , c.counter, c.time / c.counter, c.estimate); 
      c.sig.print();
      printf("--\n");
    }
  }

}
