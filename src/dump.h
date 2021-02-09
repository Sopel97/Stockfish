#include <iostream>
#include <sstream>
#include <fstream>

#include "types.h"
#include "position.h"
#include "search.h"

// Dumping facility.  The DType indicates what type of dumption you
// want to do.  Within a dumper, you also have the name of the file
// you're dumping to.

namespace Dump {
  enum DType { Q, N, P, R, E }; // Q = dump root of qsearch tree
                                // N = dump root of qsearch tree if NNUE eval
                                // P = dump position at end of qsearch PV
                                // R = results of various evaluation functions
                                // E = dump value at end of qsearch PV

  struct Dumper : std::ostringstream {
    std::string        fname;
    DType              dtype;

    Dumper(const std::string &f = std::string()) : fname(f) { }

    operator bool() { return !fname.empty(); }

    void dump() {
      if (fname.empty()) return;
      std::ofstream outf(fname,std::ios::app);
      outf << str();
      str("");
    }
  };
}

extern Dump::Dumper dumper;
