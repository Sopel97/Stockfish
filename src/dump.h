/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
  enum DType { Q, T, P, R, E }; // Q = dump root of qsearch tree
                                // T = dump depth Three nodes
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
