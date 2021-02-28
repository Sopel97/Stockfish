/*
  Copyright (C) 2021 Matt Ginsberg

  This file is intended to be used in conjunction with Stockfish, a
  UCI chess playing engine derived from Glaurung 2.1.  It is not,
  however, part of Stockfish and is licensed separately.

  This particular file is licensed not under the Gnu Public License,
  but instead under the Creative Commons Attribution-NonCommercial
  2.0 Generic (CC BY-NC 2.0) license.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  CC BY-NC 2.0 license for more details.
*/

#include "position.h"
#include "search.h"
#include <sstream>
#include <iomanip>

using namespace std;
using namespace Search;

Depth adjust_extension(Depth, const Position &) {
  return 0;
}
