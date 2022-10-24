/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

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

#ifndef MOVE_TT_H_INCLUDED
#define MOVE_TT_H_INCLUDED

#include "misc.h"
#include "types.h"

namespace Stockfish {

struct MoveTTEntry {

  void save(Key k, Move m, Depth d);

  Move move() const { return (Move)move16; }

private:
  uint16_t key16;
  uint16_t move16;
  uint8_t depth8;
  uint8_t gen8;

  friend class MoveTranspositionTable;
};
static_assert(sizeof(MoveTTEntry) == 6);


/// A TranspositionTable is an array of Cluster, of size clusterCount. Each
/// cluster consists of ClusterSize number of TTEntry. Each non-empty TTEntry
/// contains information on exactly one position. The size of a Cluster should
/// divide the size of a cache line for best performance, as the cacheline is
/// prefetched when possible.

class MoveTranspositionTable {

  static constexpr int ClusterSize = 5;

  struct Cluster {
    MoveTTEntry entry[ClusterSize];
    char padding[2]; // Pad to 32 bytes
  };

  static_assert(sizeof(Cluster) == 32, "Unexpected Cluster size");

public:
  ~MoveTranspositionTable() { aligned_large_pages_free(table); }
  void new_search() { generation8 += 1; } // Lower bits are used for other things
  MoveTTEntry* probe(const Key key, bool& found) const;
  int hashfull() const;
  void resize(size_t mbSize);
  void clear();

  MoveTTEntry* first_entry(const Key key) const {
    return &table[mul_hi64(key, clusterCount)].entry[0];
  }

private:
  friend struct MoveTTEntry;

  size_t clusterCount;
  Cluster* table;
  uint8_t generation8; // Size must be not bigger than TTEntry::genBound8
};

inline Key move_tt_key(Move a, Move b)
{
  size_t lhs = a;
  size_t rhs = b;
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return (Key)lhs;
}

extern MoveTranspositionTable MoveTT;

} // namespace Stockfish

#endif // #ifndef MOVE_TT_H_INCLUDED
