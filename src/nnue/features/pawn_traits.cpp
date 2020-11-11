#include "pawn_traits.h"
#include "index_list.h"

namespace Eval::NNUE::Features::Detail {

    void append_active_indices(
        Bitboard& bb,
        IndexType offset,
        IndexList* active)
    {
        while(bb)
        {
            Square sq = pop_lsb(&bb);
            active->push_back(offset + sq);
        }
    }

    void append_score_indices(
        Value score,
        IndexType offset,
        IndexList* active)
    {
        IndexType bucket = score == 0 ? 0 : 1 + msb(abs(score));
        active->push_back(offset + bucket);
    }

}  // namespace Eval::NNUE::Features::Detail
