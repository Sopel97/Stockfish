#include "pawn_traits.h"
#include "index_list.h"

namespace Eval::NNUE::Features::Detail {

    // Orient a square according to perspective (flip rank for black)
    static inline Square orient(Color perspective, Square s) {
      return Square(int(s) ^ (bool(perspective) * SQ_A8));
    }

    static inline IndexType make_index(Color perspective, Square s, Color color) {
        return IndexType(orient(perspective, s) + bool(perspective ^ color) * SQUARE_NB);
    }

    void append_active_indices(
        Bitboard bbs[2],
        Color perspective,
        IndexType offset,
        IndexList* active)
    {
        for (auto color : Colors) {
            Bitboard bb = bbs[color];
            while (bb) {
              Square s = pop_lsb(&bb);
              active->push_back(offset + make_index(perspective, s, color));
            }
        }
    }

}  // namespace Eval::NNUE::Features::Detail
