//Definition of input feature P of NNUE evaluation function

#include "blockers_for_king.h"
#include "index_list.h"

namespace Eval::NNUE::Features {

    // Orient a square according to perspective (flip rank for black)
    inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * SQ_A8));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    IndexType BlockersForKing::make_index(
        Color perspective, Square s, Piece pc) {
        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective]);
    }

    // Get a list of indices with a value of 1 among the features
    void BlockersForKing::append_active_indices(
        const Position& pos, Color perspective, IndexList* active) {

        for (auto color : Colors) {
            Bitboard bb = pos.state()->blockersForKing[color];
            while (bb) {
                Square s = pop_lsb(&bb);
                active->push_back(make_index(perspective, s, pos.piece_on(s)));
            }
        }
    }

    // Get a list of indices whose values ​​have changed from the previous one in the feature quantity
    void BlockersForKing::append_changed_indices(
        const Position& pos, Color perspective,
        IndexList* removed, IndexList* added) {

        assert(false);
    }

}
