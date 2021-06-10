for ARCH in x86-64-modern x86-64-avx2 x86-64-avx512 x86-64-vnni512
do
    for L1 in 128 256 512 1024 2048
    do
        for L2 in 16 32 64 128 256 512
        do

            echo "L1 = $L1 , L2 = $L2"
            make clean && make build ARCH=$ARCH l1=$L1 l2=$L2 -j
            ./stockfish bench 16 1 6

        done
    done
done