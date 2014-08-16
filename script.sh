base=/Users/dimitris/Repositories/matrix

t0=8000

for i in 2groups 3groups 4groups 5groups; do
    for j in 1rep 2rep 3rep 4rep 5rep; do
        for k in 1 2 3; do
            bin/pvals A B -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_AB.txt
            bin/pvals A C -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_AC.txt
            bin/pvals A D -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_AD.txt
            bin/pvals A E -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_AE.txt
            bin/pvals B C -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_BC.txt
            bin/pvals B D -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_BD.txt
            bin/pvals B E -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_BE.txt
            bin/pvals C D -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_CD.txt
            bin/pvals C E -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_CE.txt
            bin/pvals D E -t0 $t0 -i $base/$j/$i/_clust$k -o $base/$j/$i/pvals${k}_DE.txt
        done
    done
done


# base=/Users/dimitris/Repositories/matrix
#
# echo "2 groups"
# for k in 1 2 3; do
#     bin/clust $base/data/simdata$k.txt -o $base/1rep/2groups/_clust$k -t 10000 \
#         -s sample1 sample6 \
#         -g A B &
# done
#
# wait
#
# echo "3 groups"
# for k in 1 2 3; do
#     bin/clust $base/data/simdata$k.txt -o $base/1rep/3groups/_clust$k -t 10000 \
#         -s sample1 sample6 sample11 \
#         -g A B C &
# done
#
# wait
#
# echo "4 groups"
# for k in 1 2 3; do
#     bin/clust $base/data/simdata$k.txt -o $base/1rep/4groups/_clust$k -t 10000 \
#         -s sample1 sample6 sample11 sample16 \
#         -g A B C D &
# done
#
# wait
#
# echo "5 groups"
# for k in 1 2 3; do
#     bin/clust $base/data/simdata$k.txt -o $base/1rep/5groups/_clust$k -t 10000 \
#         -s sample1 sample6 sample11 sample16 sample21 \
#         -g A B C D E &
# done
