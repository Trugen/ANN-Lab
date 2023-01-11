StartIter=$1
SaveEvery=$2
TotalIter=$3
TotalEpoch=$[($TotalIter-$StartIter)/$SaveEvery]
SnapShots="../snapshots/ccnet_large_R2_ade20k_"

if [ $StartIter -eq 0 ]; then
    echo "Restore from pretrain model resnet101-imagenet"
    restore_from="./dataset/resnet101-imagenet.pth"
elif [ $StartIter -gt 0 ]; then
    echo "Restore from pretrain model "$SnapShots$StartIter
    restore_from=$SnapShots$StartIter
else
    echo "StartIter should be a non-negative integer."
    exit
fi

for Iter in $(seq 1 $TotalEpoch)
do
srun --gres=gpu:1 --pty python ../train.py --restore-from $restore_from --start $[$StartIter+$SaveEvery*$[$Iter-1]] --max-iter $TotalIter --random-mirror --random-scale --learning-rate 1e-2 --input-size 512 --weight-decay 1e-4 --batch-size 8 --num-steps $SaveEvery --save-pred-every $SaveEvery --recurrence 2 --model ccnet_large --dataset ade20k --num-classes 151

restore_from=$SnapShots$[$StartIter + $SaveEvery * $Iter]
echo "Last pretrain model "$restore_from
done