# ccnet-cityscapes
srun --gres=gpu:1 --pty python evaluate.py --restore-from ./snapshots/ccnet_large_R2_cityscape_80000 --input-size 512 --recurrence 2 --model ccnet_large --dataset cityscape --num-classes 19 --ignore-label 255

# ccnet-ade20k
srun --gres=gpu:1 --pty python evaluate.py --restore-from ./snapshots/ccnet_large_R2_ade20k_80000 --input-size 512 --recurrence 2 --model ccnet_large --dataset ade20k --num-classes 151 --ignore-label 0

# van_large-cityscapes
srun --gres=gpu:1 --pty python evaluate.py --restore-from ./snapshots/van_large_R2_cityscape_80000 --input-size 512 --recurrence 2 --model van --dataset cityscape --num-classes 19 --ignore-label 255

# van_large-ade20k
srun --gres=gpu:1 --pty python evaluate.py --restore-from ./snapshots/van_large_R2_ade20k_80000 --input-size 512 --recurrence 2 --model van --dataset ade20k --num-classes 151 --ignore-label 0
