sbatch --time=0-04:00:00 --ntasks=2 --tasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 -p gpu ./Run.sh
