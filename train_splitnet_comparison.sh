python train.py --config configs/lego.txt --featureC 45 --expname ParallelSplitNet --shadingMode ParallelSplitNet
python train.py --config configs/lego.txt --featureC 42 --expname ParallelSplitNet_Smaller --shadingMode ParallelSplitNet

python train.py --config configs/lego.txt --expname TensorfSiren --shadingMode TensorfSiren
python train.py --config configs/lego.txt --expname MLP_Fea
python train.py --config configs/lego.txt --featureC 119 --expname MLP_Fea_Smaller



# python train.py --config configs/lego_smaller.txt
# python train.py --config configs/lego_siren.txt