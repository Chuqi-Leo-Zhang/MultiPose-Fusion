CUDA_VISIBLE_DEVICES=9 nohup python train_renderer.py -i output/aircraft/0.png \
                         -n aircraft-nerf \
                         -b configs/nerf.yaml \
                         -l output/renderer> running_nerf.log 2>&1 &