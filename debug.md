guillem@guillem-XPS-15:~/fragile$ uv run python -m fragile.learning.rl.train_dreamer --no-use_gas --torch_compile --diagnostics_every_updates 10 --total_epochs 2500 --collect_every 1 --updates_per_epoch 5 --eval_every 30 --eval_episodes 3 --seed_episodes 10 --checkpoint_every 100 --hidden_dim 512 --num_charts 16 --num_action_charts 16 --num_action_macros 16 --codes_per_chart 8 --latent_dim 3 --d_model 64 --w_feature_recon 2 --w_chart_ot 0.8 --w_diversity 0.8 --w_dynamics 0.2 --max_episode_steps 300 --hard_routing_warmup_epochs 5 --hard_routing_tau 1.0 --hard_routing_tau_end 1. --checkpoint_dir outputs/dreamer/theory_run --collect_n_env_workers 4 --no-torch_compile
/home/guillem/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib/python3.10/runpy.py:126: RuntimeWarning: 'fragile.learning.rl.train_dreamer' found in sys.modules after import of package 'fragile.learning.rl', but prior to execution of 'fragile.learning.rl.train_dreamer'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
Device: cuda
Environment: walker-walk  obs_dim=24  action_dim=6
Encoder:     627,306 params
Decoder:     625,345 params
Jump op:     192 params
Dyn trans:   18,200 params
World model: 92,521 params
Act enc:     26,444 params
Act dec:     27,294 params
Value field: shared in world model (45,424 params)
Reward head: 25,072 params  (chart_tok/z_embed shared)
Bound RL chart centers to encoder atlas.
Collecting 10 seed episodes...
  Seed 1/10: reward=35.2  len=301
  Seed 2/10: reward=29.6  len=301
  Seed 3/10: reward=21.6  len=301
  Seed 4/10: reward=23.0  len=301
  Seed 5/10: reward=37.3  len=301
  Seed 6/10: reward=27.6  len=301
  Seed 7/10: reward=26.8  len=301
  Seed 8/10: reward=28.2  len=301
  Seed 9/10: reward=23.4  len=301
  Seed 10/10: reward=22.1  len=301
Observation normalization: min_std=0.2067  mean_std=3.0717  max_std=12.6442

Starting training for 2500 epochs (~5 updates/epoch, buffer=3010 steps, capacity=250000)
================================================================================
E0000 [5upd]  ep_rew=7.0918  rew_20=11.5325  L_geo=0.1376  L_rew=0.0084  L_chart=2.7711  L_crit=0.8405  L_bnd=1.0778  lr=0.0010  dt=14.07s
        recon=0.9516  vq=0.0101  code_H=0.1783  code_px=1.2018  ch_usage=0.0000  rtr_mrg=0.0246  enc_gn=10.9063
        ctrl=0.0000  tex=0.0282  im_rew=0.0066  im_ret=0.0934  value=0.0011  wm_gn=0.0716
        z_norm=0.6362  z_max=0.6449  jump=0.0000  cons=1.0000  sol=0.0000  e_var=0.0005  ch_ent=0.0095  ch_act=2.0000  rtr_conf=0.0811
        obj=0.0934  dret=0.0925  term=0.0011  bnd=0.0102  chart_acc=0.7921  chart_ent=2.7726  rw_drift=0.0000
        v_err=2.4808  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4209
        bnd_x=0.0101  bell=0.0673  bell_s=0.0687  rtg_e=2.4808  rtg_b=-2.4808  cal_e=2.4808  u_l2=0.0000  cov_n=0.0000
        col=7.5925  smp=0.0031  enc_t=0.2703  bnd_t=0.0357  wm_t=0.7393  crt_t=0.0069  diag_t=0.4284
        charts: 2/16 active  c0=0.00 c1=0.00 c2=0.00 c3=0.00 c4=0.00 c5=0.00 c6=0.00 c7=0.00 c8=0.00 c9=0.00 c10=0.00 c11=0.00 c12=0.00 c13=1.00 c14=0.00 c15=0.00
        symbols: 2/128 active  c0=0/8(H=0.00) c1=0/8(H=0.00) c2=0/8(H=0.00) c3=0/8(H=0.00) c4=0/8(H=0.00) c5=0/8(H=0.00) c6=0/8(H=0.00) c7=0/8(H=0.00) c8=1/8(H=0.00) c9=0/8(H=0.00) c10=0/8(H=0.00) c11=0/8(H=0.00) c12=0/8(H=0.00) c13=1/8(H=0.00) c14=0/8(H=0.00) c15=0/8(H=0.00)
E0001 [5upd]  ep_rew=8.4052  rew_20=11.2345  L_geo=0.1275  L_rew=0.0070  L_chart=2.7651  L_crit=0.7446  L_bnd=0.9008  lr=0.0010  dt=12.63s
        recon=0.9287  vq=0.0110  code_H=0.1117  code_px=1.1184  ch_usage=0.0000  rtr_mrg=0.0020  enc_gn=5.4106
        ctrl=0.0000  tex=0.0289  im_rew=0.0209  im_ret=0.2947  value=0.0034  wm_gn=0.0719
        z_norm=0.6307  z_max=0.6379  jump=0.0000  cons=1.0000  sol=0.0000  e_var=0.0004  ch_ent=0.0000  ch_act=1.0000  rtr_conf=0.0806
        obj=0.2947  dret=0.2918  term=0.0034  bnd=0.0100  chart_acc=1.0000  chart_ent=2.7726  rw_drift=0.0000
        v_err=2.4465  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9301
        bnd_x=0.0099  bell=0.0640  bell_s=0.0689  rtg_e=2.4465  rtg_b=-2.4465  cal_e=2.4465  u_l2=0.0000  cov_n=0.0000
        col=7.2644  smp=0.0020  enc_t=0.2258  bnd_t=0.0349  wm_t=0.7122  crt_t=0.0065  diag_t=0.4141
        charts: 1/16 active  c0=0.00 c1=0.00 c2=0.00 c3=0.00 c4=0.00 c5=0.00 c6=0.00 c7=0.00 c8=0.00 c9=0.00 c10=0.00 c11=0.00 c12=0.00 c13=1.00 c14=0.00 c15=0.00
        symbols: 1/128 active  c0=0/8(H=0.00) c1=0/8(H=0.00) c2=0/8(H=0.00) c3=0/8(H=0.00) c4=0/8(H=0.00) c5=0/8(H=0.00) c6=0/8(H=0.00) c7=0/8(H=0.00) c8=0/8(H=0.00) c9=0/8(H=0.00) c10=0/8(H=0.00) c11=0/8(H=0.00) c12=0/8(H=0.00) c13=1/8(H=0.00) c14=0/8(H=0.00) c15=0/8(H=0.00)
E0002 [5upd]  ep_rew=14.0345  rew_20=11.7147  L_geo=0.1081  L_rew=0.0036  L_chart=2.7518  L_crit=0.6164  L_bnd=0.8877  lr=0.0010  dt=12.87s
        recon=0.9517  vq=0.0118  code_H=0.1191  code_px=1.1269  ch_usage=0.0000  rtr_mrg=0.0012  enc_gn=8.1184
        ctrl=0.0000  tex=0.0299  im_rew=0.0508  im_ret=0.7169  value=0.0074  wm_gn=0.0698
        z_norm=0.6146  z_max=0.6285  jump=0.0000  cons=1.0000  sol=0.0001  e_var=0.0003  ch_ent=0.0000  ch_act=1.0000  rtr_conf=0.0802
        obj=0.7169  dret=0.7106  term=0.0074  bnd=0.0089  chart_acc=1.0000  chart_ent=2.7726  rw_drift=0.0000
        v_err=1.7350  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8357
        bnd_x=0.0089  bell=0.0491  bell_s=0.0532  rtg_e=1.7350  rtg_b=-1.7350  cal_e=1.7350  u_l2=0.0000  cov_n=0.0000
        col=7.4165  smp=0.0028  enc_t=0.2283  bnd_t=0.0351  wm_t=0.7316  crt_t=0.0062  diag_t=0.3892
        charts: 1/16 active  c0=0.00 c1=0.00 c2=0.00 c3=0.00 c4=0.00 c5=0.00 c6=0.00 c7=0.00 c8=0.00 c9=0.00 c10=0.00 c11=0.00 c12=0.00 c13=1.00 c14=0.00 c15=0.00
        symbols: 1/128 active  c0=0/8(H=0.00) c1=0/8(H=0.00) c2=0/8(H=0.00) c3=0/8(H=0.00) c4=0/8(H=0.00) c5=0/8(H=0.00) c6=0/8(H=0.00) c7=0/8(H=0.00) c8=0/8(H=0.00) c9=0/8(H=0.00) c10=0/8(H=0.00) c11=0/8(H=0.00) c12=0/8(H=0.00) c13=1/8(H=0.00) c14=0/8(H=0.00) c15=0/8(H=0.00)
E0003 [5upd]  ep_rew=8.7497  rew_20=10.9666  L_geo=0.1046  L_rew=0.0033  L_chart=2.7222  L_crit=0.5053  L_bnd=0.8330  lr=0.0010  dt=12.81s
        recon=0.8202  vq=0.0125  code_H=0.1766  code_px=1.1960  ch_usage=0.0000  rtr_mrg=0.0006  enc_gn=7.4546
        ctrl=0.0000  tex=0.0308  im_rew=0.0652  im_ret=0.9227  value=0.0126  wm_gn=0.0749
        z_norm=0.6008  z_max=0.6121  jump=0.0000  cons=0.9999  sol=0.0001  e_var=0.0003  ch_ent=0.0000  ch_act=1.0000  rtr_conf=0.0807
        obj=0.9227  dret=0.9119  term=0.0126  bnd=0.0119  chart_acc=1.0000  chart_ent=2.7725  rw_drift=0.0000
        v_err=1.8985  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8866
        bnd_x=0.0118  bell=0.0552  bell_s=0.0673  rtg_e=1.8985  rtg_b=-1.8985  cal_e=1.8985  u_l2=0.0000  cov_n=0.0000
        col=7.4575  smp=0.0032  enc_t=0.2264  bnd_t=0.0348  wm_t=0.7153  crt_t=0.0062  diag_t=0.3848
        charts: 1/16 active  c0=0.00 c1=0.00 c2=0.00 c3=0.00 c4=0.00 c5=0.00 c6=0.00 c7=0.00 c8=0.00 c9=0.00 c10=0.00 c11=0.00 c12=0.00 c13=1.00 c14=0.00 c15=0.00
        symbols: 2/128 active  c0=0/8(H=0.00) c1=0/8(H=0.00) c2=0/8(H=0.00) c3=0/8(H=0.00) c4=0/8(H=0.00) c5=0/8(H=0.00) c6=0/8(H=0.00) c7=0/8(H=0.00) c8=0/8(H=0.00) c9=0/8(H=0.00) c10=0/8(H=0.00) c11=0/8(H=0.00) c12=0/8(H=0.00) c13=2/8(H=0.07) c14=0/8(H=0.00) c15=0/8(H=0.00)
E0004 [5upd]  ep_rew=15.4963  rew_20=11.9851  L_geo=0.1226  L_rew=0.0028  L_chart=2.6633  L_crit=0.5429  L_bnd=0.8108  lr=0.0010  dt=13.45s
        recon=0.9075  vq=0.0131  code_H=0.1290  code_px=1.1382  ch_usage=0.0000  rtr_mrg=0.0002  enc_gn=7.2738
        ctrl=0.0000  tex=0.0323  im_rew=0.0494  im_ret=0.7091  value=0.0199  wm_gn=0.0919
        z_norm=0.5758  z_max=0.5834  jump=0.0000  cons=0.9999  sol=0.0002  e_var=0.0004  ch_ent=0.0000  ch_act=1.0000  rtr_conf=0.0812
        obj=0.7091  dret=0.6920  term=0.0199  bnd=0.0248  chart_acc=1.0000  chart_ent=2.7721  rw_drift=0.0000
        v_err=1.5441  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9627
        bnd_x=0.0242  bell=0.0412  bell_s=0.0430  rtg_e=1.5441  rtg_b=-1.5441  cal_e=1.5441  u_l2=0.0000  cov_n=0.0000
        col=7.6164  smp=0.0026  enc_t=0.2316  bnd_t=0.0352  wm_t=0.7992  crt_t=0.0064  diag_t=0.4150
        charts: 1/16 active  c0=0.00 c1=0.00 c2=0.00 c3=0.00 c4=0.00 c5=0.00 c6=0.00 c7=0.00 c8=0.00 c9=0.00 c10=0.00 c11=0.00 c12=0.00 c13=1.00 c14=0.00 c15=0.00
        symbols: 1/128 active  c0=0/8(H=0.00) c1=0/8(H=0.00) c2=0/8(H=0.00) c3=0/8(H=0.00) c4=0/8(H=0.00) c5=0/8(H=0.00) c6=0/8(H=0.00) c7=0/8(H=0.00) c8=0/8(H=0.00) c9=0/8(H=0.00) c10=0/8(H=0.00) c11=0/8(H=0.00) c12=0/8(H=0.00) c13=1/8(H=0.00) c14=0/8(H=0.00) c15=0/8(H=0.00)
E0005 [5upd]  ep_rew=6.2430  rew_20=11.9154  L_geo=1.7047  L_rew=0.0014  L_chart=2.7719  L_crit=0.3490  L_bnd=0.7928  lr=0.0010  dt=13.19s
        recon=0.8199  vq=0.0642  code_H=2.0585  code_px=7.8345  ch_usage=7.1141  rtr_mrg=0.0048  enc_gn=23.6967
        ctrl=0.0002  tex=0.0284  im_rew=0.0352  im_ret=0.5148  value=0.0266  wm_gn=0.1058
        z_norm=0.6282  z_max=0.7587  jump=0.0000  cons=0.9999  sol=0.0002  e_var=0.0010  ch_ent=2.7632  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5148  dret=0.4920  term=0.0265  bnd=0.0463  chart_acc=0.0754  chart_ent=2.7704  rw_drift=0.0000
        v_err=1.2784  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8873
        bnd_x=0.0442  bell=0.0346  bell_s=0.0360  rtg_e=1.2784  rtg_b=-1.2783  cal_e=1.2783  u_l2=0.0000  cov_n=0.0019
        col=7.6339  smp=0.0020  enc_t=0.2352  bnd_t=0.0348  wm_t=0.7432  crt_t=0.0062  diag_t=0.4067
        charts: 16/16 active  c0=0.06 c1=0.05 c2=0.06 c3=0.06 c4=0.05 c5=0.07 c6=0.06 c7=0.06 c8=0.08 c9=0.06 c10=0.07 c11=0.07 c12=0.07 c13=0.08 c14=0.06 c15=0.06
        symbols: 16/128 active  c0=1/8(H=0.00) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=1/8(H=0.00) c4=1/8(H=0.00) c5=1/8(H=0.00) c6=1/8(H=0.00) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=1/8(H=0.00) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=1/8(H=0.00) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0006 [5upd]  ep_rew=10.0538  rew_20=12.3631  L_geo=1.6997  L_rew=0.0024  L_chart=2.7677  L_crit=0.4143  L_bnd=0.7781  lr=0.0010  dt=13.27s
        recon=0.8471  vq=0.0640  code_H=1.9332  code_px=7.1024  ch_usage=7.0646  rtr_mrg=0.0337  enc_gn=22.4596
        ctrl=0.0001  tex=0.0292  im_rew=0.0308  im_ret=0.4557  value=0.0290  wm_gn=0.1175
        z_norm=0.6181  z_max=0.7574  jump=0.0000  cons=0.9998  sol=0.0004  e_var=0.0011  ch_ent=2.7691  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4557  dret=0.4310  term=0.0288  bnd=0.0574  chart_acc=0.0854  chart_ent=2.7674  rw_drift=0.0000
        v_err=1.4993  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9212
        bnd_x=0.0542  bell=0.0445  bell_s=0.0625  rtg_e=1.4993  rtg_b=-1.4992  cal_e=1.4992  u_l2=0.0000  cov_n=0.0016
        col=7.7193  smp=0.0026  enc_t=0.2290  bnd_t=0.0352  wm_t=0.7500  crt_t=0.0067  diag_t=0.3888
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.06 c3=0.06 c4=0.06 c5=0.06 c6=0.07 c7=0.06 c8=0.07 c9=0.05 c10=0.06 c11=0.06 c12=0.06 c13=0.07 c14=0.06 c15=0.07
        symbols: 18/128 active  c0=1/8(H=0.00) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=1/8(H=0.00) c4=1/8(H=0.00) c5=2/8(H=0.69) c6=2/8(H=0.68) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=1/8(H=0.00) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=1/8(H=0.00) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0007 [5upd]  ep_rew=14.5827  rew_20=11.9816  L_geo=1.6527  L_rew=0.0029  L_chart=2.7793  L_crit=0.4675  L_bnd=0.8208  lr=0.0010  dt=13.32s
        recon=0.9144  vq=0.0653  code_H=1.9441  code_px=7.0046  ch_usage=3.6721  rtr_mrg=0.0739  enc_gn=17.3335
        ctrl=0.0001  tex=0.0307  im_rew=0.0358  im_ret=0.5265  value=0.0297  wm_gn=0.1452
        z_norm=0.5954  z_max=0.7560  jump=0.0000  cons=0.9998  sol=0.0007  e_var=0.0009  ch_ent=2.7677  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5265  dret=0.5007  term=0.0300  bnd=0.0516  chart_acc=0.0650  chart_ent=2.7637  rw_drift=0.0000
        v_err=1.6576  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8522
        bnd_x=0.0490  bell=0.0473  bell_s=0.0561  rtg_e=1.6576  rtg_b=-1.6575  cal_e=1.6575  u_l2=0.0000  cov_n=0.0016
        col=7.6219  smp=0.0026  enc_t=0.2292  bnd_t=0.0349  wm_t=0.7767  crt_t=0.0062  diag_t=0.4088
        charts: 16/16 active  c0=0.06 c1=0.08 c2=0.06 c3=0.06 c4=0.06 c5=0.06 c6=0.06 c7=0.07 c8=0.06 c9=0.07 c10=0.06 c11=0.06 c12=0.06 c13=0.07 c14=0.06 c15=0.06
        symbols: 22/128 active  c0=1/8(H=0.00) c1=2/8(H=0.03) c2=1/8(H=0.00) c3=2/8(H=0.65) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.68) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.67) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0008 [5upd]  ep_rew=10.1561  rew_20=12.8991  L_geo=1.5316  L_rew=0.0021  L_chart=2.7866  L_crit=0.4455  L_bnd=0.8318  lr=0.0010  dt=13.21s
        recon=0.8904  vq=0.0683  code_H=1.5290  code_px=4.7864  ch_usage=3.5015  rtr_mrg=0.0531  enc_gn=22.7671
        ctrl=0.0001  tex=0.0306  im_rew=0.0460  im_ret=0.6696  value=0.0290  wm_gn=0.1165
        z_norm=0.5884  z_max=0.7548  jump=0.0000  cons=0.9994  sol=0.0017  e_var=0.0011  ch_ent=2.7664  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6696  dret=0.6443  term=0.0293  bnd=0.0391  chart_acc=0.0508  chart_ent=2.7634  rw_drift=0.0000
        v_err=1.8257  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7907
        bnd_x=0.0376  bell=0.0518  bell_s=0.0684  rtg_e=1.8257  rtg_b=-1.8256  cal_e=1.8256  u_l2=0.0000  cov_n=0.0012
        col=7.6331  smp=0.0027  enc_t=0.2300  bnd_t=0.0355  wm_t=0.7525  crt_t=0.0064  diag_t=0.4062
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.06 c3=0.06 c4=0.08 c5=0.06 c6=0.06 c7=0.06 c8=0.06 c9=0.07 c10=0.06 c11=0.06 c12=0.06 c13=0.06 c14=0.05 c15=0.06
        symbols: 23/128 active  c0=3/8(H=0.62) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.66) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.68) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0009 [5upd]  ep_rew=10.2529  rew_20=12.6720  L_geo=1.5624  L_rew=0.0018  L_chart=2.7838  L_crit=0.4018  L_bnd=0.8236  lr=0.0010  dt=13.24s
        recon=0.8517  vq=0.0661  code_H=1.4282  code_px=4.5139  ch_usage=3.7442  rtr_mrg=0.0470  enc_gn=14.6540
        ctrl=0.0001  tex=0.0323  im_rew=0.0492  im_ret=0.7116  value=0.0287  wm_gn=0.1051
        z_norm=0.5991  z_max=0.7535  jump=0.0000  cons=0.9989  sol=0.0032  e_var=0.0006  ch_ent=2.7689  ch_act=16.0000  rtr_conf=1.0000
        obj=0.7116  dret=0.6878  term=0.0277  bnd=0.0347  chart_acc=0.0525  chart_ent=2.7650  rw_drift=0.0000
        v_err=1.5433  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6973
        bnd_x=0.0334  bell=0.0417  bell_s=0.0397  rtg_e=1.5433  rtg_b=-1.5432  cal_e=1.5432  u_l2=0.0000  cov_n=0.0011
        col=7.5997  smp=0.0028  enc_t=0.2280  bnd_t=0.0347  wm_t=0.7630  crt_t=0.0062  diag_t=0.4278
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.06 c3=0.06 c4=0.07 c5=0.07 c6=0.06 c7=0.06 c8=0.05 c9=0.07 c10=0.07 c11=0.07 c12=0.06 c13=0.06 c14=0.06 c15=0.07
        symbols: 22/128 active  c0=2/8(H=0.20) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.45) c4=2/8(H=0.66) c5=1/8(H=0.00) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.62) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.61) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0010 [5upd]  ep_rew=22.0331  rew_20=13.4454  L_geo=1.5675  L_rew=0.0023  L_chart=2.7798  L_crit=0.4346  L_bnd=0.8274  lr=0.0010  dt=13.38s
        recon=0.8782  vq=0.0646  code_H=1.5696  code_px=4.8596  ch_usage=3.3036  rtr_mrg=0.0503  enc_gn=19.3980
        ctrl=0.0001  tex=0.0329  im_rew=0.0428  im_ret=0.6225  value=0.0282  wm_gn=0.1104
        z_norm=0.5861  z_max=0.7521  jump=0.0000  cons=0.9973  sol=0.0064  e_var=0.0008  ch_ent=2.7698  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6225  dret=0.5990  term=0.0273  bnd=0.0392  chart_acc=0.0554  chart_ent=2.7671  rw_drift=0.0000
        v_err=1.8579  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6928
        bnd_x=0.0375  bell=0.0543  bell_s=0.0708  rtg_e=1.8579  rtg_b=-1.8578  cal_e=1.8578  u_l2=0.0001  cov_n=0.0011
        col=7.8431  smp=0.0011  enc_t=0.2285  bnd_t=0.0351  wm_t=0.7479  crt_t=0.0060  diag_t=0.3982
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.07 c3=0.06 c4=0.06 c5=0.06 c6=0.07 c7=0.06 c8=0.06 c9=0.06 c10=0.06 c11=0.07 c12=0.06 c13=0.05 c14=0.06 c15=0.06
        symbols: 22/128 active  c0=2/8(H=0.12) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.59) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.68) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0011 [5upd]  ep_rew=9.0159  rew_20=13.3270  L_geo=1.4966  L_rew=0.0017  L_chart=2.7780  L_crit=0.4193  L_bnd=0.7972  lr=0.0010  dt=13.43s
        recon=0.8820  vq=0.0649  code_H=1.7223  code_px=5.7448  ch_usage=3.5063  rtr_mrg=0.0406  enc_gn=21.8014
        ctrl=0.0001  tex=0.0331  im_rew=0.0408  im_ret=0.5935  value=0.0277  wm_gn=0.1010
        z_norm=0.5637  z_max=0.7506  jump=0.0000  cons=0.9971  sol=0.0108  e_var=0.0008  ch_ent=2.7677  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5935  dret=0.5705  term=0.0267  bnd=0.0402  chart_acc=0.0538  chart_ent=2.7689  rw_drift=0.0000
        v_err=1.5076  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7743
        bnd_x=0.0384  bell=0.0413  bell_s=0.0338  rtg_e=1.5076  rtg_b=-1.5075  cal_e=1.5075  u_l2=0.0001  cov_n=0.0008
        col=7.7585  smp=0.0032  enc_t=0.2318  bnd_t=0.0349  wm_t=0.7656  crt_t=0.0065  diag_t=0.4264
        charts: 16/16 active  c0=0.07 c1=0.07 c2=0.06 c3=0.07 c4=0.06 c5=0.06 c6=0.07 c7=0.06 c8=0.06 c9=0.07 c10=0.06 c11=0.06 c12=0.05 c13=0.05 c14=0.06 c15=0.06
        symbols: 22/128 active  c0=2/8(H=0.60) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.68) c4=2/8(H=0.65) c5=1/8(H=0.00) c6=2/8(H=0.60) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.63) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.65) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0012 [5upd]  ep_rew=13.1816  rew_20=13.2511  L_geo=1.5611  L_rew=0.0023  L_chart=2.7753  L_crit=0.4147  L_bnd=0.8160  lr=0.0010  dt=13.34s
        recon=0.8161  vq=0.0636  code_H=1.8517  code_px=6.4846  ch_usage=3.3848  rtr_mrg=0.0424  enc_gn=14.9061
        ctrl=0.0001  tex=0.0313  im_rew=0.0425  im_ret=0.6190  value=0.0284  wm_gn=0.0898
        z_norm=0.5762  z_max=0.7492  jump=0.0000  cons=0.9917  sol=0.0222  e_var=0.0008  ch_ent=2.7688  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6190  dret=0.5941  term=0.0290  bnd=0.0420  chart_acc=0.0617  chart_ent=2.7697  rw_drift=0.0000
        v_err=1.3902  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7144
        bnd_x=0.0401  bell=0.0387  bell_s=0.0316  rtg_e=1.3902  rtg_b=-1.3901  cal_e=1.3901  u_l2=0.0001  cov_n=0.0009
        col=7.6928  smp=0.0025  enc_t=0.2291  bnd_t=0.0350  wm_t=0.7596  crt_t=0.0064  diag_t=0.4473
        charts: 16/16 active  c0=0.06 c1=0.08 c2=0.06 c3=0.07 c4=0.06 c5=0.06 c6=0.07 c7=0.06 c8=0.06 c9=0.06 c10=0.06 c11=0.05 c12=0.06 c13=0.06 c14=0.06 c15=0.06
        symbols: 22/128 active  c0=2/8(H=0.54) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.68) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0013 [5upd]  ep_rew=13.5603  rew_20=12.4399  L_geo=1.5427  L_rew=0.0019  L_chart=2.7762  L_crit=0.3426  L_bnd=0.8595  lr=0.0010  dt=13.51s
        recon=0.7644  vq=0.0631  code_H=1.4452  code_px=4.3333  ch_usage=3.8267  rtr_mrg=0.0398  enc_gn=15.3486
        ctrl=0.0001  tex=0.0305  im_rew=0.0420  im_ret=0.6138  value=0.0297  wm_gn=0.0748
        z_norm=0.5843  z_max=0.7478  jump=0.0000  cons=0.9903  sol=0.0317  e_var=0.0006  ch_ent=2.7704  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6138  dret=0.5883  term=0.0297  bnd=0.0434  chart_acc=0.0567  chart_ent=2.7704  rw_drift=0.0000
        v_err=1.2401  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5237
        bnd_x=0.0412  bell=0.0359  bell_s=0.0356  rtg_e=1.2401  rtg_b=-1.2400  cal_e=1.2400  u_l2=0.0001  cov_n=0.0007
        col=7.8181  smp=0.0028  enc_t=0.2297  bnd_t=0.0350  wm_t=0.7672  crt_t=0.0067  diag_t=0.4425
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.06 c3=0.06 c4=0.07 c5=0.06 c6=0.06 c7=0.06 c8=0.06 c9=0.06 c10=0.06 c11=0.05 c12=0.06 c13=0.07 c14=0.07 c15=0.07
        symbols: 23/128 active  c0=3/8(H=0.58) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.58) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.66) c13=1/8(H=0.00) c14=1/8(H=0.00) c15=1/8(H=0.00)
E0014 [5upd]  ep_rew=35.4125  rew_20=13.2268  L_geo=1.5408  L_rew=0.0026  L_chart=2.7734  L_crit=0.3911  L_bnd=0.9348  lr=0.0010  dt=13.44s
        recon=0.7781  vq=0.0633  code_H=1.2935  code_px=3.8250  ch_usage=3.6660  rtr_mrg=0.0357  enc_gn=17.7725
        ctrl=0.0001  tex=0.0318  im_rew=0.0406  im_ret=0.5942  value=0.0312  wm_gn=0.0739
        z_norm=0.5713  z_max=0.7464  jump=0.0000  cons=0.9776  sol=0.0572  e_var=0.0006  ch_ent=2.7701  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5942  dret=0.5678  term=0.0306  bnd=0.0464  chart_acc=0.0688  chart_ent=2.7708  rw_drift=0.0000
        v_err=1.6225  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2386
        bnd_x=0.0438  bell=0.0447  bell_s=0.0405  rtg_e=1.6225  rtg_b=-1.6224  cal_e=1.6224  u_l2=0.0001  cov_n=0.0008
        col=7.7054  smp=0.0032  enc_t=0.2282  bnd_t=0.0347  wm_t=0.7864  crt_t=0.0063  diag_t=0.4070
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.06 c3=0.06 c4=0.07 c5=0.06 c6=0.07 c7=0.05 c8=0.06 c9=0.06 c10=0.06 c11=0.06 c12=0.07 c13=0.06 c14=0.06 c15=0.06
        symbols: 23/128 active  c0=2/8(H=0.52) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.68) c5=1/8(H=0.00) c6=2/8(H=0.66) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=1/8(H=0.00) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=1/8(H=0.00) c14=2/8(H=0.03) c15=1/8(H=0.00)
E0015 [5upd]  ep_rew=9.6441  rew_20=12.6931  L_geo=1.4909  L_rew=0.0010  L_chart=2.7734  L_crit=0.3329  L_bnd=0.9534  lr=0.0010  dt=13.32s
        recon=0.7743  vq=0.0629  code_H=1.6072  code_px=5.1692  ch_usage=3.1915  rtr_mrg=0.0284  enc_gn=16.1889
        ctrl=0.0001  tex=0.0315  im_rew=0.0381  im_ret=0.5610  value=0.0329  wm_gn=0.0828
        z_norm=0.5794  z_max=0.7747  jump=0.0000  cons=0.9648  sol=0.1023  e_var=0.0010  ch_ent=2.7705  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5610  dret=0.5335  term=0.0319  bnd=0.0515  chart_acc=0.0633  chart_ent=2.7710  rw_drift=0.0000
        v_err=1.6510  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1376
        bnd_x=0.0485  bell=0.0461  bell_s=0.0409  rtg_e=1.6510  rtg_b=-1.6509  cal_e=1.6509  u_l2=0.0001  cov_n=0.0009
        col=7.7254  smp=0.0013  enc_t=0.2281  bnd_t=0.0352  wm_t=0.7497  crt_t=0.0064  diag_t=0.4459
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.06 c3=0.06 c4=0.07 c5=0.07 c6=0.07 c7=0.06 c8=0.06 c9=0.06 c10=0.06 c11=0.06 c12=0.06 c13=0.07 c14=0.06 c15=0.06
        symbols: 24/128 active  c0=2/8(H=0.63) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.65) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.03) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=1/8(H=0.00) c14=2/8(H=0.42) c15=1/8(H=0.00)
E0016 [5upd]  ep_rew=9.4009  rew_20=12.0267  L_geo=1.5426  L_rew=0.0014  L_chart=2.7670  L_crit=0.3081  L_bnd=0.9234  lr=0.0010  dt=18.86s
        recon=0.7278  vq=0.0617  code_H=1.8305  code_px=6.2898  ch_usage=2.8471  rtr_mrg=0.0225  enc_gn=16.6132
        ctrl=0.0001  tex=0.0281  im_rew=0.0363  im_ret=0.5406  value=0.0345  wm_gn=0.0983
        z_norm=0.5914  z_max=0.7739  jump=0.0000  cons=0.9458  sol=0.1118  e_var=0.0007  ch_ent=2.7663  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5406  dret=0.5075  term=0.0385  bnd=0.0652  chart_acc=0.0996  chart_ent=2.7712  rw_drift=0.0000
        v_err=1.1969  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2636
        bnd_x=0.0607  bell=0.0344  bell_s=0.0313  rtg_e=1.1969  rtg_b=-1.1967  cal_e=1.1967  u_l2=0.0001  cov_n=0.0019
        col=10.9355  smp=0.0037  enc_t=0.2638  bnd_t=0.0427  wm_t=1.1304  crt_t=0.0096  diag_t=0.6267
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.06 c3=0.06 c4=0.06 c5=0.06 c6=0.06 c7=0.06 c8=0.07 c9=0.06 c10=0.06 c11=0.06 c12=0.05 c13=0.08 c14=0.06 c15=0.08
        symbols: 26/128 active  c0=2/8(H=0.42) c1=2/8(H=0.03) c2=1/8(H=0.00) c3=2/8(H=0.64) c4=2/8(H=0.69) c5=2/8(H=0.03) c6=2/8(H=0.69) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.58) c11=1/8(H=0.00) c12=2/8(H=0.68) c13=1/8(H=0.00) c14=2/8(H=0.69) c15=1/8(H=0.00)
E0017 [5upd]  ep_rew=12.3538  rew_20=12.3337  L_geo=1.4884  L_rew=0.0015  L_chart=2.7692  L_crit=0.3706  L_bnd=0.8712  lr=0.0010  dt=19.31s
        recon=0.7330  vq=0.0642  code_H=1.4916  code_px=4.5897  ch_usage=3.0094  rtr_mrg=0.0164  enc_gn=18.2255
        ctrl=0.0001  tex=0.0339  im_rew=0.0352  im_ret=0.5193  value=0.0346  wm_gn=0.0996
        z_norm=0.5830  z_max=0.7727  jump=0.0000  cons=0.9039  sol=0.1875  e_var=0.0007  ch_ent=2.7630  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5193  dret=0.4924  term=0.0313  bnd=0.0546  chart_acc=0.0842  chart_ent=2.7712  rw_drift=0.0000
        v_err=1.4665  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4215
        bnd_x=0.0513  bell=0.0399  bell_s=0.0366  rtg_e=1.4665  rtg_b=-1.4663  cal_e=1.4663  u_l2=0.0001  cov_n=0.0011
        col=11.7542  smp=0.0037  enc_t=0.2571  bnd_t=0.0407  wm_t=1.0708  crt_t=0.0090  diag_t=0.6009
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.06 c3=0.05 c4=0.07 c5=0.06 c6=0.05 c7=0.06 c8=0.06 c9=0.06 c10=0.07 c11=0.05 c12=0.06 c13=0.09 c14=0.05 c15=0.08
        symbols: 27/128 active  c0=2/8(H=0.63) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.67) c5=2/8(H=0.08) c6=2/8(H=0.68) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.68) c10=2/8(H=0.62) c11=1/8(H=0.00) c12=2/8(H=0.67) c13=2/8(H=0.09) c14=2/8(H=0.68) c15=2/8(H=0.13)
E0018 [5upd]  ep_rew=11.1157  rew_20=12.8333  L_geo=1.4808  L_rew=0.0011  L_chart=2.7690  L_crit=0.3428  L_bnd=0.9020  lr=0.0010  dt=18.92s
        recon=0.7120  vq=0.0635  code_H=1.4868  code_px=4.4796  ch_usage=2.8208  rtr_mrg=0.0218  enc_gn=19.6278
        ctrl=0.0001  tex=0.0335  im_rew=0.0367  im_ret=0.5404  value=0.0344  wm_gn=0.0868
        z_norm=0.5879  z_max=0.7713  jump=0.0000  cons=0.8952  sol=0.2486  e_var=0.0007  ch_ent=2.7655  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5404  dret=0.5129  term=0.0320  bnd=0.0537  chart_acc=0.0858  chart_ent=2.7712  rw_drift=0.0000
        v_err=1.2063  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2967
        bnd_x=0.0501  bell=0.0341  bell_s=0.0274  rtg_e=1.2063  rtg_b=-1.2061  cal_e=1.2061  u_l2=0.0001  cov_n=0.0008
        col=11.5023  smp=0.0035  enc_t=0.2549  bnd_t=0.0403  wm_t=1.0502  crt_t=0.0088  diag_t=0.5804
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.06 c3=0.05 c4=0.06 c5=0.07 c6=0.07 c7=0.07 c8=0.06 c9=0.05 c10=0.07 c11=0.06 c12=0.06 c13=0.08 c14=0.06 c15=0.07
        symbols: 26/128 active  c0=2/8(H=0.49) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.61) c4=2/8(H=0.67) c5=1/8(H=0.00) c6=2/8(H=0.62) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.69) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=2/8(H=0.20) c14=2/8(H=0.69) c15=2/8(H=0.26)
E0019 [5upd]  ep_rew=15.0472  rew_20=12.0879  L_geo=1.4544  L_rew=0.0015  L_chart=2.7702  L_crit=0.3810  L_bnd=0.9831  lr=0.0010  dt=19.01s
        recon=0.7429  vq=0.0643  code_H=1.5366  code_px=4.9354  ch_usage=3.1527  rtr_mrg=0.0233  enc_gn=18.4441
        ctrl=0.0001  tex=0.0310  im_rew=0.0372  im_ret=0.5504  value=0.0340  wm_gn=0.0998
        z_norm=0.5821  z_max=0.7697  jump=0.0000  cons=0.8303  sol=0.3019  e_var=0.0008  ch_ent=2.7651  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5504  dret=0.5210  term=0.0341  bnd=0.0564  chart_acc=0.0783  chart_ent=2.7711  rw_drift=0.0000
        v_err=1.2840  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0294
        bnd_x=0.0526  bell=0.0355  bell_s=0.0283  rtg_e=1.2840  rtg_b=-1.2838  cal_e=1.2838  u_l2=0.0002  cov_n=0.0007
        col=11.4244  smp=0.0043  enc_t=0.2570  bnd_t=0.0404  wm_t=1.0806  crt_t=0.0089  diag_t=0.5802
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.05 c3=0.05 c4=0.06 c5=0.06 c6=0.06 c7=0.06 c8=0.06 c9=0.06 c10=0.08 c11=0.06 c12=0.06 c13=0.07 c14=0.06 c15=0.07
        symbols: 28/128 active  c0=2/8(H=0.63) c1=3/8(H=0.19) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.67) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.66) c11=1/8(H=0.00) c12=2/8(H=0.69) c13=2/8(H=0.62) c14=2/8(H=0.69) c15=2/8(H=0.05)
E0020 [5upd]  ep_rew=14.2788  rew_20=11.8071  L_geo=1.4772  L_rew=0.0011  L_chart=2.7683  L_crit=0.3174  L_bnd=1.0290  lr=0.0010  dt=18.65s
        recon=0.7014  vq=0.0645  code_H=1.4010  code_px=4.3255  ch_usage=3.4964  rtr_mrg=0.0273  enc_gn=14.4060
        ctrl=0.0001  tex=0.0300  im_rew=0.0359  im_ret=0.5341  value=0.0346  wm_gn=0.0793
        z_norm=0.5866  z_max=0.7683  jump=0.0000  cons=0.8025  sol=0.3315  e_var=0.0007  ch_ent=2.7602  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5341  dret=0.5022  term=0.0370  bnd=0.0634  chart_acc=0.0854  chart_ent=2.7709  rw_drift=0.0000
        v_err=1.4836  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2625
        bnd_x=0.0586  bell=0.0429  bell_s=0.0462  rtg_e=1.4836  rtg_b=-1.4834  cal_e=1.4834  u_l2=0.0002  cov_n=0.0008
        col=11.3005  smp=0.0044  enc_t=0.2547  bnd_t=0.0402  wm_t=1.0378  crt_t=0.0088  diag_t=0.5808
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.05 c3=0.05 c4=0.05 c5=0.07 c6=0.07 c7=0.06 c8=0.06 c9=0.06 c10=0.08 c11=0.06 c12=0.05 c13=0.08 c14=0.06 c15=0.08
        symbols: 26/128 active  c0=2/8(H=0.60) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.68) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.61) c11=1/8(H=0.00) c12=2/8(H=0.68) c13=2/8(H=0.55) c14=2/8(H=0.64) c15=2/8(H=0.37)
E0021 [5upd]  ep_rew=8.2085  rew_20=11.9037  L_geo=1.4733  L_rew=0.0011  L_chart=2.7693  L_crit=0.3544  L_bnd=1.0124  lr=0.0010  dt=18.70s
        recon=0.7218  vq=0.0621  code_H=1.7570  code_px=5.8718  ch_usage=2.2806  rtr_mrg=0.0463  enc_gn=17.1009
        ctrl=0.0001  tex=0.0301  im_rew=0.0365  im_ret=0.5446  value=0.0365  wm_gn=0.0768
        z_norm=0.5761  z_max=0.7669  jump=0.0000  cons=0.7998  sol=0.4129  e_var=0.0008  ch_ent=2.7645  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5446  dret=0.5114  term=0.0386  bnd=0.0650  chart_acc=0.0833  chart_ent=2.7709  rw_drift=0.0000
        v_err=1.3171  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1321
        bnd_x=0.0600  bell=0.0371  bell_s=0.0340  rtg_e=1.3171  rtg_b=-1.3168  cal_e=1.3168  u_l2=0.0002  cov_n=0.0008
        col=11.3097  smp=0.0044  enc_t=0.2581  bnd_t=0.0397  wm_t=1.0418  crt_t=0.0089  diag_t=0.5842
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.05 c3=0.06 c4=0.06 c5=0.07 c6=0.06 c7=0.06 c8=0.06 c9=0.06 c10=0.08 c11=0.06 c12=0.06 c13=0.08 c14=0.05 c15=0.08
        symbols: 27/128 active  c0=2/8(H=0.59) c1=1/8(H=0.00) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.67) c5=1/8(H=0.00) c6=2/8(H=0.62) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.63) c10=2/8(H=0.68) c11=2/8(H=0.08) c12=2/8(H=0.03) c13=2/8(H=0.64) c14=2/8(H=0.69) c15=2/8(H=0.22)
E0022 [5upd]  ep_rew=7.5084  rew_20=11.2903  L_geo=1.4514  L_rew=0.0015  L_chart=2.7681  L_crit=0.3461  L_bnd=0.8570  lr=0.0010  dt=18.61s
        recon=0.6978  vq=0.0631  code_H=1.6850  code_px=5.5159  ch_usage=2.9642  rtr_mrg=0.0304  enc_gn=12.6112
        ctrl=0.0001  tex=0.0305  im_rew=0.0360  im_ret=0.5393  value=0.0388  wm_gn=0.0720
        z_norm=0.5688  z_max=0.7657  jump=0.0000  cons=0.6951  sol=0.5351  e_var=0.0008  ch_ent=2.7662  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5393  dret=0.5039  term=0.0412  bnd=0.0704  chart_acc=0.0871  chart_ent=2.7707  rw_drift=0.0000
        v_err=1.4106  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4546
        bnd_x=0.0649  bell=0.0426  bell_s=0.0483  rtg_e=1.4106  rtg_b=-1.4103  cal_e=1.4103  u_l2=0.0003  cov_n=0.0012
        col=11.2719  smp=0.0027  enc_t=0.2538  bnd_t=0.0402  wm_t=1.0342  crt_t=0.0087  diag_t=0.5850
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.06 c3=0.06 c4=0.05 c5=0.07 c6=0.06 c7=0.05 c8=0.06 c9=0.07 c10=0.07 c11=0.06 c12=0.06 c13=0.08 c14=0.06 c15=0.07
        symbols: 27/128 active  c0=2/8(H=0.67) c1=2/8(H=0.36) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=2/8(H=0.67) c5=1/8(H=0.00) c6=2/8(H=0.65) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.65) c10=2/8(H=0.68) c11=2/8(H=0.40) c12=1/8(H=0.00) c13=2/8(H=0.67) c14=2/8(H=0.69) c15=2/8(H=0.36)
E0023 [5upd]  ep_rew=14.7799  rew_20=11.4938  L_geo=1.4053  L_rew=0.0010  L_chart=2.7678  L_crit=0.2771  L_bnd=0.8114  lr=0.0010  dt=18.49s
        recon=0.6772  vq=0.0642  code_H=1.6106  code_px=5.1986  ch_usage=1.9895  rtr_mrg=0.0251  enc_gn=15.0875
        ctrl=0.0001  tex=0.0330  im_rew=0.0354  im_ret=0.5316  value=0.0432  wm_gn=0.1128
        z_norm=0.5897  z_max=0.8128  jump=0.0000  cons=0.5838  sol=0.6099  e_var=0.0007  ch_ent=2.7502  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5316  dret=0.4957  term=0.0417  bnd=0.0724  chart_acc=0.0821  chart_ent=2.7705  rw_drift=0.0000
        v_err=1.2595  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5668
        bnd_x=0.0663  bell=0.0372  bell_s=0.0394  rtg_e=1.2595  rtg_b=-1.2590  cal_e=1.2590  u_l2=0.0003  cov_n=0.0018
        col=11.1537  smp=0.0037  enc_t=0.2533  bnd_t=0.0402  wm_t=1.0357  crt_t=0.0087  diag_t=0.5826
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.06 c3=0.05 c4=0.05 c5=0.07 c6=0.05 c7=0.06 c8=0.07 c9=0.05 c10=0.09 c11=0.05 c12=0.06 c13=0.09 c14=0.05 c15=0.08
        symbols: 29/128 active  c0=2/8(H=0.62) c1=2/8(H=0.66) c2=1/8(H=0.00) c3=2/8(H=0.62) c4=2/8(H=0.69) c5=1/8(H=0.00) c6=2/8(H=0.64) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=2/8(H=0.69) c10=2/8(H=0.62) c11=2/8(H=0.28) c12=2/8(H=0.17) c13=3/8(H=0.59) c14=2/8(H=0.68) c15=2/8(H=0.23)
E0024 [5upd]  ep_rew=7.3774  rew_20=10.1466  L_geo=1.4001  L_rew=0.0011  L_chart=2.7652  L_crit=0.3378  L_bnd=0.8464  lr=0.0010  dt=19.51s
        recon=0.6706  vq=0.0670  code_H=1.5160  code_px=4.7005  ch_usage=2.5338  rtr_mrg=0.0116  enc_gn=16.5800
        ctrl=0.0002  tex=0.0294  im_rew=0.0354  im_ret=0.5377  value=0.0423  wm_gn=0.1157
        z_norm=0.5877  z_max=0.8122  jump=0.0000  cons=0.6223  sol=0.6144  e_var=0.0008  ch_ent=2.7442  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5377  dret=0.4957  term=0.0488  bnd=0.0847  chart_acc=0.0892  chart_ent=2.7701  rw_drift=0.0000
        v_err=1.2492  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4658
        bnd_x=0.0766  bell=0.0382  bell_s=0.0408  rtg_e=1.2492  rtg_b=-1.2488  cal_e=1.2488  u_l2=0.0003  cov_n=0.0029
        col=11.4895  smp=0.0028  enc_t=0.2615  bnd_t=0.0417  wm_t=1.1546  crt_t=0.0096  diag_t=0.6134
        charts: 16/16 active  c0=0.05 c1=0.07 c2=0.05 c3=0.04 c4=0.07 c5=0.06 c6=0.06 c7=0.05 c8=0.06 c9=0.10 c10=0.09 c11=0.05 c12=0.05 c13=0.09 c14=0.05 c15=0.06
        symbols: 35/128 active  c0=2/8(H=0.64) c1=2/8(H=0.69) c2=1/8(H=0.00) c3=2/8(H=0.69) c4=3/8(H=1.09) c5=1/8(H=0.00) c6=3/8(H=1.06) c7=2/8(H=0.16) c8=1/8(H=0.00) c9=4/8(H=1.06) c10=2/8(H=0.59) c11=2/8(H=0.37) c12=2/8(H=0.44) c13=3/8(H=0.55) c14=3/8(H=0.97) c15=2/8(H=0.27)
E0025 [5upd]  ep_rew=6.7123  rew_20=10.1857  L_geo=1.3703  L_rew=0.0015  L_chart=2.7662  L_crit=0.3335  L_bnd=0.9219  lr=0.0010  dt=20.58s
        recon=0.6685  vq=0.0709  code_H=1.6220  code_px=5.1472  ch_usage=2.2275  rtr_mrg=0.0196  enc_gn=14.2767
        ctrl=0.0001  tex=0.0357  im_rew=0.0350  im_ret=0.5236  value=0.0392  wm_gn=0.1179
        z_norm=0.5312  z_max=0.7626  jump=0.0000  cons=0.5212  sol=0.6622  e_var=0.0005  ch_ent=2.7364  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5236  dret=0.4896  term=0.0394  bnd=0.0693  chart_acc=0.0758  chart_ent=2.7697  rw_drift=0.0000
        v_err=1.5536  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2042
        bnd_x=0.0636  bell=0.0446  bell_s=0.0384  rtg_e=1.5536  rtg_b=-1.5533  cal_e=1.5533  u_l2=0.0003  cov_n=0.0009
        col=12.8566  smp=0.0040  enc_t=0.2619  bnd_t=0.0419  wm_t=1.1019  crt_t=0.0092  diag_t=0.5787
        charts: 16/16 active  c0=0.07 c1=0.13 c2=0.05 c3=0.06 c4=0.06 c5=0.06 c6=0.07 c7=0.05 c8=0.06 c9=0.06 c10=0.07 c11=0.04 c12=0.05 c13=0.06 c14=0.05 c15=0.06
        symbols: 38/128 active  c0=4/8(H=1.11) c1=6/8(H=1.47) c2=1/8(H=0.00) c3=2/8(H=0.66) c4=2/8(H=0.60) c5=2/8(H=0.49) c6=3/8(H=0.99) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=3/8(H=0.89) c10=2/8(H=0.69) c11=2/8(H=0.36) c12=2/8(H=0.15) c13=2/8(H=0.56) c14=3/8(H=0.94) c15=2/8(H=0.14)
E0026 [5upd]  ep_rew=14.3961  rew_20=10.4712  L_geo=1.3713  L_rew=0.0019  L_chart=2.7688  L_crit=0.3836  L_bnd=0.9805  lr=0.0010  dt=18.39s
        recon=0.6616  vq=0.0687  code_H=1.5787  code_px=4.9207  ch_usage=2.1419  rtr_mrg=0.0274  enc_gn=14.8633
        ctrl=0.0002  tex=0.0329  im_rew=0.0374  im_ret=0.5564  value=0.0386  wm_gn=0.1155
        z_norm=0.5656  z_max=0.7619  jump=0.0000  cons=0.5084  sol=0.8052  e_var=0.0006  ch_ent=2.7610  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5564  dret=0.5230  term=0.0388  bnd=0.0637  chart_acc=0.0733  chart_ent=2.7692  rw_drift=0.0000
        v_err=1.4518  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0789
        bnd_x=0.0590  bell=0.0437  bell_s=0.0547  rtg_e=1.4518  rtg_b=-1.4514  cal_e=1.4514  u_l2=0.0004  cov_n=0.0019
        col=11.0870  smp=0.0037  enc_t=0.2537  bnd_t=0.0399  wm_t=1.0289  crt_t=0.0086  diag_t=0.5843
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.05 c3=0.05 c4=0.06 c5=0.07 c6=0.07 c7=0.05 c8=0.06 c9=0.08 c10=0.07 c11=0.06 c12=0.05 c13=0.07 c14=0.06 c15=0.06
        symbols: 38/128 active  c0=2/8(H=0.69) c1=3/8(H=0.74) c2=2/8(H=0.31) c3=3/8(H=1.02) c4=3/8(H=1.04) c5=2/8(H=0.64) c6=5/8(H=1.06) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=3/8(H=0.94) c10=2/8(H=0.62) c11=2/8(H=0.58) c12=2/8(H=0.58) c13=2/8(H=0.48) c14=3/8(H=1.06) c15=2/8(H=0.30)
E0027 [5upd]  ep_rew=14.6911  rew_20=11.9301  L_geo=1.4076  L_rew=0.0010  L_chart=2.7645  L_crit=0.2858  L_bnd=0.9359  lr=0.0010  dt=18.44s
        recon=0.6540  vq=0.0681  code_H=1.7847  code_px=5.9822  ch_usage=1.6881  rtr_mrg=0.0201  enc_gn=12.1443
        ctrl=0.0001  tex=0.0333  im_rew=0.0365  im_ret=0.5417  value=0.0399  wm_gn=0.1213
        z_norm=0.6132  z_max=0.7833  jump=0.0000  cons=0.3892  sol=0.8001  e_var=0.0009  ch_ent=2.7281  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5417  dret=0.5113  term=0.0354  bnd=0.0596  chart_acc=0.0850  chart_ent=2.7683  rw_drift=0.0000
        v_err=1.2311  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0737
        bnd_x=0.0557  bell=0.0367  bell_s=0.0349  rtg_e=1.2311  rtg_b=-1.2307  cal_e=1.2307  u_l2=0.0004  cov_n=0.0014
        col=11.1581  smp=0.0027  enc_t=0.2535  bnd_t=0.0399  wm_t=1.0264  crt_t=0.0086  diag_t=0.5732
        charts: 16/16 active  c0=0.04 c1=0.04 c2=0.05 c3=0.05 c4=0.05 c5=0.07 c6=0.05 c7=0.05 c8=0.07 c9=0.07 c10=0.09 c11=0.06 c12=0.06 c13=0.10 c14=0.08 c15=0.09
        symbols: 37/128 active  c0=2/8(H=0.29) c1=2/8(H=0.60) c2=1/8(H=0.00) c3=3/8(H=0.54) c4=3/8(H=0.93) c5=2/8(H=0.20) c6=4/8(H=1.06) c7=2/8(H=0.31) c8=1/8(H=0.00) c9=4/8(H=1.03) c10=2/8(H=0.40) c11=2/8(H=0.49) c12=2/8(H=0.56) c13=2/8(H=0.31) c14=3/8(H=0.96) c15=2/8(H=0.31)
E0028 [5upd]  ep_rew=14.2509  rew_20=11.8801  L_geo=1.3885  L_rew=0.0010  L_chart=2.7491  L_crit=0.2746  L_bnd=0.8461  lr=0.0010  dt=18.85s
        recon=0.6395  vq=0.0708  code_H=1.7472  code_px=5.7492  ch_usage=1.3999  rtr_mrg=0.0382  enc_gn=14.9132
        ctrl=0.0001  tex=0.0295  im_rew=0.0317  im_ret=0.4779  value=0.0377  wm_gn=0.1323
        z_norm=0.6037  z_max=0.8254  jump=0.0000  cons=0.3629  sol=0.8104  e_var=0.0016  ch_ent=2.7034  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4779  dret=0.4432  term=0.0403  bnd=0.0783  chart_acc=0.1058  chart_ent=2.7668  rw_drift=0.0000
        v_err=1.2881  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4783
        bnd_x=0.0719  bell=0.0402  bell_s=0.0451  rtg_e=1.2881  rtg_b=-1.2877  cal_e=1.2877  u_l2=0.0004  cov_n=0.0020
        col=11.4290  smp=0.0031  enc_t=0.2530  bnd_t=0.0401  wm_t=1.0545  crt_t=0.0086  diag_t=0.5711
        charts: 16/16 active  c0=0.06 c1=0.05 c2=0.04 c3=0.05 c4=0.03 c5=0.07 c6=0.05 c7=0.04 c8=0.05 c9=0.12 c10=0.10 c11=0.05 c12=0.05 c13=0.10 c14=0.07 c15=0.08
        symbols: 39/128 active  c0=3/8(H=0.94) c1=3/8(H=0.87) c2=2/8(H=0.35) c3=3/8(H=1.01) c4=3/8(H=0.92) c5=2/8(H=0.42) c6=3/8(H=1.04) c7=2/8(H=0.14) c8=1/8(H=0.00) c9=3/8(H=0.75) c10=2/8(H=0.31) c11=2/8(H=0.51) c12=2/8(H=0.51) c13=3/8(H=0.38) c14=3/8(H=1.01) c15=2/8(H=0.11)
E0029 [5upd]  ep_rew=9.9207  rew_20=12.5950  L_geo=1.3515  L_rew=0.0007  L_chart=2.7586  L_crit=0.2585  L_bnd=0.8917  lr=0.0010  dt=18.45s
        recon=0.5986  vq=0.0740  code_H=1.6425  code_px=5.2085  ch_usage=0.9467  rtr_mrg=0.0388  enc_gn=13.8052
        ctrl=0.0001  tex=0.0321  im_rew=0.0297  im_ret=0.4409  value=0.0314  wm_gn=0.1166
        z_norm=0.6056  z_max=0.8415  jump=0.0000  cons=0.3756  sol=0.8957  e_var=0.0016  ch_ent=2.7165  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4409  dret=0.4156  term=0.0294  bnd=0.0607  chart_acc=0.0850  chart_ent=2.7649  rw_drift=0.0000
        v_err=1.4278  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2168
        bnd_x=0.0571  bell=0.0407  bell_s=0.0307  rtg_e=1.4278  rtg_b=-1.4276  cal_e=1.4276  u_l2=0.0005  cov_n=0.0010
        col=11.1550  smp=0.0036  enc_t=0.2531  bnd_t=0.0399  wm_t=1.0284  crt_t=0.0086  diag_t=0.5823
        charts: 16/16 active  c0=0.12 c1=0.05 c2=0.06 c3=0.05 c4=0.04 c5=0.06 c6=0.05 c7=0.05 c8=0.04 c9=0.12 c10=0.07 c11=0.05 c12=0.05 c13=0.06 c14=0.07 c15=0.06
        symbols: 46/128 active  c0=7/8(H=1.37) c1=4/8(H=1.26) c2=2/8(H=0.64) c3=3/8(H=0.96) c4=3/8(H=0.85) c5=2/8(H=0.53) c6=5/8(H=1.36) c7=2/8(H=0.19) c8=1/8(H=0.00) c9=4/8(H=0.73) c10=2/8(H=0.59) c11=2/8(H=0.39) c12=2/8(H=0.40) c13=2/8(H=0.22) c14=3/8(H=1.03) c15=2/8(H=0.15)
E0030 [5upd]  ep_rew=11.7244  rew_20=13.5197  L_geo=1.2478  L_rew=0.0013  L_chart=2.7724  L_crit=0.3691  L_bnd=1.0352  lr=0.0010  dt=18.26s
        recon=0.6008  vq=0.0876  code_H=1.5529  code_px=4.7499  ch_usage=1.3665  rtr_mrg=0.0189  enc_gn=15.7482
        ctrl=0.0001  tex=0.0340  im_rew=0.0367  im_ret=0.5382  value=0.0280  wm_gn=0.0756
        z_norm=0.5592  z_max=0.7930  jump=0.0000  cons=0.2312  sol=0.8952  e_var=0.0013  ch_ent=2.5975  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5382  dret=0.5136  term=0.0286  bnd=0.0479  chart_acc=0.0588  chart_ent=2.7618  rw_drift=0.0000
        v_err=1.3829  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2656
        bnd_x=0.0455  bell=0.0386  bell_s=0.0396  rtg_e=1.3829  rtg_b=-1.3828  cal_e=1.3828  u_l2=0.0007  cov_n=0.0012
        col=11.0580  smp=0.0037  enc_t=0.2518  bnd_t=0.0397  wm_t=1.0128  crt_t=0.0085  diag_t=0.5709
        charts: 16/16 active  c0=0.20 c1=0.14 c2=0.05 c3=0.06 c4=0.03 c5=0.06 c6=0.05 c7=0.05 c8=0.05 c9=0.03 c10=0.06 c11=0.04 c12=0.04 c13=0.05 c14=0.03 c15=0.04
        symbols: 40/128 active  c0=4/8(H=1.04) c1=6/8(H=1.47) c2=2/8(H=0.57) c3=4/8(H=1.09) c4=2/8(H=0.69) c5=2/8(H=0.54) c6=4/8(H=1.28) c7=1/8(H=0.00) c8=2/8(H=0.04) c9=2/8(H=0.66) c10=2/8(H=0.03) c11=2/8(H=0.11) c12=2/8(H=0.10) c13=2/8(H=0.36) c14=2/8(H=0.56) c15=1/8(H=0.00)
  EVAL  reward=14.9 +/- 4.7  len=300
  New best: 14.9
E0031 [5upd]  ep_rew=14.1499  rew_20=13.7698  L_geo=1.2070  L_rew=0.0006  L_chart=2.7772  L_crit=0.2836  L_bnd=1.0541  lr=0.0010  dt=16.13s
        recon=0.5564  vq=0.0907  code_H=1.5872  code_px=4.9101  ch_usage=1.6274  rtr_mrg=0.0158  enc_gn=14.2235
        ctrl=0.0001  tex=0.0347  im_rew=0.0395  im_ret=0.5742  value=0.0262  wm_gn=0.1016
        z_norm=0.5403  z_max=0.8413  jump=0.0000  cons=0.2530  sol=0.9502  e_var=0.0007  ch_ent=2.7093  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5742  dret=0.5534  term=0.0241  bnd=0.0375  chart_acc=0.0513  chart_ent=2.7613  rw_drift=0.0000
        v_err=1.2855  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.4174
        bnd_x=0.0360  bell=0.0357  bell_s=0.0308  rtg_e=1.2855  rtg_b=-1.2854  cal_e=1.2854  u_l2=0.0009  cov_n=0.0009
        col=10.7558  smp=0.0040  enc_t=0.2329  bnd_t=0.0371  wm_t=0.7071  crt_t=0.0059  diag_t=0.3963
        charts: 16/16 active  c0=0.12 c1=0.11 c2=0.05 c3=0.06 c4=0.04 c5=0.07 c6=0.09 c7=0.05 c8=0.05 c9=0.05 c10=0.07 c11=0.05 c12=0.04 c13=0.05 c14=0.05 c15=0.05
        symbols: 45/128 active  c0=4/8(H=1.08) c1=5/8(H=0.74) c2=2/8(H=0.14) c3=4/8(H=1.09) c4=3/8(H=0.66) c5=2/8(H=0.23) c6=6/8(H=1.52) c7=1/8(H=0.00) c8=2/8(H=0.25) c9=4/8(H=1.16) c10=1/8(H=0.00) c11=2/8(H=0.31) c12=2/8(H=0.26) c13=2/8(H=0.11) c14=3/8(H=0.81) c15=2/8(H=0.04)
E0032 [5upd]  ep_rew=11.1978  rew_20=12.9473  L_geo=1.2034  L_rew=0.0010  L_chart=2.7823  L_crit=0.2590  L_bnd=1.0802  lr=0.0010  dt=12.40s
        recon=0.5634  vq=0.0970  code_H=1.6489  code_px=5.2084  ch_usage=1.2896  rtr_mrg=0.0240  enc_gn=19.5915
        ctrl=0.0001  tex=0.0349  im_rew=0.0343  im_ret=0.5022  value=0.0239  wm_gn=0.0793
        z_norm=0.5373  z_max=0.8822  jump=0.0000  cons=0.2351  sol=0.9476  e_var=0.0008  ch_ent=2.7189  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5022  dret=0.4803  term=0.0255  bnd=0.0457  chart_acc=0.0496  chart_ent=2.7632  rw_drift=0.0000
        v_err=1.1910  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5494
        bnd_x=0.0437  bell=0.0332  bell_s=0.0230  rtg_e=1.1910  rtg_b=-1.1909  cal_e=1.1909  u_l2=0.0011  cov_n=0.0010
        col=7.0979  smp=0.0023  enc_t=0.2259  bnd_t=0.0345  wm_t=0.7084  crt_t=0.0059  diag_t=0.3756
        charts: 16/16 active  c0=0.05 c1=0.08 c2=0.05 c3=0.06 c4=0.04 c5=0.06 c6=0.14 c7=0.05 c8=0.05 c9=0.07 c10=0.08 c11=0.05 c12=0.05 c13=0.06 c14=0.06 c15=0.06
        symbols: 41/128 active  c0=2/8(H=0.64) c1=4/8(H=1.11) c2=1/8(H=0.00) c3=3/8(H=0.94) c4=3/8(H=0.65) c5=2/8(H=0.16) c6=6/8(H=1.52) c7=1/8(H=0.00) c8=2/8(H=0.28) c9=4/8(H=1.31) c10=2/8(H=0.03) c11=2/8(H=0.65) c12=2/8(H=0.26) c13=2/8(H=0.24) c14=3/8(H=0.99) c15=2/8(H=0.13)
E0033 [5upd]  ep_rew=10.7143  rew_20=12.7777  L_geo=1.1729  L_rew=0.0009  L_chart=2.7710  L_crit=0.2834  L_bnd=1.0764  lr=0.0010  dt=12.88s
        recon=0.5440  vq=0.1230  code_H=1.4725  code_px=4.3651  ch_usage=1.1189  rtr_mrg=0.0145  enc_gn=13.1232
        ctrl=0.0001  tex=0.0333  im_rew=0.0311  im_ret=0.4546  value=0.0222  wm_gn=0.0840
        z_norm=0.5496  z_max=0.8977  jump=0.0000  cons=0.2041  sol=0.9356  e_var=0.0012  ch_ent=2.6800  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4546  dret=0.4356  term=0.0221  bnd=0.0436  chart_acc=0.0496  chart_ent=2.7660  rw_drift=0.0000
        v_err=1.2659  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5033
        bnd_x=0.0419  bell=0.0363  bell_s=0.0347  rtg_e=1.2659  rtg_b=-1.2659  cal_e=1.2659  u_l2=0.0013  cov_n=0.0006
        col=7.3725  smp=0.0012  enc_t=0.2260  bnd_t=0.0347  wm_t=0.7444  crt_t=0.0060  diag_t=0.4023
        charts: 16/16 active  c0=0.17 c1=0.08 c2=0.05 c3=0.08 c4=0.04 c5=0.06 c6=0.08 c7=0.05 c8=0.05 c9=0.05 c10=0.07 c11=0.04 c12=0.04 c13=0.05 c14=0.05 c15=0.05
        symbols: 44/128 active  c0=5/8(H=1.01) c1=4/8(H=0.97) c2=2/8(H=0.17) c3=4/8(H=0.89) c4=2/8(H=0.49) c5=2/8(H=0.26) c6=5/8(H=1.32) c7=1/8(H=0.00) c8=2/8(H=0.38) c9=4/8(H=1.10) c10=2/8(H=0.09) c11=2/8(H=0.18) c12=2/8(H=0.36) c13=2/8(H=0.20) c14=3/8(H=0.97) c15=2/8(H=0.14)
E0034 [5upd]  ep_rew=14.3621  rew_20=13.2702  L_geo=1.1919  L_rew=0.0009  L_chart=2.7709  L_crit=0.3050  L_bnd=1.0161  lr=0.0010  dt=12.97s
        recon=0.5271  vq=0.1940  code_H=1.2844  code_px=3.6524  ch_usage=1.4664  rtr_mrg=0.0064  enc_gn=19.3784
        ctrl=0.0000  tex=0.0319  im_rew=0.0324  im_ret=0.4710  value=0.0214  wm_gn=0.7174
        z_norm=0.6277  z_max=0.9452  jump=0.0000  cons=0.1673  sol=0.9397  e_var=3.0852  ch_ent=2.6684  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4710  dret=0.4530  term=0.0210  bnd=0.0398  chart_acc=0.0438  chart_ent=2.7672  rw_drift=0.0000
        v_err=1.4278  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1997
        bnd_x=0.0386  bell=0.0397  bell_s=0.0383  rtg_e=1.4278  rtg_b=-1.4277  cal_e=1.4277  u_l2=0.0016  cov_n=0.0005
        col=7.4887  smp=0.0034  enc_t=0.2309  bnd_t=0.0347  wm_t=0.7286  crt_t=0.0062  diag_t=0.4249
        charts: 16/16 active  c0=0.09 c1=0.10 c2=0.05 c3=0.05 c4=0.03 c5=0.06 c6=0.05 c7=0.05 c8=0.04 c9=0.17 c10=0.06 c11=0.04 c12=0.05 c13=0.06 c14=0.05 c15=0.05
        symbols: 40/128 active  c0=4/8(H=1.17) c1=4/8(H=1.04) c2=2/8(H=0.07) c3=2/8(H=0.63) c4=3/8(H=0.10) c5=2/8(H=0.03) c6=5/8(H=1.30) c7=1/8(H=0.00) c8=2/8(H=0.11) c9=3/8(H=0.42) c10=2/8(H=0.13) c11=2/8(H=0.16) c12=2/8(H=0.22) c13=1/8(H=0.00) c14=3/8(H=0.75) c15=2/8(H=0.09)
E0035 [5upd]  ep_rew=16.1296  rew_20=12.3034  L_geo=1.1382  L_rew=0.0008  L_chart=2.7595  L_crit=0.2856  L_bnd=0.8979  lr=0.0010  dt=13.00s
        recon=0.5013  vq=0.1541  code_H=1.0419  code_px=2.8729  ch_usage=1.5612  rtr_mrg=0.0073  enc_gn=14.7289
        ctrl=0.0000  tex=0.0346  im_rew=0.0343  im_ret=0.4966  value=0.0216  wm_gn=0.0824
        z_norm=0.5971  z_max=0.8586  jump=0.0000  cons=0.1930  sol=0.9943  e_var=0.0037  ch_ent=2.6977  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4966  dret=0.4801  term=0.0193  bnd=0.0345  chart_acc=0.0492  chart_ent=2.7681  rw_drift=0.0000
        v_err=1.2800  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2267
        bnd_x=0.0336  bell=0.0346  bell_s=0.0353  rtg_e=1.2800  rtg_b=-1.2799  cal_e=1.2799  u_l2=0.0019  cov_n=0.0004
        col=7.4948  smp=0.0036  enc_t=0.2291  bnd_t=0.0347  wm_t=0.7356  crt_t=0.0063  diag_t=0.4223
        charts: 16/16 active  c0=0.14 c1=0.04 c2=0.05 c3=0.07 c4=0.04 c5=0.05 c6=0.07 c7=0.05 c8=0.04 c9=0.10 c10=0.07 c11=0.04 c12=0.05 c13=0.06 c14=0.06 c15=0.07
        symbols: 43/128 active  c0=3/8(H=0.92) c1=4/8(H=1.09) c2=2/8(H=0.04) c3=5/8(H=1.01) c4=3/8(H=0.38) c5=2/8(H=0.13) c6=4/8(H=1.23) c7=2/8(H=0.04) c8=2/8(H=0.11) c9=3/8(H=0.67) c10=2/8(H=0.10) c11=2/8(H=0.32) c12=2/8(H=0.31) c13=2/8(H=0.14) c14=3/8(H=0.90) c15=2/8(H=0.13)
E0036 [5upd]  ep_rew=14.2080  rew_20=12.3358  L_geo=1.1751  L_rew=0.0007  L_chart=2.7593  L_crit=0.2662  L_bnd=0.8023  lr=0.0010  dt=13.02s
        recon=0.4630  vq=0.1485  code_H=1.2009  code_px=3.3562  ch_usage=1.6671  rtr_mrg=0.0092  enc_gn=13.4814
        ctrl=0.0000  tex=0.0329  im_rew=0.0333  im_ret=0.4866  value=0.0223  wm_gn=0.0745
        z_norm=0.5841  z_max=0.8795  jump=0.0000  cons=0.1688  sol=0.9547  e_var=0.0042  ch_ent=2.6352  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4866  dret=0.4664  term=0.0234  bnd=0.0432  chart_acc=0.0538  chart_ent=2.7684  rw_drift=0.0000
        v_err=1.4494  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5574
        bnd_x=0.0420  bell=0.0412  bell_s=0.0394  rtg_e=1.4494  rtg_b=-1.4493  cal_e=1.4493  u_l2=0.0024  cov_n=0.0004
        col=7.5419  smp=0.0026  enc_t=0.2262  bnd_t=0.0349  wm_t=0.7394  crt_t=0.0059  diag_t=0.3949
        charts: 16/16 active  c0=0.17 c1=0.05 c2=0.05 c3=0.08 c4=0.03 c5=0.05 c6=0.07 c7=0.05 c8=0.05 c9=0.13 c10=0.06 c11=0.03 c12=0.04 c13=0.05 c14=0.03 c15=0.05
        symbols: 43/128 active  c0=3/8(H=0.85) c1=3/8(H=0.84) c2=2/8(H=0.14) c3=5/8(H=1.10) c4=2/8(H=0.48) c5=2/8(H=0.13) c6=5/8(H=1.42) c7=1/8(H=0.00) c8=2/8(H=0.15) c9=4/8(H=0.72) c10=2/8(H=0.03) c11=3/8(H=0.58) c12=2/8(H=0.31) c13=2/8(H=0.28) c14=3/8(H=0.86) c15=2/8(H=0.21)
E0037 [5upd]  ep_rew=13.6806  rew_20=12.7447  L_geo=1.0586  L_rew=0.0011  L_chart=2.7607  L_crit=0.2983  L_bnd=0.7557  lr=0.0010  dt=13.21s
        recon=0.4654  vq=0.2405  code_H=1.1735  code_px=3.2420  ch_usage=1.1333  rtr_mrg=0.0110  enc_gn=13.4030
        ctrl=0.0001  tex=0.0344  im_rew=0.0368  im_ret=0.5362  value=0.0237  wm_gn=0.0916
        z_norm=0.5415  z_max=0.8716  jump=0.0000  cons=0.1890  sol=0.9847  e_var=0.0065  ch_ent=2.4238  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5362  dret=0.5152  term=0.0244  bnd=0.0407  chart_acc=0.0446  chart_ent=2.7681  rw_drift=0.0000
        v_err=1.3063  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7577
        bnd_x=0.0396  bell=0.0358  bell_s=0.0191  rtg_e=1.3063  rtg_b=-1.3063  cal_e=1.3063  u_l2=0.0030  cov_n=0.0005
        col=7.6367  smp=0.0022  enc_t=0.2301  bnd_t=0.0350  wm_t=0.7444  crt_t=0.0066  diag_t=0.4323
        charts: 16/16 active  c0=0.22 c1=0.12 c2=0.03 c3=0.15 c4=0.02 c5=0.03 c6=0.04 c7=0.04 c8=0.03 c9=0.14 c10=0.04 c11=0.02 c12=0.03 c13=0.03 c14=0.03 c15=0.03
        symbols: 43/128 active  c0=5/8(H=1.27) c1=3/8(H=0.64) c2=2/8(H=0.31) c3=5/8(H=0.85) c4=2/8(H=0.37) c5=2/8(H=0.20) c6=5/8(H=1.28) c7=1/8(H=0.00) c8=2/8(H=0.29) c9=4/8(H=0.46) c10=1/8(H=0.00) c11=3/8(H=0.57) c12=2/8(H=0.54) c13=2/8(H=0.20) c14=2/8(H=0.49) c15=2/8(H=0.05)
E0038 [5upd]  ep_rew=13.6753  rew_20=13.5574  L_geo=0.8919  L_rew=0.0007  L_chart=2.7489  L_crit=0.2862  L_bnd=0.7579  lr=0.0010  dt=13.13s
        recon=0.4185  vq=0.3870  code_H=1.2010  code_px=3.3506  ch_usage=0.8265  rtr_mrg=0.0112  enc_gn=15.1113
        ctrl=0.0001  tex=0.0363  im_rew=0.0361  im_ret=0.5280  value=0.0275  wm_gn=0.1241
        z_norm=0.5435  z_max=0.9063  jump=0.0000  cons=0.1553  sol=0.9781  e_var=0.0007  ch_ent=2.4885  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5280  dret=0.5052  term=0.0265  bnd=0.0451  chart_acc=0.0679  chart_ent=2.7679  rw_drift=0.0000
        v_err=1.1672  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7615
        bnd_x=0.0439  bell=0.0342  bell_s=0.0286  rtg_e=1.1672  rtg_b=-1.1671  cal_e=1.1671  u_l2=0.0037  cov_n=0.0005
        col=7.5156  smp=0.0029  enc_t=0.2281  bnd_t=0.0350  wm_t=0.7645  crt_t=0.0064  diag_t=0.3971
        charts: 16/16 active  c0=0.15 c1=0.23 c2=0.04 c3=0.09 c4=0.03 c5=0.04 c6=0.08 c7=0.04 c8=0.03 c9=0.08 c10=0.06 c11=0.02 c12=0.03 c13=0.03 c14=0.02 c15=0.03
        symbols: 40/128 active  c0=5/8(H=1.36) c1=4/8(H=0.46) c2=1/8(H=0.00) c3=4/8(H=0.65) c4=2/8(H=0.54) c5=2/8(H=0.04) c6=6/8(H=1.17) c7=1/8(H=0.00) c8=2/8(H=0.11) c9=3/8(H=0.63) c10=1/8(H=0.00) c11=2/8(H=0.13) c12=2/8(H=0.06) c13=2/8(H=0.16) c14=2/8(H=0.21) c15=1/8(H=0.00)
E0039 [5upd]  ep_rew=13.6155  rew_20=12.7763  L_geo=0.9366  L_rew=0.0005  L_chart=2.7487  L_crit=0.2511  L_bnd=0.7967  lr=0.0010  dt=17.30s
        recon=0.4295  vq=0.3366  code_H=1.1476  code_px=3.1778  ch_usage=1.0333  rtr_mrg=0.0120  enc_gn=27.8947
        ctrl=0.0000  tex=0.0335  im_rew=0.0338  im_ret=0.4990  value=0.0291  wm_gn=0.0644
        z_norm=0.5574  z_max=0.9009  jump=0.0000  cons=0.1137  sol=0.9898  e_var=0.0008  ch_ent=2.5638  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4990  dret=0.4733  term=0.0299  bnd=0.0543  chart_acc=0.1338  chart_ent=2.7664  rw_drift=0.0000
        v_err=1.3618  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5585
        bnd_x=0.0528  bell=0.0378  bell_s=0.0255  rtg_e=1.3618  rtg_b=-1.3617  cal_e=1.3617  u_l2=0.0053  cov_n=0.0004
        col=9.1986  smp=0.0048  enc_t=0.2664  bnd_t=0.0428  wm_t=1.1635  crt_t=0.0098  diag_t=0.6144
        charts: 16/16 active  c0=0.09 c1=0.13 c2=0.04 c3=0.05 c4=0.03 c5=0.04 c6=0.16 c7=0.04 c8=0.03 c9=0.16 c10=0.05 c11=0.03 c12=0.03 c13=0.04 c14=0.03 c15=0.05
        symbols: 42/128 active  c0=4/8(H=1.10) c1=5/8(H=1.00) c2=2/8(H=0.16) c3=3/8(H=1.02) c4=2/8(H=0.23) c5=2/8(H=0.32) c6=5/8(H=0.85) c7=1/8(H=0.00) c8=2/8(H=0.09) c9=3/8(H=0.50) c10=2/8(H=0.04) c11=2/8(H=0.06) c12=2/8(H=0.36) c13=2/8(H=0.30) c14=3/8(H=0.68) c15=2/8(H=0.28)
E0040 [5upd]  ep_rew=6.1865  rew_20=12.7788  L_geo=0.9786  L_rew=0.0005  L_chart=2.7333  L_crit=0.2127  L_bnd=0.9367  lr=0.0010  dt=19.28s
        recon=0.3856  vq=0.2564  code_H=1.1680  code_px=3.2422  ch_usage=1.2076  rtr_mrg=0.0087  enc_gn=13.0702
        ctrl=0.0001  tex=0.0354  im_rew=0.0302  im_ret=0.4502  value=0.0333  wm_gn=0.0521
        z_norm=0.5378  z_max=0.7328  jump=0.0000  cons=0.1565  sol=0.9682  e_var=0.0008  ch_ent=2.6186  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4502  dret=0.4220  term=0.0327  bnd=0.0667  chart_acc=0.1588  chart_ent=2.7640  rw_drift=0.0000
        v_err=1.0195  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0222
        bnd_x=0.0640  bell=0.0295  bell_s=0.0185  rtg_e=1.0195  rtg_b=-1.0191  cal_e=1.0191  u_l2=0.0079  cov_n=0.0006
        col=11.8102  smp=0.0035  enc_t=0.2574  bnd_t=0.0408  wm_t=1.0562  crt_t=0.0089  diag_t=0.5853
        charts: 16/16 active  c0=0.16 c1=0.14 c2=0.04 c3=0.08 c4=0.03 c5=0.05 c6=0.09 c7=0.06 c8=0.04 c9=0.06 c10=0.06 c11=0.03 c12=0.04 c13=0.04 c14=0.03 c15=0.04
        symbols: 37/128 active  c0=4/8(H=1.30) c1=4/8(H=0.95) c2=2/8(H=0.08) c3=3/8(H=0.73) c4=2/8(H=0.19) c5=2/8(H=0.04) c6=4/8(H=0.92) c7=1/8(H=0.00) c8=2/8(H=0.08) c9=3/8(H=0.76) c10=1/8(H=0.00) c11=3/8(H=0.12) c12=1/8(H=0.00) c13=2/8(H=0.08) c14=2/8(H=0.17) c15=1/8(H=0.00)
E0041 [5upd]  ep_rew=6.5112  rew_20=12.5133  L_geo=0.9828  L_rew=0.0007  L_chart=2.7247  L_crit=0.2518  L_bnd=1.0530  lr=0.0010  dt=19.00s
        recon=0.4307  vq=0.2401  code_H=1.2397  code_px=3.4695  ch_usage=1.0418  rtr_mrg=0.0117  enc_gn=10.2366
        ctrl=0.0001  tex=0.0327  im_rew=0.0299  im_ret=0.4514  value=0.0368  wm_gn=0.0731
        z_norm=0.5685  z_max=0.8759  jump=0.0000  cons=0.1236  sol=0.9750  e_var=0.0006  ch_ent=2.6944  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4514  dret=0.4181  term=0.0387  bnd=0.0795  chart_acc=0.1263  chart_ent=2.7583  rw_drift=0.0000
        v_err=1.3103  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.4913
        bnd_x=0.0762  bell=0.0374  bell_s=0.0274  rtg_e=1.3103  rtg_b=-1.3101  cal_e=1.3101  u_l2=0.0112  cov_n=0.0009
        col=11.5870  smp=0.0036  enc_t=0.2552  bnd_t=0.0403  wm_t=1.0471  crt_t=0.0092  diag_t=0.5898
        charts: 16/16 active  c0=0.10 c1=0.10 c2=0.05 c3=0.06 c4=0.03 c5=0.05 c6=0.13 c7=0.06 c8=0.04 c9=0.09 c10=0.07 c11=0.03 c12=0.05 c13=0.05 c14=0.04 c15=0.05
        symbols: 45/128 active  c0=3/8(H=1.07) c1=5/8(H=1.22) c2=2/8(H=0.07) c3=4/8(H=0.88) c4=3/8(H=0.21) c5=2/8(H=0.06) c6=6/8(H=1.14) c7=2/8(H=0.03) c8=2/8(H=0.04) c9=3/8(H=0.81) c10=2/8(H=0.03) c11=3/8(H=0.37) c12=2/8(H=0.27) c13=2/8(H=0.09) c14=2/8(H=0.56) c15=2/8(H=0.33)
E0042 [5upd]  ep_rew=14.1787  rew_20=12.3709  L_geo=0.8756  L_rew=0.0006  L_chart=2.6717  L_crit=0.2853  L_bnd=0.9400  lr=0.0010  dt=18.75s
        recon=0.4137  vq=0.3676  code_H=1.3313  code_px=3.8071  ch_usage=0.8227  rtr_mrg=0.0122  enc_gn=13.6253
        ctrl=0.0001  tex=0.0381  im_rew=0.0376  im_ret=0.5559  value=0.0368  wm_gn=0.1094
        z_norm=0.5079  z_max=0.7959  jump=0.0000  cons=0.1072  sol=0.9842  e_var=0.0006  ch_ent=2.4116  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5559  dret=0.5258  term=0.0350  bnd=0.0573  chart_acc=0.1863  chart_ent=2.7475  rw_drift=0.0000
        v_err=1.3843  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0677
        bnd_x=0.0548  bell=0.0396  bell_s=0.0257  rtg_e=1.3843  rtg_b=-1.3841  cal_e=1.3841  u_l2=0.0148  cov_n=0.0009
        col=11.3652  smp=0.0028  enc_t=0.2550  bnd_t=0.0405  wm_t=1.0426  crt_t=0.0088  diag_t=0.5886
        charts: 16/16 active  c0=0.19 c1=0.18 c2=0.03 c3=0.12 c4=0.02 c5=0.03 c6=0.13 c7=0.03 c8=0.02 c9=0.09 c10=0.04 c11=0.02 c12=0.02 c13=0.02 c14=0.03 c15=0.03
        symbols: 44/128 active  c0=5/8(H=1.34) c1=5/8(H=1.03) c2=2/8(H=0.06) c3=4/8(H=0.52) c4=2/8(H=0.54) c5=2/8(H=0.39) c6=5/8(H=1.38) c7=1/8(H=0.00) c8=2/8(H=0.18) c9=3/8(H=0.55) c10=1/8(H=0.00) c11=3/8(H=0.49) c12=2/8(H=0.52) c13=2/8(H=0.40) c14=3/8(H=0.89) c15=2/8(H=0.06)
E0043 [5upd]  ep_rew=14.5429  rew_20=11.9226  L_geo=0.8500  L_rew=0.0007  L_chart=2.6057  L_crit=0.2625  L_bnd=0.7496  lr=0.0010  dt=18.73s
        recon=0.3865  vq=0.4375  code_H=1.1426  code_px=3.1410  ch_usage=0.7935  rtr_mrg=0.0084  enc_gn=17.4581
        ctrl=0.0001  tex=0.0329  im_rew=0.0353  im_ret=0.5293  value=0.0402  wm_gn=0.1421
        z_norm=0.5845  z_max=0.7794  jump=0.0000  cons=0.1197  sol=0.9793  e_var=0.0008  ch_ent=2.5877  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5293  dret=0.4938  term=0.0413  bnd=0.0719  chart_acc=0.1642  chart_ent=2.7021  rw_drift=0.0000
        v_err=1.3082  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6836
        bnd_x=0.0711  bell=0.0374  bell_s=0.0295  rtg_e=1.3082  rtg_b=-1.3078  cal_e=1.3078  u_l2=0.0222  cov_n=0.0015
        col=11.2324  smp=0.0036  enc_t=0.2536  bnd_t=0.0402  wm_t=1.0697  crt_t=0.0087  diag_t=0.5758
        charts: 16/16 active  c0=0.15 c1=0.17 c2=0.04 c3=0.05 c4=0.02 c5=0.04 c6=0.09 c7=0.05 c8=0.03 c9=0.08 c10=0.07 c11=0.03 c12=0.04 c13=0.05 c14=0.03 c15=0.04
        symbols: 39/128 active  c0=4/8(H=0.95) c1=3/8(H=0.34) c2=1/8(H=0.00) c3=3/8(H=0.77) c4=3/8(H=0.46) c5=1/8(H=0.00) c6=5/8(H=0.94) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=4/8(H=1.16) c10=2/8(H=0.20) c11=2/8(H=0.06) c12=2/8(H=0.39) c13=2/8(H=0.12) c14=3/8(H=0.70) c15=2/8(H=0.33)
E0044 [5upd]  ep_rew=14.1101  rew_20=12.9450  L_geo=0.9241  L_rew=0.0012  L_chart=2.5065  L_crit=0.3114  L_bnd=0.7163  lr=0.0010  dt=18.56s
        recon=0.3957  vq=0.2909  code_H=1.2330  code_px=3.4383  ch_usage=0.9759  rtr_mrg=0.0084  enc_gn=11.7335
        ctrl=0.0001  tex=0.0344  im_rew=0.0327  im_ret=0.4901  value=0.0360  wm_gn=0.1551
        z_norm=0.5642  z_max=0.8682  jump=0.0000  cons=0.1085  sol=0.9800  e_var=0.0007  ch_ent=2.6243  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4901  dret=0.4578  term=0.0376  bnd=0.0706  chart_acc=0.1733  chart_ent=2.5311  rw_drift=0.0000
        v_err=1.3301  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7776
        bnd_x=0.0704  bell=0.0391  bell_s=0.0320  rtg_e=1.3301  rtg_b=-1.3299  cal_e=1.3299  u_l2=0.0344  cov_n=0.0010
        col=11.2307  smp=0.0039  enc_t=0.2543  bnd_t=0.0400  wm_t=1.0341  crt_t=0.0087  diag_t=0.5806
        charts: 16/16 active  c0=0.18 c1=0.10 c2=0.05 c3=0.09 c4=0.03 c5=0.04 c6=0.09 c7=0.05 c8=0.03 c9=0.08 c10=0.07 c11=0.03 c12=0.04 c13=0.05 c14=0.04 c15=0.04
        symbols: 47/128 active  c0=5/8(H=1.37) c1=5/8(H=0.87) c2=2/8(H=0.26) c3=5/8(H=0.97) c4=3/8(H=0.58) c5=2/8(H=0.17) c6=5/8(H=1.17) c7=1/8(H=0.00) c8=2/8(H=0.09) c9=5/8(H=1.48) c10=1/8(H=0.00) c11=2/8(H=0.28) c12=2/8(H=0.56) c13=2/8(H=0.28) c14=3/8(H=1.01) c15=2/8(H=0.35)
E0045 [5upd]  ep_rew=8.5686  rew_20=13.8416  L_geo=0.8394  L_rew=0.0006  L_chart=2.5008  L_crit=0.2678  L_bnd=0.9244  lr=0.0010  dt=18.47s
        recon=0.3809  vq=0.3417  code_H=1.2693  code_px=3.5654  ch_usage=0.7258  rtr_mrg=0.0072  enc_gn=13.2317
        ctrl=0.0001  tex=0.0369  im_rew=0.0357  im_ret=0.5266  value=0.0327  wm_gn=0.1095
        z_norm=0.5410  z_max=0.7626  jump=0.0000  cons=0.1159  sol=0.9951  e_var=0.0006  ch_ent=2.5757  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5266  dret=0.4995  term=0.0315  bnd=0.0542  chart_acc=0.1521  chart_ent=2.3644  rw_drift=0.0000
        v_err=1.3375  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0919
        bnd_x=0.0540  bell=0.0377  bell_s=0.0236  rtg_e=1.3375  rtg_b=-1.3373  cal_e=1.3373  u_l2=0.0492  cov_n=0.0011
        col=11.1725  smp=0.0026  enc_t=0.2533  bnd_t=0.0405  wm_t=1.0268  crt_t=0.0087  diag_t=0.5849
        charts: 16/16 active  c0=0.08 c1=0.10 c2=0.05 c3=0.13 c4=0.02 c5=0.05 c6=0.09 c7=0.04 c8=0.03 c9=0.17 c10=0.07 c11=0.02 c12=0.04 c13=0.04 c14=0.02 c15=0.04
        symbols: 48/128 active  c0=5/8(H=1.40) c1=5/8(H=1.40) c2=4/8(H=0.71) c3=5/8(H=0.79) c4=2/8(H=0.58) c5=2/8(H=0.45) c6=5/8(H=1.39) c7=1/8(H=0.00) c8=2/8(H=0.34) c9=4/8(H=0.86) c10=1/8(H=0.00) c11=4/8(H=0.26) c12=2/8(H=0.46) c13=2/8(H=0.24) c14=2/8(H=0.41) c15=2/8(H=0.20)
E0046 [5upd]  ep_rew=13.0594  rew_20=14.4845  L_geo=0.8546  L_rew=0.0009  L_chart=2.5534  L_crit=0.2869  L_bnd=1.0422  lr=0.0010  dt=18.50s
        recon=0.3761  vq=0.3131  code_H=1.3313  code_px=3.8318  ch_usage=0.6617  rtr_mrg=0.0079  enc_gn=13.5967
        ctrl=0.0001  tex=0.0335  im_rew=0.0317  im_ret=0.4697  value=0.0294  wm_gn=0.1165
        z_norm=0.5505  z_max=0.7779  jump=0.0000  cons=0.0960  sol=0.9868  e_var=0.0008  ch_ent=2.6649  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4697  dret=0.4432  term=0.0307  bnd=0.0596  chart_acc=0.1058  chart_ent=2.5574  rw_drift=0.0000
        v_err=1.3782  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5376
        bnd_x=0.0596  bell=0.0390  bell_s=0.0368  rtg_e=1.3782  rtg_b=-1.3781  cal_e=1.3781  u_l2=0.0672  cov_n=0.0011
        col=11.1976  smp=0.0037  enc_t=0.2546  bnd_t=0.0405  wm_t=1.0264  crt_t=0.0091  diag_t=0.5883
        charts: 16/16 active  c0=0.14 c1=0.04 c2=0.05 c3=0.12 c4=0.03 c5=0.06 c6=0.07 c7=0.06 c8=0.04 c9=0.09 c10=0.08 c11=0.04 c12=0.04 c13=0.05 c14=0.04 c15=0.05
        symbols: 51/128 active  c0=6/8(H=1.43) c1=3/8(H=0.58) c2=3/8(H=0.50) c3=7/8(H=1.61) c4=2/8(H=0.31) c5=3/8(H=0.09) c6=5/8(H=1.32) c7=1/8(H=0.00) c8=2/8(H=0.33) c9=5/8(H=1.28) c10=3/8(H=0.07) c11=3/8(H=0.68) c12=2/8(H=0.65) c13=2/8(H=0.39) c14=2/8(H=0.53) c15=2/8(H=0.41)
E0047 [5upd]  ep_rew=14.8185  rew_20=14.9111  L_geo=0.8858  L_rew=0.0011  L_chart=2.5511  L_crit=0.2603  L_bnd=1.0257  lr=0.0010  dt=18.61s
        recon=0.3908  vq=0.3586  code_H=1.3919  code_px=4.0386  ch_usage=0.3877  rtr_mrg=0.0134  enc_gn=13.3448
        ctrl=0.0001  tex=0.0358  im_rew=0.0368  im_ret=0.5358  value=0.0244  wm_gn=0.1263
        z_norm=0.5285  z_max=0.8082  jump=0.0000  cons=0.0959  sol=0.9877  e_var=0.0006  ch_ent=2.6250  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5358  dret=0.5156  term=0.0235  bnd=0.0391  chart_acc=0.1400  chart_ent=2.6227  rw_drift=0.0000
        v_err=1.4235  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5175
        bnd_x=0.0393  bell=0.0412  bell_s=0.0447  rtg_e=1.4235  rtg_b=-1.4234  cal_e=1.4234  u_l2=0.0846  cov_n=0.0006
        col=11.1699  smp=0.0038  enc_t=0.2532  bnd_t=0.0400  wm_t=1.0575  crt_t=0.0087  diag_t=0.5808
        charts: 16/16 active  c0=0.12 c1=0.08 c2=0.06 c3=0.14 c4=0.02 c5=0.06 c6=0.14 c7=0.05 c8=0.03 c9=0.04 c10=0.06 c11=0.04 c12=0.05 c13=0.04 c14=0.04 c15=0.03
        symbols: 47/128 active  c0=4/8(H=1.08) c1=4/8(H=0.96) c2=2/8(H=0.41) c3=5/8(H=1.08) c4=3/8(H=0.40) c5=2/8(H=0.03) c6=6/8(H=1.42) c7=1/8(H=0.00) c8=2/8(H=0.05) c9=3/8(H=0.81) c10=3/8(H=0.61) c11=3/8(H=0.43) c12=2/8(H=0.54) c13=2/8(H=0.35) c14=3/8(H=0.77) c15=2/8(H=0.44)
E0048 [5upd]  ep_rew=8.9311  rew_20=14.4929  L_geo=0.9034  L_rew=0.0015  L_chart=2.5311  L_crit=0.2942  L_bnd=0.8621  lr=0.0010  dt=18.42s
        recon=0.3956  vq=0.3288  code_H=1.3315  code_px=3.8612  ch_usage=0.4232  rtr_mrg=0.0163  enc_gn=10.8685
        ctrl=0.0001  tex=0.0340  im_rew=0.0368  im_ret=0.5346  value=0.0242  wm_gn=0.0927
        z_norm=0.5955  z_max=0.9194  jump=0.0000  cons=0.0820  sol=0.9920  e_var=0.0009  ch_ent=2.6134  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5346  dret=0.5155  term=0.0222  bnd=0.0371  chart_acc=0.1333  chart_ent=2.5783  rw_drift=0.0000
        v_err=1.4103  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1502
        bnd_x=0.0365  bell=0.0408  bell_s=0.0537  rtg_e=1.4103  rtg_b=-1.4103  cal_e=1.4103  u_l2=0.1093  cov_n=0.0006
        col=11.1449  smp=0.0045  enc_t=0.2544  bnd_t=0.0397  wm_t=1.0253  crt_t=0.0088  diag_t=0.5638
        charts: 16/16 active  c0=0.15 c1=0.08 c2=0.07 c3=0.06 c4=0.03 c5=0.06 c6=0.06 c7=0.05 c8=0.03 c9=0.17 c10=0.07 c11=0.03 c12=0.04 c13=0.04 c14=0.03 c15=0.04
        symbols: 51/128 active  c0=6/8(H=1.22) c1=8/8(H=1.30) c2=3/8(H=0.76) c3=3/8(H=0.84) c4=2/8(H=0.38) c5=2/8(H=0.08) c6=6/8(H=1.25) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=5/8(H=1.19) c10=3/8(H=0.62) c11=2/8(H=0.24) c12=2/8(H=0.53) c13=2/8(H=0.05) c14=3/8(H=0.81) c15=2/8(H=0.53)
E0049 [5upd]  ep_rew=8.5682  rew_20=13.9444  L_geo=0.9141  L_rew=0.0009  L_chart=2.4750  L_crit=0.2935  L_bnd=0.7000  lr=0.0010  dt=18.37s
        recon=0.3848  vq=0.3809  code_H=1.4747  code_px=4.4048  ch_usage=0.4112  rtr_mrg=0.0113  enc_gn=10.7077
        ctrl=0.0001  tex=0.0311  im_rew=0.0355  im_ret=0.5195  value=0.0251  wm_gn=0.0975
        z_norm=0.6038  z_max=0.8531  jump=0.0000  cons=0.0861  sol=0.9922  e_var=0.0013  ch_ent=2.6060  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5195  dret=0.4965  term=0.0268  bnd=0.0464  chart_acc=0.1279  chart_ent=2.5294  rw_drift=0.0000
        v_err=1.4029  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7964
        bnd_x=0.0479  bell=0.0386  bell_s=0.0259  rtg_e=1.4029  rtg_b=-1.4029  cal_e=1.4029  u_l2=0.1222  cov_n=0.0012
        col=11.1035  smp=0.0048  enc_t=0.2534  bnd_t=0.0398  wm_t=1.0239  crt_t=0.0086  diag_t=0.5736
        charts: 16/16 active  c0=0.08 c1=0.17 c2=0.08 c3=0.08 c4=0.03 c5=0.08 c6=0.04 c7=0.05 c8=0.03 c9=0.12 c10=0.09 c11=0.03 c12=0.04 c13=0.04 c14=0.02 c15=0.03
        symbols: 61/128 active  c0=7/8(H=1.30) c1=6/8(H=1.59) c2=4/8(H=1.10) c3=7/8(H=1.39) c4=2/8(H=0.41) c5=4/8(H=1.20) c6=8/8(H=1.36) c7=1/8(H=0.00) c8=2/8(H=0.10) c9=5/8(H=1.14) c10=3/8(H=0.79) c11=3/8(H=0.43) c12=2/8(H=0.56) c13=3/8(H=0.60) c14=2/8(H=0.62) c15=2/8(H=0.44)
E0050 [5upd]  ep_rew=11.2320  rew_20=13.0627  L_geo=0.8941  L_rew=0.0005  L_chart=2.3833  L_crit=0.2653  L_bnd=0.6565  lr=0.0010  dt=18.46s
        recon=0.3432  vq=0.4525  code_H=1.3243  code_px=3.7767  ch_usage=0.4689  rtr_mrg=0.0128  enc_gn=13.5305
        ctrl=0.0001  tex=0.0299  im_rew=0.0344  im_ret=0.5038  value=0.0265  wm_gn=0.1115
        z_norm=0.6194  z_max=0.8696  jump=0.0000  cons=0.0682  sol=0.9897  e_var=0.0028  ch_ent=2.5825  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5038  dret=0.4808  term=0.0268  bnd=0.0479  chart_acc=0.1504  chart_ent=2.3193  rw_drift=0.0000
        v_err=1.2179  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9000
        bnd_x=0.0502  bell=0.0344  bell_s=0.0304  rtg_e=1.2179  rtg_b=-1.2178  cal_e=1.2178  u_l2=0.1214  cov_n=0.0012
        col=11.2030  smp=0.0036  enc_t=0.2529  bnd_t=0.0402  wm_t=1.0225  crt_t=0.0086  diag_t=0.5732
        charts: 16/16 active  c0=0.10 c1=0.12 c2=0.05 c3=0.06 c4=0.02 c5=0.09 c6=0.14 c7=0.05 c8=0.03 c9=0.10 c10=0.10 c11=0.02 c12=0.02 c13=0.03 c14=0.03 c15=0.03
        symbols: 52/128 active  c0=5/8(H=1.30) c1=6/8(H=1.50) c2=3/8(H=0.73) c3=6/8(H=1.22) c4=2/8(H=0.09) c5=4/8(H=0.99) c6=8/8(H=1.76) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=5/8(H=1.08) c10=2/8(H=0.55) c11=3/8(H=0.64) c12=1/8(H=0.00) c13=2/8(H=0.67) c14=2/8(H=0.68) c15=1/8(H=0.00)
E0051 [5upd]  ep_rew=12.8909  rew_20=12.5691  L_geo=0.8283  L_rew=0.0006  L_chart=2.3457  L_crit=0.3003  L_bnd=0.7245  lr=0.0010  dt=18.32s
        recon=0.3506  vq=0.4493  code_H=1.3923  code_px=4.0511  ch_usage=0.4923  rtr_mrg=0.0141  enc_gn=12.0038
        ctrl=0.0001  tex=0.0353  im_rew=0.0354  im_ret=0.5165  value=0.0243  wm_gn=0.1153
        z_norm=0.5571  z_max=0.8563  jump=0.0000  cons=0.0643  sol=0.9940  e_var=0.0010  ch_ent=2.5074  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5165  dret=0.4953  term=0.0247  bnd=0.0429  chart_acc=0.1313  chart_ent=2.4172  rw_drift=0.0000
        v_err=1.5268  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6022
        bnd_x=0.0435  bell=0.0421  bell_s=0.0358  rtg_e=1.5268  rtg_b=-1.5268  cal_e=1.5268  u_l2=0.1186  cov_n=0.0009
        col=11.0816  smp=0.0036  enc_t=0.2524  bnd_t=0.0397  wm_t=1.0192  crt_t=0.0086  diag_t=0.5691
        charts: 16/16 active  c0=0.18 c1=0.14 c2=0.04 c3=0.15 c4=0.02 c5=0.05 c6=0.07 c7=0.03 c8=0.03 c9=0.08 c10=0.06 c11=0.02 c12=0.03 c13=0.03 c14=0.03 c15=0.03
        symbols: 48/128 active  c0=5/8(H=1.33) c1=5/8(H=1.38) c2=2/8(H=0.53) c3=5/8(H=1.39) c4=2/8(H=0.07) c5=2/8(H=0.64) c6=5/8(H=1.14) c7=1/8(H=0.00) c8=2/8(H=0.19) c9=3/8(H=0.96) c10=3/8(H=0.76) c11=3/8(H=0.17) c12=2/8(H=0.41) c13=2/8(H=0.31) c14=4/8(H=0.64) c15=2/8(H=0.20)
E0052 [5upd]  ep_rew=15.3048  rew_20=12.8893  L_geo=0.8018  L_rew=0.0007  L_chart=2.2811  L_crit=0.3195  L_bnd=0.8918  lr=0.0010  dt=18.46s
        recon=0.3631  vq=0.4371  code_H=1.3865  code_px=4.0457  ch_usage=0.4566  rtr_mrg=0.0085  enc_gn=12.8207
        ctrl=0.0001  tex=0.0347  im_rew=0.0368  im_ret=0.5346  value=0.0241  wm_gn=0.1469
        z_norm=0.5789  z_max=0.8159  jump=0.0000  cons=0.0748  sol=0.9927  e_var=0.0008  ch_ent=2.5912  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5346  dret=0.5144  term=0.0234  bnd=0.0392  chart_acc=0.1713  chart_ent=2.3334  rw_drift=0.0000
        v_err=1.5816  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1261
        bnd_x=0.0398  bell=0.0431  bell_s=0.0394  rtg_e=1.5816  rtg_b=-1.5816  cal_e=1.5816  u_l2=0.1198  cov_n=0.0011
        col=11.0661  smp=0.0027  enc_t=0.2540  bnd_t=0.0398  wm_t=1.0490  crt_t=0.0086  diag_t=0.5690
        charts: 16/16 active  c0=0.16 c1=0.11 c2=0.06 c3=0.09 c4=0.03 c5=0.06 c6=0.09 c7=0.05 c8=0.03 c9=0.11 c10=0.07 c11=0.02 c12=0.03 c13=0.03 c14=0.03 c15=0.03
        symbols: 56/128 active  c0=5/8(H=1.01) c1=5/8(H=1.39) c2=4/8(H=1.12) c3=7/8(H=1.44) c4=2/8(H=0.39) c5=4/8(H=0.43) c6=8/8(H=1.71) c7=1/8(H=0.00) c8=2/8(H=0.07) c9=5/8(H=1.53) c10=3/8(H=0.75) c11=2/8(H=0.53) c12=2/8(H=0.55) c13=1/8(H=0.00) c14=3/8(H=0.77) c15=2/8(H=0.45)
E0053 [5upd]  ep_rew=13.7696  rew_20=13.2915  L_geo=0.8486  L_rew=0.0005  L_chart=2.3725  L_crit=0.2420  L_bnd=0.9193  lr=0.0010  dt=18.32s
        recon=0.3281  vq=0.3654  code_H=1.2997  code_px=3.6858  ch_usage=0.5570  rtr_mrg=0.0083  enc_gn=11.8037
        ctrl=0.0001  tex=0.0330  im_rew=0.0327  im_ret=0.4768  value=0.0232  wm_gn=0.1250
        z_norm=0.5867  z_max=0.8535  jump=0.0000  cons=0.0820  sol=0.9912  e_var=0.0012  ch_ent=2.6396  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4768  dret=0.4569  term=0.0231  bnd=0.0435  chart_acc=0.1125  chart_ent=2.3864  rw_drift=0.0000
        v_err=1.1537  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2740
        bnd_x=0.0441  bell=0.0323  bell_s=0.0243  rtg_e=1.1537  rtg_b=-1.1536  cal_e=1.1536  u_l2=0.1370  cov_n=0.0007
        col=11.0799  smp=0.0028  enc_t=0.2548  bnd_t=0.0396  wm_t=1.0195  crt_t=0.0087  diag_t=0.5671
        charts: 16/16 active  c0=0.04 c1=0.08 c2=0.07 c3=0.09 c4=0.02 c5=0.09 c6=0.13 c7=0.06 c8=0.03 c9=0.11 c10=0.11 c11=0.03 c12=0.04 c13=0.04 c14=0.03 c15=0.03
        symbols: 56/128 active  c0=4/8(H=0.59) c1=6/8(H=0.73) c2=3/8(H=0.94) c3=4/8(H=0.90) c4=3/8(H=0.32) c5=4/8(H=0.87) c6=5/8(H=1.49) c7=3/8(H=0.46) c8=1/8(H=0.00) c9=3/8(H=0.98) c10=3/8(H=0.74) c11=3/8(H=0.43) c12=2/8(H=0.43) c13=3/8(H=0.53) c14=7/8(H=1.26) c15=2/8(H=0.28)
E0054 [5upd]  ep_rew=14.1489  rew_20=13.2838  L_geo=0.7897  L_rew=0.0004  L_chart=2.3051  L_crit=0.2697  L_bnd=0.9178  lr=0.0010  dt=18.77s
        recon=0.3422  vq=0.3303  code_H=1.2375  code_px=3.4630  ch_usage=0.4589  rtr_mrg=0.0114  enc_gn=11.5536
        ctrl=0.0001  tex=0.0338  im_rew=0.0332  im_ret=0.4832  value=0.0216  wm_gn=0.1330
        z_norm=0.5755  z_max=0.8141  jump=0.0000  cons=0.0658  sol=0.9975  e_var=0.0010  ch_ent=2.5234  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4832  dret=0.4645  term=0.0217  bnd=0.0402  chart_acc=0.1458  chart_ent=2.3209  rw_drift=0.0000
        v_err=1.1690  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3395
        bnd_x=0.0416  bell=0.0325  bell_s=0.0220  rtg_e=1.1690  rtg_b=-1.1690  cal_e=1.1690  u_l2=0.1456  cov_n=0.0008
        col=11.4990  smp=0.0037  enc_t=0.2536  bnd_t=0.0396  wm_t=1.0254  crt_t=0.0085  diag_t=0.5704
        charts: 16/16 active  c0=0.20 c1=0.11 c2=0.05 c3=0.10 c4=0.01 c5=0.07 c6=0.09 c7=0.04 c8=0.02 c9=0.08 c10=0.08 c11=0.02 c12=0.03 c13=0.04 c14=0.03 c15=0.03
        symbols: 49/128 active  c0=4/8(H=0.93) c1=4/8(H=0.56) c2=2/8(H=0.69) c3=6/8(H=1.21) c4=1/8(H=0.00) c5=3/8(H=0.94) c6=7/8(H=1.58) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=4/8(H=1.12) c10=3/8(H=0.75) c11=2/8(H=0.61) c12=2/8(H=0.37) c13=3/8(H=0.56) c14=3/8(H=0.93) c15=3/8(H=0.30)
E0055 [5upd]  ep_rew=14.8707  rew_20=13.8276  L_geo=0.8643  L_rew=0.0003  L_chart=2.3031  L_crit=0.2597  L_bnd=0.9175  lr=0.0010  dt=18.28s
        recon=0.3146  vq=0.3790  code_H=1.3078  code_px=3.7109  ch_usage=0.3495  rtr_mrg=0.0097  enc_gn=10.5886
        ctrl=0.0001  tex=0.0323  im_rew=0.0340  im_ret=0.4955  value=0.0231  wm_gn=0.1254
        z_norm=0.5902  z_max=0.8133  jump=0.0000  cons=0.1002  sol=0.9968  e_var=0.0008  ch_ent=2.6407  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4955  dret=0.4762  term=0.0225  bnd=0.0406  chart_acc=0.1025  chart_ent=2.3357  rw_drift=0.0000
        v_err=1.2904  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3115
        bnd_x=0.0417  bell=0.0359  bell_s=0.0213  rtg_e=1.2904  rtg_b=-1.2903  cal_e=1.2903  u_l2=0.1481  cov_n=0.0011
        col=11.0586  smp=0.0037  enc_t=0.2521  bnd_t=0.0397  wm_t=1.0163  crt_t=0.0086  diag_t=0.5722
        charts: 16/16 active  c0=0.06 c1=0.12 c2=0.07 c3=0.12 c4=0.03 c5=0.07 c6=0.07 c7=0.05 c8=0.03 c9=0.10 c10=0.10 c11=0.02 c12=0.04 c13=0.05 c14=0.03 c15=0.04
        symbols: 59/128 active  c0=4/8(H=1.05) c1=6/8(H=1.22) c2=3/8(H=0.88) c3=4/8(H=0.46) c4=3/8(H=0.50) c5=4/8(H=1.20) c6=7/8(H=1.41) c7=1/8(H=0.00) c8=2/8(H=0.14) c9=5/8(H=0.65) c10=4/8(H=0.72) c11=3/8(H=0.19) c12=2/8(H=0.51) c13=3/8(H=0.63) c14=6/8(H=0.69) c15=2/8(H=0.08)
E0056 [5upd]  ep_rew=25.6627  rew_20=14.2462  L_geo=0.7782  L_rew=0.0004  L_chart=2.2275  L_crit=0.2558  L_bnd=0.8661  lr=0.0010  dt=18.40s
        recon=0.3107  vq=0.4150  code_H=1.3136  code_px=3.7370  ch_usage=0.3419  rtr_mrg=0.0131  enc_gn=12.7530
        ctrl=0.0001  tex=0.0347  im_rew=0.0353  im_ret=0.5127  value=0.0221  wm_gn=0.1978
        z_norm=0.5493  z_max=0.8159  jump=0.0000  cons=0.0649  sol=0.9986  e_var=0.0008  ch_ent=2.5174  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5127  dret=0.4941  term=0.0215  bnd=0.0375  chart_acc=0.1617  chart_ent=2.2749  rw_drift=0.0000
        v_err=1.2036  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1836
        bnd_x=0.0377  bell=0.0327  bell_s=0.0130  rtg_e=1.2036  rtg_b=-1.2036  cal_e=1.2036  u_l2=0.1355  cov_n=0.0010
        col=11.1458  smp=0.0037  enc_t=0.2527  bnd_t=0.0400  wm_t=1.0185  crt_t=0.0086  diag_t=0.5838
        charts: 16/16 active  c0=0.20 c1=0.16 c2=0.04 c3=0.09 c4=0.02 c5=0.05 c6=0.04 c7=0.04 c8=0.03 c9=0.08 c10=0.08 c11=0.02 c12=0.03 c13=0.04 c14=0.03 c15=0.03
        symbols: 49/128 active  c0=4/8(H=1.26) c1=7/8(H=1.65) c2=3/8(H=0.70) c3=4/8(H=0.88) c4=3/8(H=0.48) c5=3/8(H=0.82) c6=3/8(H=0.84) c7=2/8(H=0.18) c8=2/8(H=0.24) c9=3/8(H=0.57) c10=3/8(H=0.79) c11=3/8(H=0.48) c12=2/8(H=0.32) c13=3/8(H=0.34) c14=2/8(H=0.53) c15=2/8(H=0.21)
E0057 [5upd]  ep_rew=41.6981  rew_20=14.9422  L_geo=0.7732  L_rew=0.0010  L_chart=2.2387  L_crit=0.2694  L_bnd=0.8001  lr=0.0010  dt=18.52s
        recon=0.2747  vq=0.5292  code_H=1.2515  code_px=3.5768  ch_usage=0.5502  rtr_mrg=0.0053  enc_gn=9.8413
        ctrl=0.0001  tex=0.0342  im_rew=0.0355  im_ret=0.5140  value=0.0198  wm_gn=0.2139
        z_norm=0.5411  z_max=0.8699  jump=0.0000  cons=0.0722  sol=0.9923  e_var=0.0013  ch_ent=2.3748  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5140  dret=0.4962  term=0.0207  bnd=0.0359  chart_acc=0.1342  chart_ent=2.1817  rw_drift=0.0000
        v_err=1.3267  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0011
        bnd_x=0.0376  bell=0.0384  bell_s=0.0476  rtg_e=1.3267  rtg_b=-1.3267  cal_e=1.3267  u_l2=0.1154  cov_n=0.0009
        col=11.3252  smp=0.0044  enc_t=0.2516  bnd_t=0.0397  wm_t=1.0113  crt_t=0.0087  diag_t=0.5689
        charts: 16/16 active  c0=0.18 c1=0.10 c2=0.03 c3=0.09 c4=0.01 c5=0.06 c6=0.17 c7=0.02 c8=0.02 c9=0.16 c10=0.07 c11=0.02 c12=0.02 c13=0.02 c14=0.01 c15=0.02
        symbols: 50/128 active  c0=6/8(H=1.33) c1=6/8(H=1.12) c2=2/8(H=0.69) c3=4/8(H=0.98) c4=2/8(H=0.51) c5=3/8(H=1.07) c6=6/8(H=1.49) c7=1/8(H=0.00) c8=1/8(H=0.00) c9=4/8(H=0.69) c10=4/8(H=1.04) c11=2/8(H=0.68) c12=2/8(H=0.27) c13=2/8(H=0.69) c14=3/8(H=0.90) c15=2/8(H=0.31)
E0058 [5upd]  ep_rew=17.1972  rew_20=14.8661  L_geo=0.8603  L_rew=0.0007  L_chart=2.4329  L_crit=0.3486  L_bnd=0.7109  lr=0.0010  dt=18.48s
        recon=0.3754  vq=0.3385  code_H=1.3949  code_px=4.1254  ch_usage=0.3785  rtr_mrg=0.0091  enc_gn=15.0280
        ctrl=0.0001  tex=0.0337  im_rew=0.0376  im_ret=0.5427  value=0.0204  wm_gn=0.1725
        z_norm=0.5548  z_max=0.8575  jump=0.0000  cons=0.0769  sol=0.9985  e_var=0.0009  ch_ent=2.5333  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5427  dret=0.5258  term=0.0196  bnd=0.0321  chart_acc=0.1042  chart_ent=2.3668  rw_drift=0.0000
        v_err=1.2772  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3111
        bnd_x=0.0330  bell=0.0346  bell_s=0.0156  rtg_e=1.2772  rtg_b=-1.2772  cal_e=1.2772  u_l2=0.1431  cov_n=0.0007
        col=11.2268  smp=0.0031  enc_t=0.2514  bnd_t=0.0400  wm_t=1.0217  crt_t=0.0086  diag_t=0.5733
        charts: 16/16 active  c0=0.05 c1=0.11 c2=0.04 c3=0.15 c4=0.03 c5=0.05 c6=0.08 c7=0.04 c8=0.03 c9=0.20 c10=0.07 c11=0.02 c12=0.03 c13=0.04 c14=0.03 c15=0.03
        symbols: 44/128 active  c0=2/8(H=0.68) c1=4/8(H=0.99) c2=3/8(H=0.71) c3=2/8(H=0.42) c4=1/8(H=0.00) c5=4/8(H=1.09) c6=5/8(H=1.11) c7=2/8(H=0.09) c8=1/8(H=0.00) c9=3/8(H=0.86) c10=3/8(H=1.04) c11=2/8(H=0.25) c12=2/8(H=0.54) c13=3/8(H=0.37) c14=5/8(H=0.78) c15=2/8(H=0.18)
E0059 [5upd]  ep_rew=8.5612  rew_20=14.3311  L_geo=0.7750  L_rew=0.0007  L_chart=2.2559  L_crit=0.2814  L_bnd=0.6600  lr=0.0010  dt=18.28s
        recon=0.3228  vq=0.3630  code_H=1.3536  code_px=3.9071  ch_usage=0.4174  rtr_mrg=0.0071  enc_gn=8.5351
        ctrl=0.0001  tex=0.0335  im_rew=0.0347  im_ret=0.5036  value=0.0212  wm_gn=0.2042
        z_norm=0.5823  z_max=0.8147  jump=0.0000  cons=0.0761  sol=0.9959  e_var=0.0014  ch_ent=2.5534  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5036  dret=0.4858  term=0.0208  bnd=0.0368  chart_acc=0.1767  chart_ent=2.2738  rw_drift=0.0000
        v_err=1.3165  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4194
        bnd_x=0.0382  bell=0.0365  bell_s=0.0310  rtg_e=1.3165  rtg_b=-1.3165  cal_e=1.3165  u_l2=0.1994  cov_n=0.0007
        col=11.0543  smp=0.0046  enc_t=0.2523  bnd_t=0.0397  wm_t=1.0179  crt_t=0.0085  diag_t=0.5665
        charts: 16/16 active  c0=0.17 c1=0.18 c2=0.05 c3=0.08 c4=0.02 c5=0.06 c6=0.07 c7=0.04 c8=0.02 c9=0.08 c10=0.07 c11=0.02 c12=0.03 c13=0.04 c14=0.04 c15=0.04
        symbols: 58/128 active  c0=5/8(H=1.16) c1=6/8(H=1.19) c2=4/8(H=0.98) c3=4/8(H=1.13) c4=3/8(H=0.75) c5=4/8(H=1.27) c6=6/8(H=1.53) c7=2/8(H=0.12) c8=1/8(H=0.00) c9=4/8(H=1.13) c10=3/8(H=0.95) c11=3/8(H=0.63) c12=2/8(H=0.43) c13=3/8(H=0.73) c14=5/8(H=1.07) c15=3/8(H=0.29)
E0060 [5upd]  ep_rew=14.3875  rew_20=14.2003  L_geo=0.9141  L_rew=0.0008  L_chart=2.3458  L_crit=0.2946  L_bnd=0.6448  lr=0.0010  dt=17.07s
        recon=0.3583  vq=0.3741  code_H=1.3188  code_px=3.7567  ch_usage=0.3983  rtr_mrg=0.0058  enc_gn=8.7333
        ctrl=0.0001  tex=0.0341  im_rew=0.0325  im_ret=0.4747  value=0.0235  wm_gn=0.1347
        z_norm=0.5792  z_max=0.8729  jump=0.0000  cons=0.0545  sol=0.9974  e_var=0.0007  ch_ent=2.5523  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4747  dret=0.4547  term=0.0232  bnd=0.0440  chart_acc=0.1629  chart_ent=2.3685  rw_drift=0.0000
        v_err=1.3340  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5634
        bnd_x=0.0449  bell=0.0371  bell_s=0.0314  rtg_e=1.3340  rtg_b=-1.3339  cal_e=1.3339  u_l2=0.1917  cov_n=0.0007
        col=11.0033  smp=0.0037  enc_t=0.2405  bnd_t=0.0376  wm_t=0.8451  crt_t=0.0075  diag_t=0.3546
        charts: 16/16 active  c0=0.16 c1=0.15 c2=0.04 c3=0.07 c4=0.02 c5=0.05 c6=0.08 c7=0.05 c8=0.02 c9=0.13 c10=0.07 c11=0.02 c12=0.04 c13=0.03 c14=0.04 c15=0.03
        symbols: 59/128 active  c0=5/8(H=1.24) c1=6/8(H=1.12) c2=3/8(H=0.98) c3=4/8(H=0.59) c4=3/8(H=0.77) c5=4/8(H=1.18) c6=5/8(H=1.19) c7=3/8(H=0.40) c8=1/8(H=0.00) c9=5/8(H=0.83) c10=3/8(H=1.02) c11=2/8(H=0.69) c12=2/8(H=0.65) c13=3/8(H=0.60) c14=7/8(H=1.40) c15=3/8(H=0.70)
  EVAL  reward=13.9 +/- 0.5  len=300
E0061 [5upd]  ep_rew=13.9616  rew_20=13.5801  L_geo=0.8400  L_rew=0.0012  L_chart=2.2436  L_crit=0.2968  L_bnd=0.6643  lr=0.0010  dt=12.43s
        recon=0.3222  vq=0.4291  code_H=1.3601  code_px=3.9057  ch_usage=0.4175  rtr_mrg=0.0055  enc_gn=8.6228
        ctrl=0.0001  tex=0.0323  im_rew=0.0375  im_ret=0.5494  value=0.0264  wm_gn=0.1292
        z_norm=0.5566  z_max=0.8978  jump=0.0000  cons=0.0573  sol=0.9972  e_var=0.0014  ch_ent=2.5401  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5494  dret=0.5249  term=0.0285  bnd=0.0468  chart_acc=0.1963  chart_ent=2.2819  rw_drift=0.0000
        v_err=1.3658  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3369
        bnd_x=0.0501  bell=0.0380  bell_s=0.0284  rtg_e=1.3658  rtg_b=-1.3657  cal_e=1.3657  u_l2=0.1542  cov_n=0.0012
        col=7.1566  smp=0.0027  enc_t=0.2225  bnd_t=0.0343  wm_t=0.7060  crt_t=0.0060  diag_t=0.3794
        charts: 16/16 active  c0=0.16 c1=0.07 c2=0.06 c3=0.10 c4=0.01 c5=0.07 c6=0.11 c7=0.03 c8=0.02 c9=0.14 c10=0.06 c11=0.02 c12=0.02 c13=0.03 c14=0.07 c15=0.03
        symbols: 67/128 active  c0=6/8(H=1.36) c1=8/8(H=1.58) c2=3/8(H=0.47) c3=4/8(H=1.04) c4=3/8(H=0.89) c5=4/8(H=1.15) c6=7/8(H=1.32) c7=3/8(H=0.59) c8=2/8(H=0.14) c9=5/8(H=1.36) c10=5/8(H=0.67) c11=4/8(H=1.05) c12=2/8(H=0.67) c13=3/8(H=0.79) c14=6/8(H=1.61) c15=2/8(H=0.66)
E0062 [5upd]  ep_rew=14.1625  rew_20=11.6780  L_geo=0.8443  L_rew=0.0007  L_chart=2.2822  L_crit=0.2701  L_bnd=0.7800  lr=0.0010  dt=19.19s
        recon=0.3374  vq=0.4043  code_H=1.3842  code_px=4.0133  ch_usage=0.3030  rtr_mrg=0.0064  enc_gn=7.9473
        ctrl=0.0001  tex=0.0336  im_rew=0.0391  im_ret=0.5716  value=0.0290  wm_gn=0.2041
        z_norm=0.5959  z_max=0.8381  jump=0.0000  cons=0.0649  sol=0.9950  e_var=0.0010  ch_ent=2.4520  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5716  dret=0.5470  term=0.0286  bnd=0.0450  chart_acc=0.1796  chart_ent=2.2495  rw_drift=0.0000
        v_err=1.2463  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1369
        bnd_x=0.0464  bell=0.0350  bell_s=0.0248  rtg_e=1.2463  rtg_b=-1.2462  cal_e=1.2462  u_l2=0.1050  cov_n=0.0012
        col=11.4285  smp=0.0045  enc_t=0.2597  bnd_t=0.0418  wm_t=1.1076  crt_t=0.0091  diag_t=0.6016
        charts: 16/16 active  c0=0.20 c1=0.13 c2=0.05 c3=0.15 c4=0.01 c5=0.08 c6=0.02 c7=0.03 c8=0.02 c9=0.07 c10=0.10 c11=0.02 c12=0.03 c13=0.04 c14=0.03 c15=0.03
        symbols: 58/128 active  c0=7/8(H=1.55) c1=4/8(H=0.78) c2=4/8(H=0.84) c3=5/8(H=1.41) c4=3/8(H=0.45) c5=5/8(H=0.73) c6=3/8(H=0.76) c7=3/8(H=0.26) c8=1/8(H=0.00) c9=3/8(H=0.66) c10=5/8(H=0.69) c11=2/8(H=0.46) c12=2/8(H=0.69) c13=4/8(H=0.83) c14=5/8(H=0.82) c15=2/8(H=0.46)
E0063 [5upd]  ep_rew=10.8264  rew_20=11.4490  L_geo=0.9065  L_rew=0.0011  L_chart=2.3433  L_crit=0.2663  L_bnd=0.7048  lr=0.0010  dt=18.73s
        recon=0.3149  vq=0.3346  code_H=1.3726  code_px=3.9609  ch_usage=0.3829  rtr_mrg=0.0049  enc_gn=7.5729
        ctrl=0.0001  tex=0.0335  im_rew=0.0375  im_ret=0.5472  value=0.0270  wm_gn=0.2011
        z_norm=0.5977  z_max=0.8441  jump=0.0000  cons=0.0764  sol=1.0005  e_var=0.0009  ch_ent=2.4736  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5472  dret=0.5242  term=0.0268  bnd=0.0439  chart_acc=0.1638  chart_ent=2.3627  rw_drift=0.0000
        v_err=1.4342  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0069
        bnd_x=0.0437  bell=0.0395  bell_s=0.0245  rtg_e=1.4342  rtg_b=-1.4342  cal_e=1.4342  u_l2=0.0910  cov_n=0.0011
        col=11.3861  smp=0.0027  enc_t=0.2543  bnd_t=0.0399  wm_t=1.0363  crt_t=0.0088  diag_t=0.5763
        charts: 16/16 active  c0=0.12 c1=0.09 c2=0.03 c3=0.08 c4=0.02 c5=0.04 c6=0.05 c7=0.03 c8=0.03 c9=0.26 c10=0.05 c11=0.04 c12=0.03 c13=0.04 c14=0.06 c15=0.05
        symbols: 73/128 active  c0=6/8(H=1.14) c1=6/8(H=1.45) c2=4/8(H=0.32) c3=5/8(H=1.14) c4=3/8(H=0.77) c5=5/8(H=0.45) c6=7/8(H=1.57) c7=3/8(H=0.22) c8=1/8(H=0.00) c9=5/8(H=1.01) c10=6/8(H=1.11) c11=5/8(H=1.03) c12=4/8(H=0.78) c13=3/8(H=0.59) c14=7/8(H=1.50) c15=3/8(H=0.63)
E0064 [5upd]  ep_rew=18.6454  rew_20=13.3788  L_geo=0.7934  L_rew=0.0005  L_chart=2.2014  L_crit=0.2573  L_bnd=0.6209  lr=0.0010  dt=18.40s
        recon=0.3144  vq=0.3444  code_H=1.2960  code_px=3.7030  ch_usage=0.3792  rtr_mrg=0.0062  enc_gn=7.9847
        ctrl=0.0001  tex=0.0348  im_rew=0.0346  im_ret=0.5011  value=0.0212  wm_gn=0.1724
        z_norm=0.5804  z_max=0.8260  jump=0.0000  cons=0.0742  sol=0.9940  e_var=0.0009  ch_ent=2.6057  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5011  dret=0.4841  term=0.0197  bnd=0.0351  chart_acc=0.1742  chart_ent=2.2822  rw_drift=0.0000
        v_err=1.1918  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4179
        bnd_x=0.0357  bell=0.0330  bell_s=0.0189  rtg_e=1.1918  rtg_b=-1.1918  cal_e=1.1918  u_l2=0.1114  cov_n=0.0006
        col=11.1169  smp=0.0030  enc_t=0.2527  bnd_t=0.0397  wm_t=1.0249  crt_t=0.0086  diag_t=0.5782
        charts: 16/16 active  c0=0.11 c1=0.07 c2=0.06 c3=0.16 c4=0.01 c5=0.06 c6=0.07 c7=0.05 c8=0.02 c9=0.08 c10=0.10 c11=0.03 c12=0.03 c13=0.04 c14=0.06 c15=0.04
        symbols: 64/128 active  c0=6/8(H=1.21) c1=4/8(H=0.44) c2=3/8(H=1.07) c3=5/8(H=1.27) c4=2/8(H=0.56) c5=5/8(H=1.21) c6=5/8(H=1.22) c7=4/8(H=0.59) c8=2/8(H=0.08) c9=4/8(H=1.13) c10=5/8(H=1.50) c11=4/8(H=1.10) c12=3/8(H=0.71) c13=3/8(H=0.75) c14=6/8(H=1.16) c15=3/8(H=0.67)
E0065 [5upd]  ep_rew=6.8725  rew_20=12.7504  L_geo=0.8626  L_rew=0.0007  L_chart=2.2326  L_crit=0.2927  L_bnd=0.7090  lr=0.0010  dt=18.31s
        recon=0.3082  vq=0.3649  code_H=1.3142  code_px=3.7615  ch_usage=0.3280  rtr_mrg=0.0052  enc_gn=9.1562
        ctrl=0.0001  tex=0.0328  im_rew=0.0363  im_ret=0.5233  value=0.0187  wm_gn=0.1656
        z_norm=0.6093  z_max=0.8380  jump=0.0000  cons=0.0717  sol=0.9998  e_var=0.0013  ch_ent=2.5128  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5233  dret=0.5077  term=0.0180  bnd=0.0306  chart_acc=0.2025  chart_ent=2.2122  rw_drift=0.0000
        v_err=1.2256  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1402
        bnd_x=0.0336  bell=0.0340  bell_s=0.0254  rtg_e=1.2256  rtg_b=-1.2255  cal_e=1.2255  u_l2=0.1466  cov_n=0.0004
        col=11.0799  smp=0.0036  enc_t=0.2532  bnd_t=0.0399  wm_t=1.0188  crt_t=0.0085  diag_t=0.5640
        charts: 16/16 active  c0=0.08 c1=0.04 c2=0.04 c3=0.23 c4=0.02 c5=0.05 c6=0.02 c7=0.04 c8=0.03 c9=0.11 c10=0.11 c11=0.03 c12=0.04 c13=0.06 c14=0.04 c15=0.08
        symbols: 61/128 active  c0=4/8(H=1.06) c1=2/8(H=0.58) c2=3/8(H=0.72) c3=6/8(H=1.32) c4=2/8(H=0.33) c5=4/8(H=0.96) c6=2/8(H=0.29) c7=4/8(H=0.72) c8=2/8(H=0.06) c9=3/8(H=0.98) c10=5/8(H=1.55) c11=5/8(H=0.75) c12=4/8(H=1.11) c13=3/8(H=0.78) c14=7/8(H=0.99) c15=5/8(H=1.18)
E0066 [5upd]  ep_rew=10.3949  rew_20=13.2934  L_geo=0.8389  L_rew=0.0006  L_chart=2.1936  L_crit=0.2789  L_bnd=0.7832  lr=0.0010  dt=16.46s
        recon=0.3056  vq=0.3379  code_H=1.3273  code_px=3.7758  ch_usage=0.3316  rtr_mrg=0.0054  enc_gn=9.9170
        ctrl=0.0001  tex=0.0315  im_rew=0.0403  im_ret=0.5806  value=0.0196  wm_gn=0.2387
        z_norm=0.5989  z_max=0.8738  jump=0.0000  cons=0.0502  sol=0.9973  e_var=0.0015  ch_ent=2.5101  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5806  dret=0.5634  term=0.0200  bnd=0.0305  chart_acc=0.1708  chart_ent=2.1809  rw_drift=0.0000
        v_err=1.3242  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2562
        bnd_x=0.0318  bell=0.0361  bell_s=0.0262  rtg_e=1.3242  rtg_b=-1.3241  cal_e=1.3241  u_l2=0.1756  cov_n=0.0007
        col=11.2216  smp=0.0023  enc_t=0.2308  bnd_t=0.0356  wm_t=0.6892  crt_t=0.0060  diag_t=0.3691
        charts: 16/16 active  c0=0.18 c1=0.17 c2=0.03 c3=0.05 c4=0.02 c5=0.04 c6=0.10 c7=0.04 c8=0.02 c9=0.08 c10=0.09 c11=0.02 c12=0.03 c13=0.04 c14=0.04 c15=0.03
        symbols: 69/128 active  c0=5/8(H=0.92) c1=6/8(H=1.31) c2=5/8(H=0.80) c3=3/8(H=0.70) c4=4/8(H=1.12) c5=3/8(H=0.79) c6=6/8(H=1.34) c7=5/8(H=0.49) c8=3/8(H=0.39) c9=4/8(H=1.08) c10=5/8(H=1.31) c11=4/8(H=1.14) c12=2/8(H=0.57) c13=4/8(H=0.88) c14=7/8(H=1.20) c15=3/8(H=0.57)
E0067 [5upd]  ep_rew=6.6921  rew_20=13.2687  L_geo=0.7501  L_rew=0.0008  L_chart=2.1278  L_crit=0.2797  L_bnd=0.7749  lr=0.0010  dt=12.76s
        recon=0.2809  vq=0.3903  code_H=1.3631  code_px=3.9091  ch_usage=0.1996  rtr_mrg=0.0071  enc_gn=7.7704
        ctrl=0.0001  tex=0.0325  im_rew=0.0357  im_ret=0.5186  value=0.0222  wm_gn=0.2494
        z_norm=0.6081  z_max=0.9046  jump=0.0000  cons=0.0574  sol=0.9983  e_var=0.0015  ch_ent=2.5761  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5186  dret=0.4988  term=0.0231  bnd=0.0398  chart_acc=0.2138  chart_ent=2.1383  rw_drift=0.0000
        v_err=1.1922  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1817
        bnd_x=0.0418  bell=0.0337  bell_s=0.0238  rtg_e=1.1922  rtg_b=-1.1921  cal_e=1.1921  u_l2=0.1393  cov_n=0.0008
        col=7.1849  smp=0.0026  enc_t=0.2252  bnd_t=0.0343  wm_t=0.7550  crt_t=0.0062  diag_t=0.4153
        charts: 16/16 active  c0=0.16 c1=0.05 c2=0.06 c3=0.13 c4=0.01 c5=0.08 c6=0.07 c7=0.03 c8=0.02 c9=0.11 c10=0.09 c11=0.04 c12=0.02 c13=0.05 c14=0.05 c15=0.03
        symbols: 68/128 active  c0=7/8(H=0.74) c1=8/8(H=1.35) c2=4/8(H=0.87) c3=6/8(H=1.60) c4=3/8(H=0.85) c5=5/8(H=1.13) c6=8/8(H=1.60) c7=3/8(H=0.61) c8=2/8(H=0.27) c9=3/8(H=0.93) c10=4/8(H=1.20) c11=3/8(H=1.07) c12=2/8(H=0.64) c13=3/8(H=1.09) c14=4/8(H=1.03) c15=3/8(H=0.69)
E0068 [5upd]  ep_rew=14.1617  rew_20=13.4968  L_geo=0.7417  L_rew=0.0009  L_chart=2.0997  L_crit=0.3108  L_bnd=0.6876  lr=0.0010  dt=13.85s
        recon=0.2911  vq=0.3700  code_H=1.4329  code_px=4.2192  ch_usage=0.2270  rtr_mrg=0.0051  enc_gn=8.2873
        ctrl=0.0001  tex=0.0319  im_rew=0.0329  im_ret=0.4833  value=0.0263  wm_gn=0.2233
        z_norm=0.6220  z_max=0.8816  jump=0.0000  cons=0.0681  sol=0.9931  e_var=0.0018  ch_ent=2.5113  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4833  dret=0.4603  term=0.0267  bnd=0.0499  chart_acc=0.2350  chart_ent=2.1113  rw_drift=0.0000
        v_err=1.3574  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1798
        bnd_x=0.0529  bell=0.0394  bell_s=0.0344  rtg_e=1.3574  rtg_b=-1.3573  cal_e=1.3573  u_l2=0.1454  cov_n=0.0010
        col=7.4546  smp=0.0031  enc_t=0.2327  bnd_t=0.0368  wm_t=0.8598  crt_t=0.0074  diag_t=0.6491
        charts: 16/16 active  c0=0.09 c1=0.19 c2=0.05 c3=0.10 c4=0.02 c5=0.08 c6=0.07 c7=0.04 c8=0.02 c9=0.09 c10=0.13 c11=0.03 c12=0.02 c13=0.04 c14=0.02 c15=0.03
        symbols: 73/128 active  c0=7/8(H=1.45) c1=8/8(H=1.68) c2=4/8(H=0.87) c3=6/8(H=1.41) c4=2/8(H=0.68) c5=5/8(H=1.25) c6=7/8(H=1.37) c7=3/8(H=0.25) c8=6/8(H=0.93) c9=4/8(H=1.22) c10=5/8(H=1.37) c11=4/8(H=1.07) c12=2/8(H=0.58) c13=4/8(H=0.82) c14=3/8(H=0.48) c15=3/8(H=0.61)
E0069 [5upd]  ep_rew=15.3280  rew_20=11.8173  L_geo=0.8714  L_rew=0.0013  L_chart=2.1175  L_crit=0.2621  L_bnd=0.7003  lr=0.0010  dt=17.65s
        recon=0.2755  vq=0.3515  code_H=1.4082  code_px=4.1012  ch_usage=0.3157  rtr_mrg=0.0062  enc_gn=5.0471
        ctrl=0.0001  tex=0.0296  im_rew=0.0356  im_ret=0.5215  value=0.0280  wm_gn=0.1865
        z_norm=0.6344  z_max=0.9033  jump=0.0000  cons=0.0690  sol=0.9881  e_var=0.0027  ch_ent=2.4490  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5215  dret=0.4976  term=0.0277  bnd=0.0479  chart_acc=0.2504  chart_ent=2.1939  rw_drift=0.0000
        v_err=1.1523  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0427
        bnd_x=0.0508  bell=0.0357  bell_s=0.0678  rtg_e=1.1523  rtg_b=-1.1522  cal_e=1.1522  u_l2=0.1863  cov_n=0.0011
        col=12.2015  smp=0.0031  enc_t=0.2350  bnd_t=0.0364  wm_t=0.7226  crt_t=0.0063  diag_t=0.3881
        charts: 16/16 active  c0=0.26 c1=0.09 c2=0.04 c3=0.09 c4=0.01 c5=0.06 c6=0.06 c7=0.03 c8=0.02 c9=0.08 c10=0.11 c11=0.02 c12=0.03 c13=0.04 c14=0.03 c15=0.04
        symbols: 53/128 active  c0=7/8(H=1.34) c1=4/8(H=1.28) c2=4/8(H=0.96) c3=5/8(H=1.34) c4=3/8(H=0.41) c5=4/8(H=1.00) c6=4/8(H=1.28) c7=3/8(H=0.85) c8=1/8(H=0.00) c9=3/8(H=0.95) c10=4/8(H=1.18) c11=2/8(H=0.64) c12=2/8(H=0.69) c13=3/8(H=0.85) c14=2/8(H=0.69) c15=2/8(H=0.60)
E0070 [5upd]  ep_rew=14.0349  rew_20=12.6744  L_geo=0.8777  L_rew=0.0007  L_chart=2.1590  L_crit=0.2442  L_bnd=0.7475  lr=0.0010  dt=12.65s
        recon=0.2478  vq=0.3434  code_H=1.1956  code_px=3.3262  ch_usage=0.2832  rtr_mrg=0.0059  enc_gn=8.5571
        ctrl=0.0001  tex=0.0311  im_rew=0.0376  im_ret=0.5493  value=0.0272  wm_gn=0.2219
        z_norm=0.6237  z_max=0.9010  jump=0.0000  cons=0.0506  sol=0.9947  e_var=0.0018  ch_ent=2.5585  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5493  dret=0.5255  term=0.0277  bnd=0.0453  chart_acc=0.2025  chart_ent=2.1501  rw_drift=0.0000
        v_err=1.4292  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1713
        bnd_x=0.0473  bell=0.0393  bell_s=0.0418  rtg_e=1.4292  rtg_b=-1.4291  cal_e=1.4291  u_l2=0.1677  cov_n=0.0010
        col=7.3064  smp=0.0030  enc_t=0.2226  bnd_t=0.0345  wm_t=0.7197  crt_t=0.0058  diag_t=0.3808
        charts: 16/16 active  c0=0.15 c1=0.07 c2=0.08 c3=0.10 c4=0.02 c5=0.07 c6=0.09 c7=0.04 c8=0.02 c9=0.14 c10=0.07 c11=0.02 c12=0.02 c13=0.02 c14=0.06 c15=0.03
        symbols: 68/128 active  c0=6/8(H=1.41) c1=7/8(H=1.47) c2=5/8(H=1.22) c3=3/8(H=0.34) c4=4/8(H=0.93) c5=4/8(H=0.77) c6=8/8(H=1.17) c7=5/8(H=0.99) c8=2/8(H=0.08) c9=5/8(H=0.52) c10=4/8(H=0.97) c11=3/8(H=0.74) c12=2/8(H=0.66) c13=3/8(H=0.78) c14=5/8(H=1.26) c15=2/8(H=0.60)
E0071 [5upd]  ep_rew=6.9808  rew_20=11.9296  L_geo=0.8590  L_rew=0.0011  L_chart=2.1803  L_crit=0.3574  L_bnd=0.7543  lr=0.0010  dt=12.86s
        recon=0.3175  vq=0.3150  code_H=1.4049  code_px=4.0913  ch_usage=0.1292  rtr_mrg=0.0072  enc_gn=6.9182
        ctrl=0.0001  tex=0.0299  im_rew=0.0401  im_ret=0.5823  value=0.0253  wm_gn=0.2364
        z_norm=0.6347  z_max=0.9641  jump=0.0000  cons=0.0477  sol=0.9992  e_var=0.0022  ch_ent=2.5319  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5823  dret=0.5612  term=0.0245  bnd=0.0376  chart_acc=0.2542  chart_ent=2.1645  rw_drift=0.0000
        v_err=1.4819  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1422
        bnd_x=0.0378  bell=0.0408  bell_s=0.0303  rtg_e=1.4819  rtg_b=-1.4819  cal_e=1.4819  u_l2=0.1242  cov_n=0.0010
        col=7.3855  smp=0.0034  enc_t=0.2277  bnd_t=0.0346  wm_t=0.7322  crt_t=0.0065  diag_t=0.4151
        charts: 16/16 active  c0=0.14 c1=0.14 c2=0.04 c3=0.07 c4=0.02 c5=0.07 c6=0.04 c7=0.02 c8=0.02 c9=0.17 c10=0.04 c11=0.06 c12=0.03 c13=0.03 c14=0.07 c15=0.03
        symbols: 82/128 active  c0=7/8(H=1.41) c1=8/8(H=1.39) c2=4/8(H=1.13) c3=5/8(H=1.11) c4=4/8(H=1.19) c5=4/8(H=0.86) c6=8/8(H=1.84) c7=4/8(H=0.79) c8=3/8(H=0.72) c9=5/8(H=1.26) c10=4/8(H=0.98) c11=7/8(H=1.38) c12=3/8(H=0.74) c13=5/8(H=1.34) c14=8/8(H=1.54) c15=3/8(H=1.07)
E0072 [5upd]  ep_rew=7.6061  rew_20=11.8942  L_geo=0.8539  L_rew=0.0008  L_chart=2.1066  L_crit=0.2420  L_bnd=0.7743  lr=0.0010  dt=13.15s
        recon=0.2673  vq=0.3056  code_H=1.2583  code_px=3.5454  ch_usage=0.1628  rtr_mrg=0.0040  enc_gn=8.4336
        ctrl=0.0001  tex=0.0305  im_rew=0.0368  im_ret=0.5347  value=0.0248  wm_gn=0.1650
        z_norm=0.6535  z_max=0.8389  jump=0.0000  cons=0.0424  sol=0.9994  e_var=0.0023  ch_ent=2.6542  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5347  dret=0.5149  term=0.0230  bnd=0.0385  chart_acc=0.2604  chart_ent=2.1518  rw_drift=0.0000
        v_err=1.2512  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2574
        bnd_x=0.0396  bell=0.0363  bell_s=0.0419  rtg_e=1.2512  rtg_b=-1.2511  cal_e=1.2511  u_l2=0.1166  cov_n=0.0008
        col=7.5623  smp=0.0023  enc_t=0.2291  bnd_t=0.0346  wm_t=0.7513  crt_t=0.0064  diag_t=0.4233
        charts: 16/16 active  c0=0.12 c1=0.10 c2=0.06 c3=0.05 c4=0.02 c5=0.10 c6=0.08 c7=0.04 c8=0.03 c9=0.08 c10=0.09 c11=0.04 c12=0.02 c13=0.07 c14=0.07 c15=0.05
        symbols: 77/128 active  c0=7/8(H=1.56) c1=5/8(H=0.38) c2=4/8(H=0.69) c3=3/8(H=0.88) c4=4/8(H=0.51) c5=6/8(H=1.25) c6=5/8(H=1.21) c7=3/8(H=0.46) c8=3/8(H=0.41) c9=6/8(H=0.69) c10=4/8(H=0.95) c11=5/8(H=1.22) c12=2/8(H=0.69) c13=7/8(H=1.39) c14=8/8(H=1.59) c15=5/8(H=1.26)
E0073 [5upd]  ep_rew=8.3491  rew_20=11.7876  L_geo=0.9284  L_rew=0.0007  L_chart=2.1719  L_crit=0.2882  L_bnd=0.8325  lr=0.0010  dt=13.24s
        recon=0.3104  vq=0.2875  code_H=1.2799  code_px=3.6262  ch_usage=0.2107  rtr_mrg=0.0050  enc_gn=7.3317
        ctrl=0.0001  tex=0.0307  im_rew=0.0387  im_ret=0.5604  value=0.0225  wm_gn=0.2520
        z_norm=0.6353  z_max=0.9422  jump=0.0000  cons=0.0767  sol=0.9864  e_var=0.0016  ch_ent=2.6483  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5604  dret=0.5419  term=0.0216  bnd=0.0342  chart_acc=0.2550  chart_ent=2.2023  rw_drift=0.0000
        v_err=1.4508  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.4279
        bnd_x=0.0356  bell=0.0404  bell_s=0.0351  rtg_e=1.4508  rtg_b=-1.4508  cal_e=1.4508  u_l2=0.1245  cov_n=0.0008
        col=7.7091  smp=0.0015  enc_t=0.2300  bnd_t=0.0350  wm_t=0.7468  crt_t=0.0066  diag_t=0.3875
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.05 c3=0.16 c4=0.02 c5=0.05 c6=0.09 c7=0.04 c8=0.03 c9=0.07 c10=0.08 c11=0.07 c12=0.03 c13=0.06 c14=0.10 c15=0.05
        symbols: 87/128 active  c0=6/8(H=1.07) c1=6/8(H=1.31) c2=6/8(H=1.46) c3=4/8(H=1.07) c4=3/8(H=0.44) c5=4/8(H=1.02) c6=8/8(H=1.56) c7=4/8(H=0.98) c8=5/8(H=0.70) c9=6/8(H=1.34) c10=6/8(H=1.27) c11=6/8(H=1.29) c12=4/8(H=1.17) c13=7/8(H=1.49) c14=7/8(H=1.30) c15=5/8(H=1.29)
E0074 [5upd]  ep_rew=9.6893  rew_20=12.4505  L_geo=0.8689  L_rew=0.0008  L_chart=2.1539  L_crit=0.3053  L_bnd=0.7427  lr=0.0010  dt=17.99s
        recon=0.2829  vq=0.2988  code_H=1.3186  code_px=3.7545  ch_usage=0.2279  rtr_mrg=0.0038  enc_gn=9.2443
        ctrl=0.0001  tex=0.0292  im_rew=0.0379  im_ret=0.5490  value=0.0212  wm_gn=0.2249
        z_norm=0.6411  z_max=0.9111  jump=0.0000  cons=0.0586  sol=0.9962  e_var=0.0025  ch_ent=2.5641  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5490  dret=0.5306  term=0.0214  bnd=0.0347  chart_acc=0.2275  chart_ent=2.1720  rw_drift=0.0000
        v_err=1.3238  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1459
        bnd_x=0.0375  bell=0.0363  bell_s=0.0220  rtg_e=1.3238  rtg_b=-1.3238  cal_e=1.3238  u_l2=0.1376  cov_n=0.0010
        col=9.8504  smp=0.0043  enc_t=0.2670  bnd_t=0.0427  wm_t=1.1679  crt_t=0.0100  diag_t=0.6270
        charts: 16/16 active  c0=0.02 c1=0.11 c2=0.07 c3=0.07 c4=0.02 c5=0.05 c6=0.15 c7=0.03 c8=0.02 c9=0.17 c10=0.07 c11=0.03 c12=0.03 c13=0.04 c14=0.07 c15=0.05
        symbols: 62/128 active  c0=3/8(H=1.00) c1=6/8(H=1.27) c2=4/8(H=1.10) c3=3/8(H=1.02) c4=3/8(H=1.02) c5=4/8(H=1.08) c6=5/8(H=1.32) c7=4/8(H=0.94) c8=1/8(H=0.00) c9=6/8(H=1.41) c10=5/8(H=1.31) c11=3/8(H=1.00) c12=4/8(H=1.10) c13=3/8(H=1.03) c14=6/8(H=1.14) c15=2/8(H=0.69)
E0075 [5upd]  ep_rew=10.4625  rew_20=11.8026  L_geo=0.8399  L_rew=0.0008  L_chart=2.0355  L_crit=0.2600  L_bnd=0.5615  lr=0.0010  dt=19.20s
        recon=0.2558  vq=0.3881  code_H=1.2302  code_px=3.4313  ch_usage=0.1569  rtr_mrg=0.0037  enc_gn=8.6028
        ctrl=0.0001  tex=0.0306  im_rew=0.0383  im_ret=0.5529  value=0.0211  wm_gn=0.3252
        z_norm=0.6298  z_max=0.9490  jump=0.0000  cons=0.0647  sol=1.0001  e_var=0.0019  ch_ent=2.5094  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5529  dret=0.5360  term=0.0196  bnd=0.0315  chart_acc=0.2054  chart_ent=2.1014  rw_drift=0.0000
        v_err=1.3255  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6241
        bnd_x=0.0333  bell=0.0361  bell_s=0.0229  rtg_e=1.3255  rtg_b=-1.3255  cal_e=1.3255  u_l2=0.1278  cov_n=0.0008
        col=11.7823  smp=0.0049  enc_t=0.2552  bnd_t=0.0404  wm_t=1.0490  crt_t=0.0087  diag_t=0.5856
        charts: 16/16 active  c0=0.08 c1=0.03 c2=0.08 c3=0.19 c4=0.01 c5=0.13 c6=0.14 c7=0.02 c8=0.03 c9=0.05 c10=0.05 c11=0.04 c12=0.02 c13=0.05 c14=0.06 c15=0.03
        symbols: 80/128 active  c0=7/8(H=1.35) c1=6/8(H=1.03) c2=4/8(H=0.97) c3=7/8(H=1.41) c4=3/8(H=0.92) c5=5/8(H=1.11) c6=8/8(H=1.39) c7=3/8(H=0.80) c8=5/8(H=1.16) c9=4/8(H=0.79) c10=4/8(H=1.31) c11=6/8(H=1.69) c12=3/8(H=0.70) c13=5/8(H=1.38) c14=7/8(H=1.60) c15=3/8(H=1.10)
E0076 [5upd]  ep_rew=15.7847  rew_20=12.4125  L_geo=0.8676  L_rew=0.0004  L_chart=2.1177  L_crit=0.2461  L_bnd=0.4858  lr=0.0010  dt=18.84s
        recon=0.2496  vq=0.3284  code_H=1.3337  code_px=3.8307  ch_usage=0.1431  rtr_mrg=0.0041  enc_gn=9.3129
        ctrl=0.0001  tex=0.0307  im_rew=0.0317  im_ret=0.4625  value=0.0216  wm_gn=0.2360
        z_norm=0.6211  z_max=0.8933  jump=0.0000  cons=0.0710  sol=0.9909  e_var=0.0011  ch_ent=2.6264  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4625  dret=0.4430  term=0.0227  bnd=0.0440  chart_acc=0.2425  chart_ent=2.1132  rw_drift=0.0000
        v_err=1.2594  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8192
        bnd_x=0.0444  bell=0.0356  bell_s=0.0289  rtg_e=1.2594  rtg_b=-1.2594  cal_e=1.2594  u_l2=0.1433  cov_n=0.0011
        col=11.4620  smp=0.0028  enc_t=0.2558  bnd_t=0.0399  wm_t=1.0402  crt_t=0.0088  diag_t=0.5890
        charts: 16/16 active  c0=0.16 c1=0.07 c2=0.05 c3=0.12 c4=0.03 c5=0.04 c6=0.08 c7=0.02 c8=0.03 c9=0.08 c10=0.05 c11=0.06 c12=0.02 c13=0.08 c14=0.07 c15=0.04
        symbols: 79/128 active  c0=5/8(H=1.34) c1=8/8(H=1.56) c2=3/8(H=1.09) c3=7/8(H=1.42) c4=5/8(H=1.27) c5=4/8(H=1.27) c6=7/8(H=1.34) c7=4/8(H=0.83) c8=3/8(H=0.68) c9=7/8(H=1.53) c10=4/8(H=1.29) c11=5/8(H=1.25) c12=3/8(H=0.79) c13=5/8(H=1.29) c14=6/8(H=1.73) c15=3/8(H=1.03)
E0077 [5upd]  ep_rew=14.2387  rew_20=13.3000  L_geo=0.7877  L_rew=0.0008  L_chart=1.9172  L_crit=0.3222  L_bnd=0.5033  lr=0.0010  dt=18.84s
        recon=0.2511  vq=0.3765  code_H=1.1710  code_px=3.2538  ch_usage=0.1896  rtr_mrg=0.0031  enc_gn=7.5325
        ctrl=0.0001  tex=0.0283  im_rew=0.0368  im_ret=0.5344  value=0.0229  wm_gn=0.2276
        z_norm=0.6504  z_max=0.9035  jump=0.0000  cons=0.0647  sol=0.9952  e_var=0.0024  ch_ent=2.5653  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5344  dret=0.5144  term=0.0232  bnd=0.0388  chart_acc=0.3117  chart_ent=2.0315  rw_drift=0.0000
        v_err=1.2304  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8152
        bnd_x=0.0420  bell=0.0337  bell_s=0.0159  rtg_e=1.2304  rtg_b=-1.2303  cal_e=1.2303  u_l2=0.1606  cov_n=0.0013
        col=11.2994  smp=0.0030  enc_t=0.2545  bnd_t=0.0399  wm_t=1.0737  crt_t=0.0088  diag_t=0.5933
        charts: 16/16 active  c0=0.13 c1=0.09 c2=0.04 c3=0.16 c4=0.02 c5=0.04 c6=0.07 c7=0.02 c8=0.01 c9=0.11 c10=0.07 c11=0.05 c12=0.02 c13=0.07 c14=0.04 c15=0.06
        symbols: 82/128 active  c0=6/8(H=1.31) c1=7/8(H=1.51) c2=5/8(H=1.09) c3=8/8(H=1.14) c4=4/8(H=1.10) c5=4/8(H=0.89) c6=6/8(H=1.38) c7=5/8(H=1.50) c8=3/8(H=0.69) c9=4/8(H=0.47) c10=5/8(H=1.38) c11=5/8(H=1.28) c12=4/8(H=1.17) c13=6/8(H=1.34) c14=6/8(H=1.56) c15=4/8(H=1.11)
E0078 [5upd]  ep_rew=14.3082  rew_20=13.7117  L_geo=0.8735  L_rew=0.0007  L_chart=2.0048  L_crit=0.3201  L_bnd=0.5415  lr=0.0010  dt=18.60s
        recon=0.2851  vq=0.2723  code_H=1.2336  code_px=3.4466  ch_usage=0.1733  rtr_mrg=0.0026  enc_gn=7.8170
        ctrl=0.0001  tex=0.0285  im_rew=0.0443  im_ret=0.6410  value=0.0242  wm_gn=0.1975
        z_norm=0.6681  z_max=0.8583  jump=0.0000  cons=0.0654  sol=0.9921  e_var=0.0017  ch_ent=2.4781  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6410  dret=0.6203  term=0.0240  bnd=0.0333  chart_acc=0.2854  chart_ent=2.1693  rw_drift=0.0000
        v_err=1.5823  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7065
        bnd_x=0.0366  bell=0.0427  bell_s=0.0318  rtg_e=1.5823  rtg_b=-1.5823  cal_e=1.5823  u_l2=0.1309  cov_n=0.0013
        col=11.2331  smp=0.0034  enc_t=0.2542  bnd_t=0.0402  wm_t=1.0414  crt_t=0.0088  diag_t=0.5766
        charts: 16/16 active  c0=0.04 c1=0.15 c2=0.03 c3=0.14 c4=0.02 c5=0.02 c6=0.02 c7=0.03 c8=0.02 c9=0.18 c10=0.08 c11=0.03 c12=0.03 c13=0.04 c14=0.11 c15=0.06
        symbols: 101/128 active  c0=5/8(H=0.80) c1=7/8(H=1.09) c2=3/8(H=0.86) c3=7/8(H=1.31) c4=4/8(H=1.12) c5=4/8(H=0.80) c6=7/8(H=1.63) c7=7/8(H=1.62) c8=6/8(H=0.58) c9=8/8(H=1.15) c10=6/8(H=1.24) c11=7/8(H=1.29) c12=7/8(H=1.41) c13=7/8(H=1.62) c14=8/8(H=1.73) c15=8/8(H=1.70)
E0079 [5upd]  ep_rew=6.6964  rew_20=12.5069  L_geo=0.7479  L_rew=0.0006  L_chart=1.8427  L_crit=0.2580  L_bnd=0.5025  lr=0.0010  dt=19.03s
        recon=0.2136  vq=0.3659  code_H=1.1949  code_px=3.3127  ch_usage=0.2034  rtr_mrg=0.0027  enc_gn=7.8052
        ctrl=0.0001  tex=0.0293  im_rew=0.0374  im_ret=0.5407  value=0.0209  wm_gn=0.2744
        z_norm=0.6670  z_max=0.9231  jump=0.0000  cons=0.0305  sol=0.9990  e_var=0.0026  ch_ent=2.3040  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5407  dret=0.5228  term=0.0208  bnd=0.0343  chart_acc=0.3667  chart_ent=1.9603  rw_drift=0.0000
        v_err=1.2666  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6392
        bnd_x=0.0358  bell=0.0348  bell_s=0.0326  rtg_e=1.2666  rtg_b=-1.2666  cal_e=1.2666  u_l2=0.0915  cov_n=0.0008
        col=11.7327  smp=0.0031  enc_t=0.2540  bnd_t=0.0399  wm_t=1.0297  crt_t=0.0088  diag_t=0.5720
        charts: 16/16 active  c0=0.28 c1=0.01 c2=0.10 c3=0.07 c4=0.01 c5=0.10 c6=0.04 c7=0.03 c8=0.01 c9=0.13 c10=0.05 c11=0.06 c12=0.02 c13=0.01 c14=0.08 c15=0.02
        symbols: 88/128 active  c0=8/8(H=1.47) c1=3/8(H=0.85) c2=7/8(H=1.20) c3=7/8(H=1.25) c4=3/8(H=0.51) c5=7/8(H=0.66) c6=4/8(H=0.67) c7=4/8(H=0.79) c8=7/8(H=1.27) c9=5/8(H=1.50) c10=6/8(H=1.27) c11=6/8(H=0.96) c12=3/8(H=1.02) c13=6/8(H=1.54) c14=7/8(H=1.14) c15=5/8(H=1.08)
E0080 [5upd]  ep_rew=14.5785  rew_20=13.2263  L_geo=0.7913  L_rew=0.0007  L_chart=1.9727  L_crit=0.3569  L_bnd=0.5346  lr=0.0010  dt=18.46s
        recon=0.3089  vq=0.3317  code_H=1.2536  code_px=3.5187  ch_usage=0.1690  rtr_mrg=0.0037  enc_gn=7.7289
        ctrl=0.0001  tex=0.0302  im_rew=0.0370  im_ret=0.5354  value=0.0202  wm_gn=0.2842
        z_norm=0.6580  z_max=0.9744  jump=0.0000  cons=0.0573  sol=1.0017  e_var=0.0015  ch_ent=2.5084  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5354  dret=0.5180  term=0.0202  bnd=0.0336  chart_acc=0.2988  chart_ent=1.8660  rw_drift=0.0000
        v_err=1.5994  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5669
        bnd_x=0.0352  bell=0.0436  bell_s=0.0360  rtg_e=1.5994  rtg_b=-1.5994  cal_e=1.5994  u_l2=0.0569  cov_n=0.0006
        col=11.1655  smp=0.0038  enc_t=0.2535  bnd_t=0.0396  wm_t=1.0267  crt_t=0.0087  diag_t=0.5805
        charts: 16/16 active  c0=0.09 c1=0.08 c2=0.08 c3=0.19 c4=0.05 c5=0.16 c6=0.06 c7=0.02 c8=0.02 c9=0.07 c10=0.03 c11=0.05 c12=0.02 c13=0.03 c14=0.02 c15=0.02
        symbols: 92/128 active  c0=7/8(H=1.23) c1=8/8(H=1.46) c2=7/8(H=1.05) c3=7/8(H=1.36) c4=5/8(H=1.09) c5=7/8(H=1.00) c6=7/8(H=1.56) c7=4/8(H=1.06) c8=4/8(H=1.33) c9=4/8(H=0.78) c10=4/8(H=1.29) c11=8/8(H=1.59) c12=3/8(H=1.06) c13=6/8(H=1.20) c14=7/8(H=1.33) c15=4/8(H=1.09)
E0081 [5upd]  ep_rew=14.5017  rew_20=13.4665  L_geo=0.7328  L_rew=0.0010  L_chart=1.8084  L_crit=0.2572  L_bnd=0.5747  lr=0.0010  dt=18.54s
        recon=0.2183  vq=0.3798  code_H=1.2436  code_px=3.4905  ch_usage=0.1735  rtr_mrg=0.0010  enc_gn=7.3293
        ctrl=0.0001  tex=0.0264  im_rew=0.0389  im_ret=0.5650  value=0.0237  wm_gn=0.2148
        z_norm=0.6831  z_max=0.8583  jump=0.0000  cons=0.0820  sol=0.9889  e_var=0.0013  ch_ent=2.4602  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5650  dret=0.5439  term=0.0245  bnd=0.0388  chart_acc=0.4246  chart_ent=1.8159  rw_drift=0.0000
        v_err=1.2666  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2867
        bnd_x=0.0397  bell=0.0346  bell_s=0.0203  rtg_e=1.2666  rtg_b=-1.2666  cal_e=1.2666  u_l2=0.0776  cov_n=0.0011
        col=11.2558  smp=0.0029  enc_t=0.2548  bnd_t=0.0398  wm_t=1.0269  crt_t=0.0087  diag_t=0.5662
        charts: 16/16 active  c0=0.11 c1=0.16 c2=0.05 c3=0.15 c4=0.03 c5=0.15 c6=0.01 c7=0.02 c8=0.03 c9=0.08 c10=0.06 c11=0.01 c12=0.02 c13=0.07 c14=0.03 c15=0.02
        symbols: 73/128 active  c0=7/8(H=1.50) c1=5/8(H=0.35) c2=3/8(H=0.70) c3=6/8(H=1.42) c4=4/8(H=1.12) c5=6/8(H=0.91) c6=4/8(H=0.95) c7=4/8(H=0.72) c8=5/8(H=1.50) c9=5/8(H=1.19) c10=4/8(H=1.16) c11=4/8(H=1.24) c12=3/8(H=0.78) c13=6/8(H=1.47) c14=4/8(H=0.97) c15=3/8(H=0.79)
E0082 [5upd]  ep_rew=13.7893  rew_20=14.0859  L_geo=0.7723  L_rew=0.0012  L_chart=2.1030  L_crit=0.3690  L_bnd=0.6225  lr=0.0010  dt=18.67s
        recon=0.2784  vq=0.3403  code_H=1.2538  code_px=3.5147  ch_usage=0.1226  rtr_mrg=0.0027  enc_gn=5.0188
        ctrl=0.0001  tex=0.0302  im_rew=0.0365  im_ret=0.5333  value=0.0253  wm_gn=0.6630
        z_norm=0.6426  z_max=0.8612  jump=0.0000  cons=0.0535  sol=0.9963  e_var=0.0008  ch_ent=2.4694  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5333  dret=0.5105  term=0.0265  bnd=0.0446  chart_acc=0.3338  chart_ent=1.9476  rw_drift=0.0000
        v_err=1.3560  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3461
        bnd_x=0.0452  bell=0.0362  bell_s=0.0437  rtg_e=1.3560  rtg_b=-1.3559  cal_e=1.3559  u_l2=0.1145  cov_n=0.0011
        col=11.1715  smp=0.0037  enc_t=0.2539  bnd_t=0.0401  wm_t=1.0659  crt_t=0.0093  diag_t=0.5852
        charts: 16/16 active  c0=0.20 c1=0.07 c2=0.02 c3=0.01 c4=0.01 c5=0.12 c6=0.10 c7=0.01 c8=0.04 c9=0.04 c10=0.06 c11=0.07 c12=0.02 c13=0.13 c14=0.07 c15=0.03
        symbols: 83/128 active  c0=6/8(H=1.24) c1=6/8(H=1.22) c2=7/8(H=1.19) c3=2/8(H=0.38) c4=3/8(H=0.41) c5=5/8(H=0.74) c6=5/8(H=1.03) c7=3/8(H=0.79) c8=7/8(H=1.49) c9=3/8(H=0.76) c10=4/8(H=1.08) c11=7/8(H=1.60) c12=3/8(H=0.91) c13=8/8(H=1.63) c14=8/8(H=1.35) c15=6/8(H=1.15)
E0083 [5upd]  ep_rew=11.6526  rew_20=13.9326  L_geo=0.8774  L_rew=0.0007  L_chart=2.0932  L_crit=0.3030  L_bnd=0.5726  lr=0.0010  dt=18.44s
        recon=0.2609  vq=0.2962  code_H=1.2285  code_px=3.4206  ch_usage=0.1225  rtr_mrg=0.0024  enc_gn=8.0483
        ctrl=0.0001  tex=0.0280  im_rew=0.0363  im_ret=0.5312  value=0.0273  wm_gn=0.2019
        z_norm=0.6566  z_max=0.8647  jump=0.0000  cons=0.0630  sol=1.0025  e_var=0.0011  ch_ent=2.4785  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5312  dret=0.5083  term=0.0267  bnd=0.0451  chart_acc=0.2875  chart_ent=2.1005  rw_drift=0.0000
        v_err=1.2433  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4784
        bnd_x=0.0454  bell=0.0359  bell_s=0.0308  rtg_e=1.2433  rtg_b=-1.2432  cal_e=1.2432  u_l2=0.1707  cov_n=0.0007
        col=11.1810  smp=0.0035  enc_t=0.2530  bnd_t=0.0401  wm_t=1.0234  crt_t=0.0086  diag_t=0.5684
        charts: 16/16 active  c0=0.09 c1=0.05 c2=0.02 c3=0.07 c4=0.02 c5=0.03 c6=0.06 c7=0.02 c8=0.02 c9=0.23 c10=0.04 c11=0.04 c12=0.03 c13=0.10 c14=0.06 c15=0.13
        symbols: 90/128 active  c0=4/8(H=0.80) c1=6/8(H=1.14) c2=7/8(H=1.12) c3=4/8(H=1.07) c4=5/8(H=1.09) c5=3/8(H=0.74) c6=5/8(H=1.31) c7=6/8(H=1.65) c8=3/8(H=0.74) c9=6/8(H=1.46) c10=7/8(H=1.00) c11=7/8(H=1.62) c12=6/8(H=1.41) c13=6/8(H=1.03) c14=8/8(H=1.81) c15=7/8(H=1.46)
E0084 [5upd]  ep_rew=15.0128  rew_20=16.3639  L_geo=0.8173  L_rew=0.0003  L_chart=1.9677  L_crit=0.2381  L_bnd=0.7419  lr=0.0010  dt=18.40s
        recon=0.2049  vq=0.3183  code_H=1.2200  code_px=3.4101  ch_usage=0.1091  rtr_mrg=0.0032  enc_gn=8.2666
        ctrl=0.0001  tex=0.0273  im_rew=0.0322  im_ret=0.4796  value=0.0310  wm_gn=0.1619
        z_norm=0.6696  z_max=0.8545  jump=0.0000  cons=0.0559  sol=0.9935  e_var=0.0012  ch_ent=2.6652  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4796  dret=0.4508  term=0.0335  bnd=0.0639  chart_acc=0.3042  chart_ent=2.0213  rw_drift=0.0000
        v_err=1.2684  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2650
        bnd_x=0.0653  bell=0.0355  bell_s=0.0277  rtg_e=1.2684  rtg_b=-1.2682  cal_e=1.2682  u_l2=0.1731  cov_n=0.0013
        col=11.1342  smp=0.0040  enc_t=0.2532  bnd_t=0.0399  wm_t=1.0232  crt_t=0.0085  diag_t=0.5761
        charts: 16/16 active  c0=0.14 c1=0.09 c2=0.05 c3=0.09 c4=0.03 c5=0.08 c6=0.03 c7=0.04 c8=0.03 c9=0.10 c10=0.06 c11=0.04 c12=0.04 c13=0.08 c14=0.06 c15=0.04
        symbols: 96/128 active  c0=5/8(H=0.47) c1=6/8(H=0.99) c2=7/8(H=1.51) c3=5/8(H=1.24) c4=4/8(H=0.92) c5=7/8(H=1.32) c6=4/8(H=0.93) c7=6/8(H=1.51) c8=6/8(H=1.30) c9=7/8(H=1.09) c10=5/8(H=1.24) c11=6/8(H=1.19) c12=7/8(H=1.60) c13=8/8(H=1.40) c14=6/8(H=1.39) c15=7/8(H=1.68)
E0085 [5upd]  ep_rew=14.0707  rew_20=16.0449  L_geo=0.8578  L_rew=0.0005  L_chart=2.0368  L_crit=0.2950  L_bnd=0.8261  lr=0.0010  dt=13.31s
        recon=0.2762  vq=0.2783  code_H=1.2128  code_px=3.3664  ch_usage=0.1627  rtr_mrg=0.0042  enc_gn=10.0933
        ctrl=0.0001  tex=0.0276  im_rew=0.0387  im_ret=0.5710  value=0.0331  wm_gn=0.1475
        z_norm=0.6588  z_max=0.8423  jump=0.0000  cons=0.0566  sol=0.9963  e_var=0.0011  ch_ent=2.6329  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5710  dret=0.5411  term=0.0348  bnd=0.0553  chart_acc=0.2858  chart_ent=2.0956  rw_drift=0.0000
        v_err=1.2774  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5306
        bnd_x=0.0602  bell=0.0371  bell_s=0.0276  rtg_e=1.2774  rtg_b=-1.2772  cal_e=1.2772  u_l2=0.1277  cov_n=0.0015
        col=7.9183  smp=0.0025  enc_t=0.2237  bnd_t=0.0343  wm_t=0.7209  crt_t=0.0058  diag_t=0.4094
        charts: 16/16 active  c0=0.06 c1=0.11 c2=0.06 c3=0.05 c4=0.02 c5=0.05 c6=0.03 c7=0.06 c8=0.02 c9=0.13 c10=0.08 c11=0.14 c12=0.03 c13=0.04 c14=0.06 c15=0.05
        symbols: 87/128 active  c0=3/8(H=0.85) c1=5/8(H=1.15) c2=5/8(H=1.50) c3=5/8(H=1.22) c4=4/8(H=0.95) c5=4/8(H=1.08) c6=5/8(H=1.14) c7=8/8(H=1.73) c8=4/8(H=0.82) c9=5/8(H=1.15) c10=6/8(H=1.38) c11=6/8(H=1.30) c12=5/8(H=1.47) c13=8/8(H=1.56) c14=7/8(H=1.25) c15=7/8(H=1.63)
E0086 [5upd]  ep_rew=9.5819  rew_20=15.0954  L_geo=0.8301  L_rew=0.0006  L_chart=1.8373  L_crit=0.2442  L_bnd=0.7690  lr=0.0010  dt=12.72s
        recon=0.1906  vq=0.3256  code_H=1.1805  code_px=3.2815  ch_usage=0.1911  rtr_mrg=0.0023  enc_gn=9.3195
        ctrl=0.0001  tex=0.0264  im_rew=0.0362  im_ret=0.5391  value=0.0362  wm_gn=0.2163
        z_norm=0.6930  z_max=0.9266  jump=0.0000  cons=0.0533  sol=0.9978  e_var=0.0035  ch_ent=2.5639  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5391  dret=0.5069  term=0.0375  bnd=0.0636  chart_acc=0.3304  chart_ent=1.8900  rw_drift=0.0000
        v_err=1.3452  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.4631
        bnd_x=0.0733  bell=0.0375  bell_s=0.0268  rtg_e=1.3452  rtg_b=-1.3450  cal_e=1.3450  u_l2=0.0934  cov_n=0.0018
        col=7.4558  smp=0.0036  enc_t=0.2260  bnd_t=0.0364  wm_t=0.6967  crt_t=0.0063  diag_t=0.3764
        charts: 16/16 active  c0=0.11 c1=0.11 c2=0.06 c3=0.10 c4=0.04 c5=0.05 c6=0.06 c7=0.05 c8=0.02 c9=0.14 c10=0.12 c11=0.05 c12=0.02 c13=0.03 c14=0.01 c15=0.03
        symbols: 85/128 active  c0=5/8(H=0.98) c1=8/8(H=1.34) c2=5/8(H=1.19) c3=8/8(H=1.13) c4=3/8(H=0.44) c5=4/8(H=1.15) c6=5/8(H=1.19) c7=5/8(H=1.11) c8=5/8(H=1.06) c9=7/8(H=1.34) c10=7/8(H=1.48) c11=4/8(H=1.10) c12=5/8(H=1.41) c13=4/8(H=0.83) c14=5/8(H=1.12) c15=5/8(H=1.41)
E0087 [5upd]  ep_rew=6.8756  rew_20=13.9831  L_geo=0.6906  L_rew=0.0009  L_chart=1.7300  L_crit=0.3288  L_bnd=0.6832  lr=0.0010  dt=17.83s
        recon=0.2293  vq=0.4366  code_H=1.2247  code_px=3.4511  ch_usage=0.0907  rtr_mrg=0.0041  enc_gn=5.4759
        ctrl=0.0001  tex=0.0261  im_rew=0.0381  im_ret=0.5618  value=0.0354  wm_gn=0.7597
        z_norm=0.6996  z_max=0.9625  jump=0.0000  cons=0.0587  sol=1.0017  e_var=0.0020  ch_ent=2.4909  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5618  dret=0.5329  term=0.0335  bnd=0.0541  chart_acc=0.4058  chart_ent=1.7574  rw_drift=0.0000
        v_err=1.3984  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1112
        bnd_x=0.0554  bell=0.0398  bell_s=0.0475  rtg_e=1.3984  rtg_b=-1.3982  cal_e=1.3982  u_l2=0.0729  cov_n=0.0012
        col=9.8450  smp=0.0049  enc_t=0.2618  bnd_t=0.0418  wm_t=1.1465  crt_t=0.0094  diag_t=0.6193
        charts: 16/16 active  c0=0.12 c1=0.08 c2=0.12 c3=0.21 c4=0.01 c5=0.09 c6=0.05 c7=0.02 c8=0.03 c9=0.07 c10=0.05 c11=0.04 c12=0.02 c13=0.05 c14=0.03 c15=0.02
        symbols: 93/128 active  c0=8/8(H=0.73) c1=7/8(H=1.34) c2=7/8(H=1.10) c3=8/8(H=1.36) c4=3/8(H=0.73) c5=6/8(H=1.29) c6=7/8(H=1.58) c7=5/8(H=1.26) c8=4/8(H=1.26) c9=6/8(H=1.03) c10=7/8(H=1.67) c11=6/8(H=1.34) c12=3/8(H=0.74) c13=7/8(H=1.62) c14=6/8(H=1.11) c15=3/8(H=0.93)
E0088 [5upd]  ep_rew=16.1694  rew_20=13.9493  L_geo=0.6955  L_rew=0.0005  L_chart=1.8103  L_crit=0.2812  L_bnd=0.4856  lr=0.0010  dt=18.93s
        recon=0.2327  vq=0.3482  code_H=1.1763  code_px=3.2546  ch_usage=0.0966  rtr_mrg=0.0004  enc_gn=7.4973
        ctrl=0.0001  tex=0.0318  im_rew=0.0403  im_ret=0.5891  value=0.0304  wm_gn=0.2447
        z_norm=0.6272  z_max=0.8840  jump=0.0000  cons=0.0385  sol=0.9975  e_var=0.0010  ch_ent=2.4603  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5891  dret=0.5632  term=0.0301  bnd=0.0459  chart_acc=0.3904  chart_ent=1.9302  rw_drift=0.0000
        v_err=1.3599  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5957
        bnd_x=0.0466  bell=0.0385  bell_s=0.0264  rtg_e=1.3599  rtg_b=-1.3599  cal_e=1.3599  u_l2=0.0738  cov_n=0.0011
        col=11.5139  smp=0.0051  enc_t=0.2555  bnd_t=0.0402  wm_t=1.0464  crt_t=0.0088  diag_t=0.5887
        charts: 16/16 active  c0=0.16 c1=0.07 c2=0.02 c3=0.20 c4=0.03 c5=0.11 c6=0.05 c7=0.01 c8=0.05 c9=0.10 c10=0.02 c11=0.07 c12=0.01 c13=0.03 c14=0.03 c15=0.02
        symbols: 79/128 active  c0=7/8(H=1.14) c1=7/8(H=1.32) c2=3/8(H=0.76) c3=8/8(H=0.80) c4=4/8(H=0.79) c5=5/8(H=0.65) c6=5/8(H=1.00) c7=3/8(H=0.72) c8=4/8(H=0.90) c9=7/8(H=1.48) c10=4/8(H=1.25) c11=7/8(H=1.62) c12=3/8(H=0.63) c13=5/8(H=1.45) c14=4/8(H=1.09) c15=3/8(H=1.04)
E0089 [5upd]  ep_rew=14.6537  rew_20=12.3007  L_geo=0.7023  L_rew=0.0007  L_chart=1.8316  L_crit=0.2454  L_bnd=0.6598  lr=0.0010  dt=18.61s
        recon=0.2223  vq=0.2897  code_H=1.2097  code_px=3.3652  ch_usage=0.1076  rtr_mrg=0.0032  enc_gn=6.0384
        ctrl=0.0001  tex=0.0273  im_rew=0.0343  im_ret=0.5040  value=0.0265  wm_gn=0.2484
        z_norm=0.6813  z_max=0.8869  jump=0.0000  cons=0.0694  sol=0.9942  e_var=0.0011  ch_ent=2.6042  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5040  dret=0.4802  term=0.0277  bnd=0.0497  chart_acc=0.3938  chart_ent=1.9412  rw_drift=0.0000
        v_err=1.1692  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1723
        bnd_x=0.0515  bell=0.0329  bell_s=0.0308  rtg_e=1.1692  rtg_b=-1.1691  cal_e=1.1691  u_l2=0.0887  cov_n=0.0017
        col=11.2786  smp=0.0043  enc_t=0.2544  bnd_t=0.0401  wm_t=1.0343  crt_t=0.0088  diag_t=0.5748
        charts: 16/16 active  c0=0.08 c1=0.07 c2=0.08 c3=0.13 c4=0.01 c5=0.14 c6=0.04 c7=0.04 c8=0.04 c9=0.04 c10=0.05 c11=0.03 c12=0.02 c13=0.12 c14=0.06 c15=0.04
        symbols: 78/128 active  c0=6/8(H=1.24) c1=5/8(H=1.03) c2=7/8(H=1.54) c3=5/8(H=1.26) c4=3/8(H=0.42) c5=7/8(H=1.56) c6=5/8(H=0.82) c7=4/8(H=0.78) c8=6/8(H=1.37) c9=5/8(H=0.96) c10=4/8(H=1.31) c11=4/8(H=1.24) c12=2/8(H=0.65) c13=7/8(H=1.18) c14=5/8(H=1.00) c15=3/8(H=1.02)
E0090 [5upd]  ep_rew=13.9849  rew_20=12.5992  L_geo=0.7131  L_rew=0.0007  L_chart=1.8488  L_crit=0.2485  L_bnd=0.7883  lr=0.0010  dt=18.57s
        recon=0.2186  vq=0.3304  code_H=1.2326  code_px=3.4382  ch_usage=0.0984  rtr_mrg=0.0030  enc_gn=5.8308
        ctrl=0.0001  tex=0.0269  im_rew=0.0322  im_ret=0.4697  value=0.0223  wm_gn=0.2843
        z_norm=0.6967  z_max=0.9052  jump=0.0000  cons=0.0615  sol=0.9953  e_var=0.0014  ch_ent=2.5576  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4697  dret=0.4500  term=0.0229  bnd=0.0438  chart_acc=0.4596  chart_ent=1.7906  rw_drift=0.0000
        v_err=1.1323  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.6690
        bnd_x=0.0447  bell=0.0311  bell_s=0.0168  rtg_e=1.1323  rtg_b=-1.1323  cal_e=1.1323  u_l2=0.0799  cov_n=0.0015
        col=11.2485  smp=0.0027  enc_t=0.2558  bnd_t=0.0401  wm_t=1.0316  crt_t=0.0088  diag_t=0.5791
        charts: 16/16 active  c0=0.14 c1=0.12 c2=0.12 c3=0.10 c4=0.02 c5=0.12 c6=0.03 c7=0.03 c8=0.02 c9=0.05 c10=0.05 c11=0.04 c12=0.03 c13=0.06 c14=0.02 c15=0.04
        symbols: 79/128 active  c0=6/8(H=1.48) c1=7/8(H=0.85) c2=5/8(H=0.93) c3=6/8(H=1.02) c4=3/8(H=0.75) c5=5/8(H=1.43) c6=5/8(H=1.08) c7=5/8(H=1.16) c8=4/8(H=1.20) c9=4/8(H=0.75) c10=5/8(H=1.56) c11=6/8(H=0.88) c12=4/8(H=0.79) c13=6/8(H=1.40) c14=5/8(H=1.18) c15=3/8(H=0.99)
  EVAL  reward=11.0 +/- 3.3  len=300
E0091 [5upd]  ep_rew=11.8550  rew_20=13.4159  L_geo=0.8198  L_rew=0.0009  L_chart=1.8840  L_crit=0.3246  L_bnd=0.7075  lr=0.0010  dt=18.06s
        recon=0.2681  vq=0.3509  code_H=1.2496  code_px=3.5008  ch_usage=0.0721  rtr_mrg=0.0039  enc_gn=8.7146
        ctrl=0.0001  tex=0.0278  im_rew=0.0403  im_ret=0.5799  value=0.0190  wm_gn=0.2125
        z_norm=0.6756  z_max=0.9428  jump=0.0000  cons=0.0522  sol=1.0005  e_var=0.0017  ch_ent=2.6303  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5799  dret=0.5636  term=0.0189  bnd=0.0289  chart_acc=0.4333  chart_ent=1.8175  rw_drift=0.0000
        v_err=1.5363  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3790
        bnd_x=0.0295  bell=0.0417  bell_s=0.0259  rtg_e=1.5363  rtg_b=-1.5363  cal_e=1.5363  u_l2=0.0777  cov_n=0.0009
        col=10.7886  smp=0.0031  enc_t=0.2548  bnd_t=0.0398  wm_t=1.0238  crt_t=0.0085  diag_t=0.5646
        charts: 16/16 active  c0=0.08 c1=0.08 c2=0.10 c3=0.13 c4=0.04 c5=0.08 c6=0.02 c7=0.04 c8=0.03 c9=0.12 c10=0.03 c11=0.08 c12=0.03 c13=0.04 c14=0.04 c15=0.04
        symbols: 94/128 active  c0=4/8(H=1.04) c1=7/8(H=1.48) c2=7/8(H=1.17) c3=7/8(H=1.20) c4=4/8(H=0.68) c5=6/8(H=1.05) c6=4/8(H=1.22) c7=6/8(H=1.46) c8=5/8(H=1.09) c9=6/8(H=1.48) c10=5/8(H=1.54) c11=6/8(H=1.54) c12=6/8(H=1.55) c13=8/8(H=1.29) c14=6/8(H=1.45) c15=7/8(H=1.54)
E0092 [5upd]  ep_rew=14.0805  rew_20=13.4748  L_geo=0.7240  L_rew=0.0002  L_chart=1.6837  L_crit=0.2312  L_bnd=0.5692  lr=0.0010  dt=18.46s
        recon=0.1814  vq=0.3684  code_H=1.0697  code_px=2.9309  ch_usage=0.1589  rtr_mrg=0.0023  enc_gn=11.9424
        ctrl=0.0001  tex=0.0268  im_rew=0.0381  im_ret=0.5512  value=0.0218  wm_gn=0.2304
        z_norm=0.7086  z_max=0.9159  jump=0.0000  cons=0.0372  sol=0.9998  e_var=0.0034  ch_ent=2.4073  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5512  dret=0.5328  term=0.0213  bnd=0.0344  chart_acc=0.4683  chart_ent=1.7446  rw_drift=0.0000
        v_err=1.0876  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3199
        bnd_x=0.0369  bell=0.0303  bell_s=0.0169  rtg_e=1.0876  rtg_b=-1.0875  cal_e=1.0875  u_l2=0.1368  cov_n=0.0009
        col=11.0620  smp=0.0042  enc_t=0.2538  bnd_t=0.0401  wm_t=1.0506  crt_t=0.0087  diag_t=0.5685
        charts: 16/16 active  c0=0.10 c1=0.05 c2=0.15 c3=0.10 c4=0.00 c5=0.03 c6=0.01 c7=0.05 c8=0.01 c9=0.21 c10=0.09 c11=0.01 c12=0.02 c13=0.07 c14=0.04 c15=0.05
        symbols: 85/128 active  c0=5/8(H=0.89) c1=5/8(H=0.36) c2=6/8(H=0.97) c3=8/8(H=1.27) c4=3/8(H=0.90) c5=3/8(H=0.92) c6=5/8(H=1.26) c7=6/8(H=1.22) c8=6/8(H=1.43) c9=5/8(H=0.72) c10=8/8(H=1.69) c11=4/8(H=1.07) c12=4/8(H=1.23) c13=5/8(H=0.82) c14=6/8(H=1.31) c15=6/8(H=1.48)
E0093 [5upd]  ep_rew=10.6372  rew_20=12.6657  L_geo=0.8700  L_rew=0.0007  L_chart=2.0409  L_crit=0.3244  L_bnd=0.4848  lr=0.0010  dt=18.35s
        recon=0.2788  vq=0.2520  code_H=1.2868  code_px=3.6261  ch_usage=0.1482  rtr_mrg=0.0018  enc_gn=8.4688
        ctrl=0.0001  tex=0.0303  im_rew=0.0357  im_ret=0.5197  value=0.0247  wm_gn=0.2192
        z_norm=0.6288  z_max=0.9056  jump=0.0000  cons=0.0611  sol=0.9980  e_var=0.0008  ch_ent=2.6411  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5197  dret=0.4991  term=0.0240  bnd=0.0413  chart_acc=0.3275  chart_ent=1.9728  rw_drift=0.0000
        v_err=1.6835  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6833
        bnd_x=0.0439  bell=0.0459  bell_s=0.0359  rtg_e=1.6835  rtg_b=-1.6834  cal_e=1.6834  u_l2=0.1858  cov_n=0.0010
        col=11.1066  smp=0.0047  enc_t=0.2539  bnd_t=0.0397  wm_t=1.0198  crt_t=0.0086  diag_t=0.5600
        charts: 16/16 active  c0=0.10 c1=0.07 c2=0.01 c3=0.10 c4=0.04 c5=0.02 c6=0.05 c7=0.04 c8=0.02 c9=0.06 c10=0.09 c11=0.06 c12=0.05 c13=0.10 c14=0.07 c15=0.12
        symbols: 89/128 active  c0=4/8(H=0.95) c1=4/8(H=0.47) c2=3/8(H=0.65) c3=5/8(H=1.30) c4=4/8(H=1.20) c5=5/8(H=0.98) c6=6/8(H=1.53) c7=6/8(H=1.58) c8=5/8(H=1.13) c9=6/8(H=1.25) c10=7/8(H=1.62) c11=6/8(H=1.47) c12=6/8(H=1.60) c13=8/8(H=1.61) c14=6/8(H=1.67) c15=8/8(H=1.57)
E0094 [5upd]  ep_rew=21.3526  rew_20=12.9005  L_geo=0.8915  L_rew=0.0006  L_chart=2.1224  L_crit=0.3102  L_bnd=0.5012  lr=0.0010  dt=14.16s
        recon=0.2921  vq=0.2489  code_H=1.3740  code_px=3.9604  ch_usage=0.0347  rtr_mrg=0.0036  enc_gn=4.9625
        ctrl=0.0001  tex=0.0302  im_rew=0.0421  im_ret=0.6090  value=0.0242  wm_gn=0.2116
        z_norm=0.6509  z_max=0.8871  jump=0.0000  cons=0.0450  sol=0.9960  e_var=0.0013  ch_ent=2.5987  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6090  dret=0.5892  term=0.0231  bnd=0.0337  chart_acc=0.3454  chart_ent=2.0236  rw_drift=0.0000
        v_err=1.2880  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6877
        bnd_x=0.0352  bell=0.0357  bell_s=0.0248  rtg_e=1.2880  rtg_b=-1.2879  cal_e=1.2879  u_l2=0.1811  cov_n=0.0009
        col=9.0561  smp=0.0013  enc_t=0.2224  bnd_t=0.0343  wm_t=0.6701  crt_t=0.0056  diag_t=0.3836
        charts: 16/16 active  c0=0.10 c1=0.02 c2=0.10 c3=0.12 c4=0.03 c5=0.08 c6=0.08 c7=0.03 c8=0.02 c9=0.13 c10=0.08 c11=0.03 c12=0.03 c13=0.07 c14=0.04 c15=0.04
        symbols: 82/128 active  c0=6/8(H=1.32) c1=5/8(H=1.36) c2=5/8(H=1.04) c3=7/8(H=0.93) c4=5/8(H=1.00) c5=4/8(H=0.97) c6=4/8(H=0.94) c7=5/8(H=1.08) c8=4/8(H=1.19) c9=5/8(H=0.44) c10=6/8(H=1.54) c11=4/8(H=0.99) c12=6/8(H=1.54) c13=6/8(H=1.62) c14=5/8(H=1.33) c15=5/8(H=1.28)
E0095 [5upd]  ep_rew=14.2623  rew_20=12.6181  L_geo=0.7533  L_rew=0.0007  L_chart=1.7978  L_crit=0.3089  L_bnd=0.5189  lr=0.0010  dt=12.43s
        recon=0.2269  vq=0.3728  code_H=1.2814  code_px=3.6229  ch_usage=0.0866  rtr_mrg=0.0022  enc_gn=6.6549
        ctrl=0.0001  tex=0.0248  im_rew=0.0358  im_ret=0.5229  value=0.0251  wm_gn=0.2146
        z_norm=0.7119  z_max=0.9195  jump=0.0000  cons=0.0514  sol=0.9948  e_var=0.0015  ch_ent=2.5232  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5229  dret=0.5013  term=0.0250  bnd=0.0429  chart_acc=0.4704  chart_ent=1.8191  rw_drift=0.0000
        v_err=1.3618  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6148
        bnd_x=0.0448  bell=0.0378  bell_s=0.0281  rtg_e=1.3618  rtg_b=-1.3617  cal_e=1.3617  u_l2=0.1720  cov_n=0.0009
        col=7.2028  smp=0.0035  enc_t=0.2243  bnd_t=0.0345  wm_t=0.6928  crt_t=0.0059  diag_t=0.3794
        charts: 16/16 active  c0=0.12 c1=0.10 c2=0.08 c3=0.18 c4=0.04 c5=0.14 c6=0.03 c7=0.06 c8=0.02 c9=0.04 c10=0.04 c11=0.03 c12=0.03 c13=0.04 c14=0.02 c15=0.03
        symbols: 89/128 active  c0=4/8(H=0.84) c1=6/8(H=1.24) c2=6/8(H=1.56) c3=7/8(H=1.38) c4=3/8(H=0.87) c5=7/8(H=0.88) c6=4/8(H=0.88) c7=6/8(H=1.25) c8=5/8(H=1.24) c9=4/8(H=0.62) c10=6/8(H=1.45) c11=6/8(H=1.59) c12=7/8(H=1.22) c13=6/8(H=1.34) c14=6/8(H=1.55) c15=6/8(H=1.19)
E0096 [5upd]  ep_rew=9.5663  rew_20=12.2797  L_geo=0.7922  L_rew=0.0005  L_chart=1.9395  L_crit=0.3099  L_bnd=0.5919  lr=0.0010  dt=12.79s
        recon=0.2285  vq=0.3265  code_H=1.1707  code_px=3.2453  ch_usage=0.1386  rtr_mrg=0.0026  enc_gn=11.4787
        ctrl=0.0001  tex=0.0272  im_rew=0.0362  im_ret=0.5337  value=0.0311  wm_gn=0.2062
        z_norm=0.6701  z_max=0.8784  jump=0.0000  cons=0.0424  sol=0.9963  e_var=0.0013  ch_ent=2.5641  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5337  dret=0.5061  term=0.0320  bnd=0.0544  chart_acc=0.4179  chart_ent=1.9090  rw_drift=0.0000
        v_err=1.3436  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1942
        bnd_x=0.0567  bell=0.0362  bell_s=0.0268  rtg_e=1.3436  rtg_b=-1.3435  cal_e=1.3435  u_l2=0.1255  cov_n=0.0013
        col=7.4090  smp=0.0021  enc_t=0.2259  bnd_t=0.0353  wm_t=0.7187  crt_t=0.0062  diag_t=0.3978
        charts: 16/16 active  c0=0.03 c1=0.18 c2=0.09 c3=0.06 c4=0.03 c5=0.17 c6=0.05 c7=0.06 c8=0.04 c9=0.06 c10=0.04 c11=0.03 c12=0.06 c13=0.02 c14=0.05 c15=0.03
        symbols: 97/128 active  c0=4/8(H=0.97) c1=6/8(H=1.04) c2=7/8(H=1.37) c3=7/8(H=1.05) c4=4/8(H=1.02) c5=6/8(H=1.27) c6=5/8(H=1.03) c7=7/8(H=1.41) c8=6/8(H=1.36) c9=7/8(H=1.49) c10=6/8(H=1.43) c11=7/8(H=1.44) c12=5/8(H=1.50) c13=7/8(H=1.45) c14=7/8(H=1.38) c15=6/8(H=1.05)
E0097 [5upd]  ep_rew=14.6546  rew_20=12.3145  L_geo=0.7911  L_rew=0.0010  L_chart=1.9723  L_crit=0.2706  L_bnd=0.6722  lr=0.0010  dt=12.93s
        recon=0.2453  vq=0.2657  code_H=1.3109  code_px=3.7366  ch_usage=0.1159  rtr_mrg=0.0017  enc_gn=7.9864
        ctrl=0.0002  tex=0.0298  im_rew=0.0367  im_ret=0.5456  value=0.0355  wm_gn=0.1744
        z_norm=0.6432  z_max=0.8940  jump=0.0000  cons=0.0534  sol=0.9926  e_var=0.0008  ch_ent=2.4981  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5456  dret=0.5142  term=0.0365  bnd=0.0611  chart_acc=0.3983  chart_ent=1.9662  rw_drift=0.0000
        v_err=1.3378  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1670
        bnd_x=0.0626  bell=0.0382  bell_s=0.0291  rtg_e=1.3378  rtg_b=-1.3376  cal_e=1.3376  u_l2=0.0752  cov_n=0.0018
        col=7.5085  smp=0.0036  enc_t=0.2276  bnd_t=0.0362  wm_t=0.7254  crt_t=0.0063  diag_t=0.3875
        charts: 16/16 active  c0=0.05 c1=0.14 c2=0.11 c3=0.11 c4=0.03 c5=0.06 c6=0.05 c7=0.03 c8=0.04 c9=0.19 c10=0.05 c11=0.03 c12=0.01 c13=0.06 c14=0.01 c15=0.02
        symbols: 90/128 active  c0=4/8(H=1.11) c1=8/8(H=1.62) c2=4/8(H=0.92) c3=7/8(H=0.95) c4=7/8(H=1.24) c5=6/8(H=0.72) c6=6/8(H=1.16) c7=5/8(H=1.17) c8=4/8(H=1.15) c9=6/8(H=1.21) c10=4/8(H=0.96) c11=7/8(H=1.61) c12=5/8(H=1.25) c13=6/8(H=1.26) c14=6/8(H=1.36) c15=5/8(H=1.05)
E0098 [5upd]  ep_rew=7.4818  rew_20=12.9313  L_geo=0.7651  L_rew=0.0010  L_chart=1.8770  L_crit=0.3342  L_bnd=0.5627  lr=0.0010  dt=14.72s
        recon=0.2768  vq=0.2447  code_H=1.2952  code_px=3.6574  ch_usage=0.1004  rtr_mrg=0.0026  enc_gn=9.5281
        ctrl=0.0001  tex=0.0332  im_rew=0.0404  im_ret=0.5958  value=0.0345  wm_gn=0.2030
        z_norm=0.5732  z_max=0.9171  jump=0.0000  cons=0.0604  sol=0.9983  e_var=0.0010  ch_ent=2.6011  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5958  dret=0.5655  term=0.0353  bnd=0.0536  chart_acc=0.4058  chart_ent=1.8788  rw_drift=0.0000
        v_err=1.4496  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2892
        bnd_x=0.0574  bell=0.0393  bell_s=0.0201  rtg_e=1.4496  rtg_b=-1.4495  cal_e=1.4495  u_l2=0.0576  cov_n=0.0014
        col=7.6364  smp=0.0026  enc_t=0.2437  bnd_t=0.0388  wm_t=0.9824  crt_t=0.0082  diag_t=0.6562
        charts: 16/16 active  c0=0.08 c1=0.06 c2=0.03 c3=0.15 c4=0.02 c5=0.06 c6=0.10 c7=0.03 c8=0.07 c9=0.15 c10=0.04 c11=0.04 c12=0.03 c13=0.07 c14=0.03 c15=0.06
        symbols: 93/128 active  c0=5/8(H=1.23) c1=4/8(H=0.71) c2=5/8(H=1.17) c3=6/8(H=1.38) c4=4/8(H=1.19) c5=6/8(H=0.96) c6=7/8(H=1.70) c7=7/8(H=1.65) c8=7/8(H=1.07) c9=4/8(H=0.33) c10=6/8(H=1.51) c11=6/8(H=1.35) c12=6/8(H=1.49) c13=7/8(H=1.23) c14=7/8(H=1.57) c15=6/8(H=1.47)
E0099 [5upd]  ep_rew=13.7544  rew_20=13.5097  L_geo=0.7316  L_rew=0.0006  L_chart=1.7954  L_crit=0.2486  L_bnd=0.5701  lr=0.0010  dt=19.60s
        recon=0.2283  vq=0.2804  code_H=1.3060  code_px=3.7158  ch_usage=0.0691  rtr_mrg=0.0021  enc_gn=7.9221
        ctrl=0.0002  tex=0.0311  im_rew=0.0372  im_ret=0.5444  value=0.0306  wm_gn=0.2634
        z_norm=0.6380  z_max=0.9351  jump=0.0000  cons=0.0484  sol=0.9993  e_var=0.0021  ch_ent=2.4774  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5444  dret=0.5200  term=0.0284  bnd=0.0469  chart_acc=0.4000  chart_ent=1.8527  rw_drift=0.0000
        v_err=1.2526  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2815
        bnd_x=0.0489  bell=0.0347  bell_s=0.0166  rtg_e=1.2526  rtg_b=-1.2525  cal_e=1.2525  u_l2=0.0791  cov_n=0.0020
        col=12.1102  smp=0.0041  enc_t=0.2576  bnd_t=0.0402  wm_t=1.0565  crt_t=0.0088  diag_t=0.5991
        charts: 16/16 active  c0=0.20 c1=0.04 c2=0.05 c3=0.16 c4=0.00 c5=0.07 c6=0.08 c7=0.02 c8=0.02 c9=0.09 c10=0.07 c11=0.03 c12=0.02 c13=0.05 c14=0.06 c15=0.04
        symbols: 89/128 active  c0=5/8(H=1.20) c1=3/8(H=1.06) c2=4/8(H=1.35) c3=8/8(H=1.92) c4=2/8(H=0.24) c5=7/8(H=1.50) c6=6/8(H=1.33) c7=6/8(H=1.54) c8=5/8(H=1.43) c9=3/8(H=0.35) c10=6/8(H=1.61) c11=8/8(H=1.68) c12=4/8(H=1.24) c13=7/8(H=1.56) c14=7/8(H=1.41) c15=8/8(H=1.65)
E0100 [5upd]  ep_rew=14.4833  rew_20=12.9869  L_geo=0.7180  L_rew=0.0007  L_chart=1.6638  L_crit=0.3101  L_bnd=0.6333  lr=0.0010  dt=18.92s
        recon=0.1930  vq=0.3060  code_H=1.2444  code_px=3.4754  ch_usage=0.0583  rtr_mrg=0.0032  enc_gn=5.7022
        ctrl=0.0001  tex=0.0281  im_rew=0.0346  im_ret=0.5016  value=0.0212  wm_gn=0.2413
        z_norm=0.6808  z_max=0.9151  jump=0.0000  cons=0.0374  sol=0.9952  e_var=0.0030  ch_ent=2.5614  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5016  dret=0.4836  term=0.0209  bnd=0.0373  chart_acc=0.4138  chart_ent=1.7283  rw_drift=0.0000
        v_err=1.1876  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0312
        bnd_x=0.0404  bell=0.0333  bell_s=0.0240  rtg_e=1.1876  rtg_b=-1.1875  cal_e=1.1875  u_l2=0.0955  cov_n=0.0013
        col=11.4364  smp=0.0043  enc_t=0.2591  bnd_t=0.0406  wm_t=1.0561  crt_t=0.0089  diag_t=0.5916
        charts: 16/16 active  c0=0.07 c1=0.06 c2=0.14 c3=0.11 c4=0.01 c5=0.05 c6=0.14 c7=0.04 c8=0.01 c9=0.10 c10=0.06 c11=0.06 c12=0.02 c13=0.05 c14=0.03 c15=0.05
        symbols: 95/128 active  c0=6/8(H=0.67) c1=8/8(H=1.33) c2=6/8(H=0.92) c3=7/8(H=1.48) c4=4/8(H=0.55) c5=4/8(H=1.14) c6=8/8(H=1.56) c7=6/8(H=1.47) c8=4/8(H=0.61) c9=6/8(H=0.35) c10=8/8(H=1.69) c11=7/8(H=1.21) c12=5/8(H=1.32) c13=5/8(H=1.14) c14=5/8(H=1.12) c15=6/8(H=1.51)
E0101 [5upd]  ep_rew=6.3771  rew_20=13.0355  L_geo=0.7517  L_rew=0.0004  L_chart=1.6386  L_crit=0.2926  L_bnd=0.5407  lr=0.0010  dt=20.28s
        recon=0.1969  vq=0.2986  code_H=1.2212  code_px=3.4210  ch_usage=0.0855  rtr_mrg=0.0033  enc_gn=6.9127
        ctrl=0.0000  tex=0.0280  im_rew=0.0388  im_ret=0.5557  value=0.0161  wm_gn=0.1548
        z_norm=0.6628  z_max=0.9244  jump=0.0000  cons=0.0479  sol=0.9966  e_var=0.0013  ch_ent=2.5638  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5557  dret=0.5423  term=0.0156  bnd=0.0248  chart_acc=0.4792  chart_ent=1.7461  rw_drift=0.0000
        v_err=1.3947  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2719
        bnd_x=0.0261  bell=0.0384  bell_s=0.0234  rtg_e=1.3947  rtg_b=-1.3947  cal_e=1.3947  u_l2=0.0928  cov_n=0.0005
        col=12.7988  smp=0.0027  enc_t=0.2859  bnd_t=0.0401  wm_t=1.0341  crt_t=0.0087  diag_t=0.5760
        charts: 16/16 active  c0=0.02 c1=0.07 c2=0.14 c3=0.11 c4=0.03 c5=0.04 c6=0.05 c7=0.03 c8=0.03 c9=0.15 c10=0.03 c11=0.09 c12=0.02 c13=0.04 c14=0.11 c15=0.04
        symbols: 93/128 active  c0=4/8(H=1.14) c1=8/8(H=0.82) c2=6/8(H=0.37) c3=8/8(H=1.41) c4=5/8(H=1.07) c5=5/8(H=0.80) c6=4/8(H=0.95) c7=7/8(H=1.74) c8=5/8(H=1.10) c9=6/8(H=0.99) c10=5/8(H=1.20) c11=7/8(H=1.54) c12=6/8(H=1.61) c13=6/8(H=1.39) c14=6/8(H=1.29) c15=5/8(H=1.24)
E0102 [5upd]  ep_rew=15.4622  rew_20=13.7020  L_geo=0.7947  L_rew=0.0004  L_chart=1.8371  L_crit=0.2588  L_bnd=0.5041  lr=0.0010  dt=18.61s
        recon=0.2319  vq=0.2576  code_H=1.3186  code_px=3.7552  ch_usage=0.1029  rtr_mrg=0.0038  enc_gn=6.3966
        ctrl=0.0000  tex=0.0291  im_rew=0.0384  im_ret=0.5540  value=0.0196  wm_gn=0.1470
        z_norm=0.6371  z_max=0.9175  jump=0.0000  cons=0.0375  sol=0.9981  e_var=0.0008  ch_ent=2.6331  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5540  dret=0.5367  term=0.0201  bnd=0.0322  chart_acc=0.4292  chart_ent=1.8523  rw_drift=0.0000
        v_err=1.2099  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3529
        bnd_x=0.0335  bell=0.0342  bell_s=0.0242  rtg_e=1.2099  rtg_b=-1.2099  cal_e=1.2099  u_l2=0.0980  cov_n=0.0005
        col=11.2605  smp=0.0038  enc_t=0.2541  bnd_t=0.0402  wm_t=1.0371  crt_t=0.0090  diag_t=0.5827
        charts: 16/16 active  c0=0.07 c1=0.20 c2=0.06 c3=0.06 c4=0.03 c5=0.08 c6=0.08 c7=0.05 c8=0.03 c9=0.07 c10=0.03 c11=0.04 c12=0.03 c13=0.06 c14=0.05 c15=0.06
        symbols: 102/128 active  c0=7/8(H=1.52) c1=7/8(H=1.25) c2=7/8(H=1.61) c3=7/8(H=1.27) c4=4/8(H=1.10) c5=5/8(H=1.39) c6=8/8(H=1.25) c7=7/8(H=1.46) c8=8/8(H=1.54) c9=4/8(H=0.67) c10=5/8(H=1.54) c11=7/8(H=1.56) c12=6/8(H=1.61) c13=7/8(H=1.41) c14=6/8(H=1.41) c15=7/8(H=1.52)
E0103 [5upd]  ep_rew=6.2094  rew_20=13.6026  L_geo=0.7075  L_rew=0.0004  L_chart=1.9326  L_crit=0.3124  L_bnd=0.6636  lr=0.0010  dt=18.85s
        recon=0.2551  vq=0.3081  code_H=1.2686  code_px=3.5660  ch_usage=0.0717  rtr_mrg=0.0011  enc_gn=6.8564
        ctrl=0.0001  tex=0.0279  im_rew=0.0355  im_ret=0.5181  value=0.0255  wm_gn=0.3037
        z_norm=0.6527  z_max=0.9477  jump=0.0000  cons=0.0411  sol=0.9981  e_var=0.0015  ch_ent=2.5281  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5181  dret=0.4962  term=0.0255  bnd=0.0442  chart_acc=0.4108  chart_ent=1.6441  rw_drift=0.0000
        v_err=1.3359  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2016
        bnd_x=0.0449  bell=0.0362  bell_s=0.0227  rtg_e=1.3359  rtg_b=-1.3359  cal_e=1.3359  u_l2=0.1131  cov_n=0.0008
        col=11.2134  smp=0.0040  enc_t=0.2581  bnd_t=0.0410  wm_t=1.0882  crt_t=0.0093  diag_t=0.5874
        charts: 16/16 active  c0=0.24 c1=0.07 c2=0.06 c3=0.13 c4=0.04 c5=0.04 c6=0.07 c7=0.04 c8=0.02 c9=0.05 c10=0.03 c11=0.05 c12=0.04 c13=0.03 c14=0.06 c15=0.03
        symbols: 87/128 active  c0=7/8(H=1.38) c1=7/8(H=1.06) c2=3/8(H=0.94) c3=8/8(H=1.45) c4=4/8(H=0.98) c5=4/8(H=1.19) c6=7/8(H=1.39) c7=5/8(H=1.41) c8=6/8(H=1.53) c9=5/8(H=1.12) c10=5/8(H=1.44) c11=6/8(H=1.27) c12=6/8(H=1.61) c13=6/8(H=1.35) c14=4/8(H=1.22) c15=4/8(H=1.25)
E0104 [5upd]  ep_rew=8.7784  rew_20=12.4684  L_geo=0.7953  L_rew=0.0006  L_chart=1.8775  L_crit=0.2454  L_bnd=0.7164  lr=0.0010  dt=18.95s
        recon=0.2092  vq=0.2982  code_H=1.3308  code_px=3.7849  ch_usage=0.1414  rtr_mrg=0.0022  enc_gn=6.6351
        ctrl=0.0001  tex=0.0280  im_rew=0.0352  im_ret=0.5219  value=0.0332  wm_gn=0.1449
        z_norm=0.6632  z_max=0.9394  jump=0.0000  cons=0.0658  sol=0.9927  e_var=0.0009  ch_ent=2.7119  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5219  dret=0.4931  term=0.0335  bnd=0.0585  chart_acc=0.3700  chart_ent=1.8245  rw_drift=0.0000
        v_err=1.1724  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3572
        bnd_x=0.0621  bell=0.0326  bell_s=0.0237  rtg_e=1.1724  rtg_b=-1.1722  cal_e=1.1722  u_l2=0.1200  cov_n=0.0012
        col=11.5956  smp=0.0049  enc_t=0.2549  bnd_t=0.0408  wm_t=1.0380  crt_t=0.0087  diag_t=0.5712
        charts: 16/16 active  c0=0.07 c1=0.07 c2=0.06 c3=0.08 c4=0.02 c5=0.09 c6=0.05 c7=0.05 c8=0.04 c9=0.06 c10=0.05 c11=0.10 c12=0.05 c13=0.09 c14=0.06 c15=0.06
        symbols: 86/128 active  c0=4/8(H=0.52) c1=7/8(H=1.29) c2=3/8(H=0.67) c3=7/8(H=1.05) c4=5/8(H=0.73) c5=4/8(H=0.68) c6=8/8(H=1.53) c7=7/8(H=1.16) c8=6/8(H=1.14) c9=5/8(H=1.11) c10=5/8(H=1.37) c11=5/8(H=1.37) c12=6/8(H=1.34) c13=5/8(H=1.31) c14=5/8(H=1.23) c15=4/8(H=1.04)
E0105 [5upd]  ep_rew=40.5358  rew_20=14.4302  L_geo=0.7207  L_rew=0.0008  L_chart=1.8014  L_crit=0.2848  L_bnd=0.6728  lr=0.0010  dt=18.48s
        recon=0.2104  vq=0.3052  code_H=1.3213  code_px=3.7523  ch_usage=0.0516  rtr_mrg=0.0039  enc_gn=5.0011
        ctrl=0.0001  tex=0.0293  im_rew=0.0389  im_ret=0.5764  value=0.0386  wm_gn=0.2715
        z_norm=0.6451  z_max=0.9489  jump=0.0000  cons=0.0465  sol=0.9981  e_var=0.0010  ch_ent=2.4870  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5764  dret=0.5441  term=0.0376  bnd=0.0594  chart_acc=0.4125  chart_ent=1.8467  rw_drift=0.0000
        v_err=1.3957  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2081
        bnd_x=0.0676  bell=0.0406  bell_s=0.0481  rtg_e=1.3957  rtg_b=-1.3955  cal_e=1.3955  u_l2=0.0959  cov_n=0.0008
        col=11.1965  smp=0.0049  enc_t=0.2534  bnd_t=0.0398  wm_t=1.0253  crt_t=0.0086  diag_t=0.5748
        charts: 16/16 active  c0=0.02 c1=0.15 c2=0.02 c3=0.11 c4=0.02 c5=0.04 c6=0.07 c7=0.02 c8=0.05 c9=0.14 c10=0.04 c11=0.13 c12=0.01 c13=0.13 c14=0.03 c15=0.03
        symbols: 95/128 active  c0=6/8(H=1.31) c1=8/8(H=1.28) c2=4/8(H=0.47) c3=6/8(H=1.30) c4=5/8(H=0.87) c5=4/8(H=0.74) c6=8/8(H=1.71) c7=4/8(H=1.05) c8=6/8(H=0.92) c9=6/8(H=1.04) c10=5/8(H=1.01) c11=8/8(H=1.29) c12=6/8(H=1.53) c13=7/8(H=1.23) c14=7/8(H=1.00) c15=5/8(H=1.36)
E0106 [5upd]  ep_rew=13.4181  rew_20=14.4366  L_geo=0.7565  L_rew=0.0003  L_chart=1.7578  L_crit=0.2308  L_bnd=0.5805  lr=0.0010  dt=18.57s
        recon=0.1840  vq=0.2739  code_H=1.2213  code_px=3.4006  ch_usage=0.0961  rtr_mrg=0.0033  enc_gn=7.1274
        ctrl=0.0001  tex=0.0264  im_rew=0.0362  im_ret=0.5414  value=0.0422  wm_gn=0.2231
        z_norm=0.6929  z_max=0.9150  jump=0.0000  cons=0.0469  sol=0.9995  e_var=0.0020  ch_ent=2.5032  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5414  dret=0.5063  term=0.0409  bnd=0.0694  chart_acc=0.4783  chart_ent=1.8679  rw_drift=0.0000
        v_err=1.3168  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1696
        bnd_x=0.0753  bell=0.0372  bell_s=0.0294  rtg_e=1.3168  rtg_b=-1.3165  cal_e=1.3165  u_l2=0.0796  cov_n=0.0014
        col=11.1664  smp=0.0025  enc_t=0.2530  bnd_t=0.0399  wm_t=1.0522  crt_t=0.0086  diag_t=0.5728
        charts: 16/16 active  c0=0.06 c1=0.07 c2=0.10 c3=0.04 c4=0.02 c5=0.01 c6=0.05 c7=0.07 c8=0.01 c9=0.24 c10=0.05 c11=0.06 c12=0.02 c13=0.08 c14=0.05 c15=0.05
        symbols: 85/128 active  c0=2/8(H=0.69) c1=6/8(H=1.24) c2=4/8(H=0.93) c3=6/8(H=1.31) c4=6/8(H=1.31) c5=4/8(H=1.01) c6=6/8(H=1.29) c7=7/8(H=1.43) c8=4/8(H=0.76) c9=6/8(H=0.57) c10=5/8(H=1.29) c11=5/8(H=1.30) c12=5/8(H=1.55) c13=7/8(H=1.04) c14=6/8(H=1.24) c15=6/8(H=1.60)
E0107 [5upd]  ep_rew=15.0820  rew_20=14.0734  L_geo=0.8643  L_rew=0.0003  L_chart=1.9591  L_crit=0.2383  L_bnd=0.5311  lr=0.0010  dt=18.70s
        recon=0.2107  vq=0.2699  code_H=1.2898  code_px=3.6529  ch_usage=0.1250  rtr_mrg=0.0045  enc_gn=5.7694
        ctrl=0.0001  tex=0.0272  im_rew=0.0348  im_ret=0.5194  value=0.0373  wm_gn=0.2980
        z_norm=0.6682  z_max=0.9726  jump=0.0000  cons=0.0542  sol=1.0006  e_var=0.0030  ch_ent=2.5344  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5194  dret=0.4869  term=0.0379  bnd=0.0669  chart_acc=0.3613  chart_ent=2.0075  rw_drift=0.0000
        v_err=1.2793  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2938
        bnd_x=0.0757  bell=0.0353  bell_s=0.0171  rtg_e=1.2793  rtg_b=-1.2791  cal_e=1.2791  u_l2=0.0547  cov_n=0.0014
        col=11.4317  smp=0.0048  enc_t=0.2541  bnd_t=0.0396  wm_t=1.0227  crt_t=0.0086  diag_t=0.5720
        charts: 16/16 active  c0=0.02 c1=0.14 c2=0.12 c3=0.11 c4=0.01 c5=0.05 c6=0.05 c7=0.05 c8=0.01 c9=0.16 c10=0.07 c11=0.04 c12=0.02 c13=0.08 c14=0.02 c15=0.04
        symbols: 79/128 active  c0=3/8(H=1.03) c1=7/8(H=1.47) c2=4/8(H=1.15) c3=6/8(H=1.27) c4=5/8(H=1.11) c5=5/8(H=1.02) c6=5/8(H=1.03) c7=6/8(H=1.19) c8=4/8(H=0.95) c9=7/8(H=1.30) c10=6/8(H=1.52) c11=3/8(H=0.72) c12=4/8(H=1.19) c13=5/8(H=1.30) c14=3/8(H=1.02) c15=6/8(H=1.57)
E0108 [5upd]  ep_rew=10.1610  rew_20=14.2510  L_geo=0.7772  L_rew=0.0006  L_chart=1.7629  L_crit=0.2892  L_bnd=0.5755  lr=0.0010  dt=18.46s
        recon=0.2106  vq=0.2576  code_H=1.2629  code_px=3.5523  ch_usage=0.1118  rtr_mrg=0.0028  enc_gn=6.7538
        ctrl=0.0001  tex=0.0296  im_rew=0.0381  im_ret=0.5581  value=0.0290  wm_gn=0.2058
        z_norm=0.6337  z_max=0.9297  jump=0.0000  cons=0.0506  sol=0.9998  e_var=0.0006  ch_ent=2.5457  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5581  dret=0.5328  term=0.0294  bnd=0.0475  chart_acc=0.4413  chart_ent=1.7772  rw_drift=0.0000
        v_err=1.2715  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1267
        bnd_x=0.0506  bell=0.0351  bell_s=0.0279  rtg_e=1.2715  rtg_b=-1.2714  cal_e=1.2714  u_l2=0.0423  cov_n=0.0011
        col=11.2029  smp=0.0037  enc_t=0.2539  bnd_t=0.0399  wm_t=1.0205  crt_t=0.0085  diag_t=0.5713
        charts: 16/16 active  c0=0.12 c1=0.13 c2=0.06 c3=0.16 c4=0.04 c5=0.04 c6=0.04 c7=0.07 c8=0.01 c9=0.10 c10=0.05 c11=0.04 c12=0.02 c13=0.08 c14=0.01 c15=0.03
        symbols: 92/128 active  c0=6/8(H=1.19) c1=7/8(H=1.50) c2=8/8(H=1.47) c3=7/8(H=1.43) c4=4/8(H=0.82) c5=5/8(H=1.03) c6=5/8(H=1.37) c7=7/8(H=0.89) c8=5/8(H=1.17) c9=6/8(H=0.92) c10=6/8(H=1.54) c11=6/8(H=0.89) c12=4/8(H=1.16) c13=7/8(H=1.21) c14=5/8(H=0.92) c15=4/8(H=1.24)
E0109 [5upd]  ep_rew=14.1789  rew_20=14.4424  L_geo=0.7752  L_rew=0.0004  L_chart=1.9276  L_crit=0.2880  L_bnd=0.7415  lr=0.0010  dt=18.44s
        recon=0.2413  vq=0.2421  code_H=1.1948  code_px=3.3416  ch_usage=0.0716  rtr_mrg=0.0027  enc_gn=5.7904
        ctrl=0.0001  tex=0.0320  im_rew=0.0369  im_ret=0.5375  value=0.0249  wm_gn=0.2174
        z_norm=0.6142  z_max=0.9282  jump=0.0000  cons=0.0560  sol=1.0031  e_var=0.0007  ch_ent=2.5985  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5375  dret=0.5156  term=0.0255  bnd=0.0425  chart_acc=0.3388  chart_ent=1.7952  rw_drift=0.0000
        v_err=1.3270  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.4106
        bnd_x=0.0443  bell=0.0360  bell_s=0.0211  rtg_e=1.3270  rtg_b=-1.3270  cal_e=1.3270  u_l2=0.0621  cov_n=0.0013
        col=11.1775  smp=0.0039  enc_t=0.2532  bnd_t=0.0398  wm_t=1.0221  crt_t=0.0086  diag_t=0.5780
        charts: 16/16 active  c0=0.06 c1=0.10 c2=0.01 c3=0.07 c4=0.01 c5=0.09 c6=0.11 c7=0.04 c8=0.05 c9=0.13 c10=0.03 c11=0.11 c12=0.03 c13=0.08 c14=0.03 c15=0.04
        symbols: 103/128 active  c0=5/8(H=0.98) c1=8/8(H=1.59) c2=5/8(H=0.95) c3=5/8(H=1.31) c4=4/8(H=0.77) c5=7/8(H=0.92) c6=8/8(H=1.80) c7=6/8(H=1.00) c8=8/8(H=1.41) c9=6/8(H=1.17) c10=5/8(H=1.43) c11=7/8(H=1.46) c12=7/8(H=1.75) c13=8/8(H=1.20) c14=7/8(H=1.48) c15=7/8(H=1.27)
E0110 [5upd]  ep_rew=9.6063  rew_20=12.8083  L_geo=0.8310  L_rew=0.0004  L_chart=1.8828  L_crit=0.2955  L_bnd=0.6704  lr=0.0010  dt=18.44s
        recon=0.2699  vq=0.2201  code_H=1.2886  code_px=3.6431  ch_usage=0.1238  rtr_mrg=0.0043  enc_gn=9.0676
        ctrl=0.0001  tex=0.0321  im_rew=0.0338  im_ret=0.4915  value=0.0211  wm_gn=0.1738
        z_norm=0.5718  z_max=0.9439  jump=0.0000  cons=0.0531  sol=1.0002  e_var=0.0006  ch_ent=2.5304  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4915  dret=0.4728  term=0.0217  bnd=0.0396  chart_acc=0.3929  chart_ent=1.8964  rw_drift=0.0000
        v_err=1.2715  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2181
        bnd_x=0.0425  bell=0.0346  bell_s=0.0163  rtg_e=1.2715  rtg_b=-1.2715  cal_e=1.2715  u_l2=0.0986  cov_n=0.0009
        col=11.1876  smp=0.0029  enc_t=0.2533  bnd_t=0.0400  wm_t=1.0203  crt_t=0.0086  diag_t=0.5700
        charts: 16/16 active  c0=0.10 c1=0.12 c2=0.01 c3=0.15 c4=0.01 c5=0.05 c6=0.15 c7=0.03 c8=0.02 c9=0.09 c10=0.04 c11=0.03 c12=0.04 c13=0.07 c14=0.04 c15=0.06
        symbols: 82/128 active  c0=5/8(H=0.86) c1=7/8(H=1.22) c2=3/8(H=0.70) c3=7/8(H=1.43) c4=3/8(H=0.78) c5=3/8(H=0.47) c6=8/8(H=1.39) c7=4/8(H=1.07) c8=4/8(H=1.38) c9=6/8(H=1.21) c10=4/8(H=1.18) c11=7/8(H=1.16) c12=5/8(H=1.26) c13=6/8(H=1.12) c14=5/8(H=1.28) c15=5/8(H=1.35)
E0111 [5upd]  ep_rew=29.8135  rew_20=14.3080  L_geo=0.8109  L_rew=0.0009  L_chart=1.8850  L_crit=0.3265  L_bnd=0.6644  lr=0.0010  dt=18.44s
        recon=0.2454  vq=0.2681  code_H=1.3417  code_px=3.8316  ch_usage=0.0958  rtr_mrg=0.0043  enc_gn=7.9992
        ctrl=0.0001  tex=0.0293  im_rew=0.0411  im_ret=0.5940  value=0.0211  wm_gn=0.3649
        z_norm=0.6561  z_max=0.9373  jump=0.0000  cons=0.0450  sol=1.0024  e_var=0.0015  ch_ent=2.4561  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5940  dret=0.5757  term=0.0212  bnd=0.0317  chart_acc=0.3863  chart_ent=1.8832  rw_drift=0.0000
        v_err=1.3448  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0278
        bnd_x=0.0340  bell=0.0391  bell_s=0.0281  rtg_e=1.3448  rtg_b=-1.3448  cal_e=1.3448  u_l2=0.1311  cov_n=0.0008
        col=11.0840  smp=0.0045  enc_t=0.2524  bnd_t=0.0400  wm_t=1.0426  crt_t=0.0086  diag_t=0.5677
        charts: 16/16 active  c0=0.09 c1=0.03 c2=0.10 c3=0.17 c4=0.01 c5=0.03 c6=0.14 c7=0.03 c8=0.01 c9=0.15 c10=0.06 c11=0.03 c12=0.02 c13=0.04 c14=0.05 c15=0.04
        symbols: 93/128 active  c0=5/8(H=0.67) c1=7/8(H=1.14) c2=6/8(H=1.27) c3=7/8(H=1.75) c4=7/8(H=1.67) c5=6/8(H=0.85) c6=8/8(H=1.50) c7=8/8(H=1.60) c8=3/8(H=0.68) c9=5/8(H=0.20) c10=5/8(H=1.43) c11=6/8(H=1.37) c12=6/8(H=1.18) c13=4/8(H=0.84) c14=5/8(H=1.17) c15=5/8(H=1.46)
E0112 [5upd]  ep_rew=14.1378  rew_20=13.9156  L_geo=0.6284  L_rew=0.0004  L_chart=1.5743  L_crit=0.3394  L_bnd=0.6088  lr=0.0010  dt=18.29s
        recon=0.2175  vq=0.3261  code_H=1.1812  code_px=3.2682  ch_usage=0.1539  rtr_mrg=0.0025  enc_gn=9.3885
        ctrl=0.0001  tex=0.0285  im_rew=0.0476  im_ret=0.6901  value=0.0277  wm_gn=0.1700
        z_norm=0.6427  z_max=0.9422  jump=0.0000  cons=0.0475  sol=0.9999  e_var=0.0010  ch_ent=2.4477  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6901  dret=0.6662  term=0.0278  bnd=0.0358  chart_acc=0.5154  chart_ent=1.7016  rw_drift=0.0000
        v_err=1.5721  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0817
        bnd_x=0.0379  bell=0.0431  bell_s=0.0280  rtg_e=1.5721  rtg_b=-1.5720  cal_e=1.5720  u_l2=0.1344  cov_n=0.0012
        col=11.0395  smp=0.0050  enc_t=0.2532  bnd_t=0.0392  wm_t=1.0200  crt_t=0.0085  diag_t=0.5821
        charts: 16/16 active  c0=0.09 c1=0.03 c2=0.04 c3=0.22 c4=0.02 c5=0.03 c6=0.15 c7=0.01 c8=0.02 c9=0.07 c10=0.02 c11=0.10 c12=0.02 c13=0.05 c14=0.08 c15=0.04
        symbols: 90/128 active  c0=4/8(H=0.73) c1=3/8(H=0.54) c2=3/8(H=0.47) c3=8/8(H=1.60) c4=4/8(H=0.76) c5=5/8(H=0.66) c6=7/8(H=1.25) c7=5/8(H=1.21) c8=7/8(H=1.70) c9=6/8(H=1.38) c10=4/8(H=1.03) c11=6/8(H=1.01) c12=7/8(H=1.63) c13=7/8(H=1.27) c14=8/8(H=1.00) c15=6/8(H=1.38)
E0113 [5upd]  ep_rew=14.6190  rew_20=13.8551  L_geo=0.7740  L_rew=0.0004  L_chart=1.7428  L_crit=0.2719  L_bnd=0.6666  lr=0.0010  dt=18.34s
        recon=0.1908  vq=0.2800  code_H=1.2816  code_px=3.6119  ch_usage=0.0969  rtr_mrg=0.0013  enc_gn=6.7921
        ctrl=0.0002  tex=0.0302  im_rew=0.0368  im_ret=0.5468  value=0.0364  wm_gn=0.1996
        z_norm=0.6397  z_max=0.9219  jump=0.0000  cons=0.0424  sol=0.9986  e_var=0.0007  ch_ent=2.6197  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5468  dret=0.5144  term=0.0377  bnd=0.0630  chart_acc=0.4113  chart_ent=1.7827  rw_drift=0.0000
        v_err=1.2941  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0608
        bnd_x=0.0626  bell=0.0353  bell_s=0.0231  rtg_e=1.2941  rtg_b=-1.2939  cal_e=1.2939  u_l2=0.0979  cov_n=0.0018
        col=11.1319  smp=0.0029  enc_t=0.2531  bnd_t=0.0393  wm_t=1.0143  crt_t=0.0086  diag_t=0.5667
        charts: 16/16 active  c0=0.12 c1=0.09 c2=0.10 c3=0.12 c4=0.02 c5=0.04 c6=0.04 c7=0.06 c8=0.06 c9=0.04 c10=0.04 c11=0.10 c12=0.03 c13=0.10 c14=0.02 c15=0.04
        symbols: 100/128 active  c0=6/8(H=1.37) c1=4/8(H=0.76) c2=7/8(H=1.29) c3=5/8(H=1.16) c4=6/8(H=1.29) c5=5/8(H=1.06) c6=5/8(H=1.14) c7=7/8(H=1.66) c8=8/8(H=1.65) c9=4/8(H=0.80) c10=6/8(H=1.62) c11=7/8(H=1.57) c12=8/8(H=1.61) c13=8/8(H=1.79) c14=7/8(H=1.32) c15=7/8(H=1.35)
E0114 [5upd]  ep_rew=14.1445  rew_20=14.2051  L_geo=0.8122  L_rew=0.0006  L_chart=1.9909  L_crit=0.3099  L_bnd=0.6081  lr=0.0010  dt=18.25s
        recon=0.2520  vq=0.2354  code_H=1.3241  code_px=3.7842  ch_usage=0.0306  rtr_mrg=0.0022  enc_gn=4.7893
        ctrl=0.0001  tex=0.0283  im_rew=0.0352  im_ret=0.5224  value=0.0351  wm_gn=0.2638
        z_norm=0.6427  z_max=0.9216  jump=0.0000  cons=0.0532  sol=1.0027  e_var=0.0007  ch_ent=2.6317  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5224  dret=0.4927  term=0.0346  bnd=0.0603  chart_acc=0.3450  chart_ent=1.9745  rw_drift=0.0000
        v_err=1.4059  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1631
        bnd_x=0.0619  bell=0.0405  bell_s=0.0402  rtg_e=1.4059  rtg_b=-1.4058  cal_e=1.4058  u_l2=0.0559  cov_n=0.0013
        col=11.0153  smp=0.0039  enc_t=0.2531  bnd_t=0.0398  wm_t=1.0162  crt_t=0.0085  diag_t=0.5757
        charts: 16/16 active  c0=0.05 c1=0.06 c2=0.03 c3=0.15 c4=0.03 c5=0.04 c6=0.05 c7=0.07 c8=0.03 c9=0.12 c10=0.06 c11=0.11 c12=0.04 c13=0.09 c14=0.04 c15=0.05
        symbols: 106/128 active  c0=6/8(H=1.56) c1=7/8(H=0.98) c2=6/8(H=1.16) c3=7/8(H=1.56) c4=5/8(H=1.10) c5=6/8(H=1.33) c6=6/8(H=1.35) c7=7/8(H=1.51) c8=8/8(H=1.73) c9=8/8(H=1.50) c10=5/8(H=1.47) c11=8/8(H=1.37) c12=7/8(H=1.67) c13=7/8(H=1.54) c14=7/8(H=1.39) c15=6/8(H=1.24)
E0115 [5upd]  ep_rew=14.0536  rew_20=14.3979  L_geo=0.7013  L_rew=0.0005  L_chart=1.7531  L_crit=0.3183  L_bnd=0.6531  lr=0.0010  dt=18.26s
        recon=0.2171  vq=0.2784  code_H=1.2472  code_px=3.4910  ch_usage=0.0640  rtr_mrg=0.0016  enc_gn=5.3948
        ctrl=0.0001  tex=0.0288  im_rew=0.0413  im_ret=0.6000  value=0.0250  wm_gn=0.2062
        z_norm=0.6630  z_max=0.9442  jump=0.0000  cons=0.0522  sol=0.9991  e_var=0.0011  ch_ent=2.5462  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6000  dret=0.5784  term=0.0252  bnd=0.0374  chart_acc=0.4842  chart_ent=1.7589  rw_drift=0.0000
        v_err=1.5255  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0594
        bnd_x=0.0409  bell=0.0429  bell_s=0.0336  rtg_e=1.5255  rtg_b=-1.5255  cal_e=1.5255  u_l2=0.0356  cov_n=0.0015
        col=11.0245  smp=0.0042  enc_t=0.2556  bnd_t=0.0395  wm_t=1.0166  crt_t=0.0085  diag_t=0.5698
        charts: 16/16 active  c0=0.10 c1=0.19 c2=0.04 c3=0.09 c4=0.04 c5=0.04 c6=0.11 c7=0.01 c8=0.02 c9=0.02 c10=0.03 c11=0.09 c12=0.03 c13=0.08 c14=0.06 c15=0.05
        symbols: 87/128 active  c0=6/8(H=0.86) c1=4/8(H=0.73) c2=4/8(H=0.64) c3=8/8(H=1.44) c4=5/8(H=0.91) c5=4/8(H=0.39) c6=5/8(H=1.16) c7=5/8(H=1.12) c8=5/8(H=0.97) c9=5/8(H=1.50) c10=5/8(H=1.16) c11=8/8(H=1.09) c12=7/8(H=0.87) c13=5/8(H=0.82) c14=6/8(H=1.14) c15=5/8(H=1.40)
E0116 [5upd]  ep_rew=14.0805  rew_20=13.0136  L_geo=0.7422  L_rew=0.0005  L_chart=1.7077  L_crit=0.2607  L_bnd=0.5391  lr=0.0010  dt=17.55s
        recon=0.1802  vq=0.3138  code_H=1.2270  code_px=3.4440  ch_usage=0.0472  rtr_mrg=0.0016  enc_gn=6.1393
        ctrl=0.0001  tex=0.0254  im_rew=0.0409  im_ret=0.5915  value=0.0222  wm_gn=0.5170
        z_norm=0.7079  z_max=0.9659  jump=0.0000  cons=0.0411  sol=0.9998  e_var=0.0016  ch_ent=2.5789  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5915  dret=0.5726  term=0.0220  bnd=0.0331  chart_acc=0.4683  chart_ent=1.7622  rw_drift=0.0000
        v_err=1.3818  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4492
        bnd_x=0.0374  bell=0.0377  bell_s=0.0190  rtg_e=1.3818  rtg_b=-1.3818  cal_e=1.3818  u_l2=0.0458  cov_n=0.0017
        col=11.0614  smp=0.0030  enc_t=0.2474  bnd_t=0.0388  wm_t=0.9170  crt_t=0.0075  diag_t=0.3671
        charts: 16/16 active  c0=0.10 c1=0.15 c2=0.05 c3=0.13 c4=0.04 c5=0.05 c6=0.01 c7=0.07 c8=0.01 c9=0.11 c10=0.06 c11=0.05 c12=0.03 c13=0.04 c14=0.06 c15=0.04
        symbols: 82/128 active  c0=5/8(H=1.27) c1=6/8(H=1.20) c2=4/8(H=0.40) c3=6/8(H=0.88) c4=4/8(H=0.73) c5=5/8(H=0.93) c6=5/8(H=1.31) c7=7/8(H=1.54) c8=3/8(H=1.02) c9=7/8(H=1.02) c10=6/8(H=1.37) c11=5/8(H=1.03) c12=5/8(H=1.31) c13=4/8(H=0.84) c14=5/8(H=0.95) c15=5/8(H=1.43)
E0117 [5upd]  ep_rew=10.2514  rew_20=12.7385  L_geo=0.6891  L_rew=0.0007  L_chart=1.6358  L_crit=0.3126  L_bnd=0.4756  lr=0.0010  dt=12.48s
        recon=0.2049  vq=0.3082  code_H=1.1701  code_px=3.2387  ch_usage=0.0408  rtr_mrg=0.0016  enc_gn=9.2103
        ctrl=0.0001  tex=0.0283  im_rew=0.0381  im_ret=0.5536  value=0.0232  wm_gn=0.2751
        z_norm=0.6698  z_max=0.9347  jump=0.0000  cons=0.0480  sol=0.9983  e_var=0.0014  ch_ent=2.6940  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5536  dret=0.5338  term=0.0231  bnd=0.0371  chart_acc=0.4625  chart_ent=1.7255  rw_drift=0.0000
        v_err=1.4165  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8395
        bnd_x=0.0444  bell=0.0392  bell_s=0.0208  rtg_e=1.4165  rtg_b=-1.4165  cal_e=1.4165  u_l2=0.1266  cov_n=0.0012
        col=7.2782  smp=0.0036  enc_t=0.2234  bnd_t=0.0345  wm_t=0.6872  crt_t=0.0058  diag_t=0.3914
        charts: 16/16 active  c0=0.06 c1=0.06 c2=0.05 c3=0.07 c4=0.03 c5=0.07 c6=0.07 c7=0.06 c8=0.02 c9=0.11 c10=0.10 c11=0.10 c12=0.04 c13=0.04 c14=0.05 c15=0.06
        symbols: 102/128 active  c0=5/8(H=0.98) c1=6/8(H=1.34) c2=5/8(H=1.22) c3=6/8(H=1.24) c4=5/8(H=1.16) c5=6/8(H=1.56) c6=7/8(H=1.51) c7=8/8(H=1.67) c8=7/8(H=1.14) c9=7/8(H=1.13) c10=6/8(H=1.72) c11=8/8(H=1.22) c12=7/8(H=1.55) c13=6/8(H=1.38) c14=6/8(H=1.17) c15=7/8(H=1.40)
E0118 [5upd]  ep_rew=13.3983  rew_20=13.0860  L_geo=0.7392  L_rew=0.0003  L_chart=1.7863  L_crit=0.2720  L_bnd=0.5159  lr=0.0010  dt=12.77s
        recon=0.2035  vq=0.2701  code_H=1.2586  code_px=3.5285  ch_usage=0.0666  rtr_mrg=0.0036  enc_gn=5.2223
        ctrl=0.0001  tex=0.0284  im_rew=0.0353  im_ret=0.5135  value=0.0237  wm_gn=0.3674
        z_norm=0.6465  z_max=0.9312  jump=0.0000  cons=0.0547  sol=0.9946  e_var=0.0011  ch_ent=2.5909  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5135  dret=0.4932  term=0.0236  bnd=0.0412  chart_acc=0.3629  chart_ent=1.7273  rw_drift=0.0000
        v_err=1.2613  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6860
        bnd_x=0.0453  bell=0.0346  bell_s=0.0255  rtg_e=1.2613  rtg_b=-1.2612  cal_e=1.2612  u_l2=0.1702  cov_n=0.0011
        col=7.5199  smp=0.0021  enc_t=0.2247  bnd_t=0.0342  wm_t=0.6985  crt_t=0.0059  diag_t=0.3777
        charts: 16/16 active  c0=0.10 c1=0.03 c2=0.06 c3=0.18 c4=0.01 c5=0.10 c6=0.09 c7=0.05 c8=0.03 c9=0.05 c10=0.07 c11=0.07 c12=0.04 c13=0.07 c14=0.03 c15=0.04
        symbols: 88/128 active  c0=6/8(H=1.16) c1=3/8(H=0.76) c2=4/8(H=1.05) c3=6/8(H=1.51) c4=3/8(H=0.76) c5=5/8(H=1.22) c6=7/8(H=1.60) c7=7/8(H=0.93) c8=7/8(H=1.47) c9=4/8(H=0.62) c10=4/8(H=0.93) c11=7/8(H=0.92) c12=7/8(H=1.05) c13=7/8(H=1.64) c14=5/8(H=1.19) c15=6/8(H=1.14)
E0119 [5upd]  ep_rew=14.6176  rew_20=12.4883  L_geo=0.7806  L_rew=0.0006  L_chart=1.7609  L_crit=0.2781  L_bnd=0.6178  lr=0.0010  dt=12.61s
        recon=0.2006  vq=0.2807  code_H=1.3085  code_px=3.7231  ch_usage=0.1016  rtr_mrg=0.0026  enc_gn=7.7687
        ctrl=0.0001  tex=0.0278  im_rew=0.0356  im_ret=0.5181  value=0.0235  wm_gn=0.1885
        z_norm=0.6645  z_max=0.9538  jump=0.0000  cons=0.0499  sol=0.9973  e_var=0.0012  ch_ent=2.5148  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5181  dret=0.4975  term=0.0239  bnd=0.0414  chart_acc=0.4125  chart_ent=1.7696  rw_drift=0.0000
        v_err=1.1622  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3209
        bnd_x=0.0468  bell=0.0323  bell_s=0.0224  rtg_e=1.1622  rtg_b=-1.1621  cal_e=1.1621  u_l2=0.1600  cov_n=0.0016
        col=7.3196  smp=0.0029  enc_t=0.2264  bnd_t=0.0345  wm_t=0.7010  crt_t=0.0059  diag_t=0.3914
        charts: 16/16 active  c0=0.12 c1=0.07 c2=0.09 c3=0.06 c4=0.01 c5=0.19 c6=0.04 c7=0.03 c8=0.02 c9=0.12 c10=0.06 c11=0.04 c12=0.03 c13=0.08 c14=0.02 c15=0.02
        symbols: 83/128 active  c0=5/8(H=1.14) c1=4/8(H=0.51) c2=5/8(H=0.96) c3=4/8(H=0.92) c4=4/8(H=1.13) c5=6/8(H=1.19) c6=3/8(H=0.83) c7=5/8(H=1.54) c8=6/8(H=1.46) c9=4/8(H=1.18) c10=4/8(H=1.29) c11=7/8(H=1.39) c12=6/8(H=1.20) c13=8/8(H=1.22) c14=6/8(H=1.24) c15=6/8(H=1.66)
E0120 [5upd]  ep_rew=14.8525  rew_20=12.7695  L_geo=0.6343  L_rew=0.0003  L_chart=1.6523  L_crit=0.2558  L_bnd=0.6392  lr=0.0010  dt=13.11s
        recon=0.1909  vq=0.2817  code_H=1.2294  code_px=3.4270  ch_usage=0.1154  rtr_mrg=0.0021  enc_gn=6.3631
        ctrl=0.0001  tex=0.0315  im_rew=0.0374  im_ret=0.5390  value=0.0175  wm_gn=0.2232
        z_norm=0.6273  z_max=0.9218  jump=0.0000  cons=0.0412  sol=0.9974  e_var=0.0005  ch_ent=2.4575  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5390  dret=0.5237  term=0.0179  bnd=0.0293  chart_acc=0.4758  chart_ent=1.6924  rw_drift=0.0000
        v_err=1.2102  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2025
        bnd_x=0.0319  bell=0.0330  bell_s=0.0204  rtg_e=1.2102  rtg_b=-1.2102  cal_e=1.2102  u_l2=0.1128  cov_n=0.0012
        col=7.4638  smp=0.0037  enc_t=0.2288  bnd_t=0.0349  wm_t=0.7635  crt_t=0.0061  diag_t=0.4189
        charts: 16/16 active  c0=0.19 c1=0.05 c2=0.05 c3=0.15 c4=0.01 c5=0.10 c6=0.08 c7=0.05 c8=0.02 c9=0.01 c10=0.04 c11=0.12 c12=0.01 c13=0.06 c14=0.02 c15=0.03
        symbols: 91/128 active  c0=7/8(H=1.50) c1=4/8(H=0.40) c2=6/8(H=1.45) c3=5/8(H=1.18) c4=4/8(H=0.89) c5=6/8(H=0.85) c6=7/8(H=1.40) c7=8/8(H=1.60) c8=5/8(H=1.25) c9=4/8(H=1.12) c10=6/8(H=1.36) c11=7/8(H=1.09) c12=4/8(H=1.12) c13=6/8(H=0.81) c14=7/8(H=1.88) c15=5/8(H=1.09)
  EVAL  reward=13.0 +/- 4.1  len=300
E0121 [5upd]  ep_rew=13.3545  rew_20=12.8990  L_geo=0.7067  L_rew=0.0014  L_chart=1.7788  L_crit=0.2971  L_bnd=0.5020  lr=0.0010  dt=14.05s
        recon=0.2300  vq=0.2520  code_H=1.3107  code_px=3.7277  ch_usage=0.0667  rtr_mrg=0.0047  enc_gn=4.6445
        ctrl=0.0001  tex=0.0303  im_rew=0.0386  im_ret=0.5540  value=0.0165  wm_gn=0.2345
        z_norm=0.6330  z_max=0.9148  jump=0.0000  cons=0.0542  sol=0.9979  e_var=0.0007  ch_ent=2.6402  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5540  dret=0.5401  term=0.0161  bnd=0.0257  chart_acc=0.4258  chart_ent=1.7678  rw_drift=0.0000
        v_err=1.3201  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6647
        bnd_x=0.0273  bell=0.0380  bell_s=0.0424  rtg_e=1.3201  rtg_b=-1.3201  cal_e=1.3201  u_l2=0.0932  cov_n=0.0009
        col=7.4123  smp=0.0025  enc_t=0.2365  bnd_t=0.0366  wm_t=0.9024  crt_t=0.0075  diag_t=0.6531
        charts: 16/16 active  c0=0.14 c1=0.05 c2=0.06 c3=0.12 c4=0.04 c5=0.09 c6=0.07 c7=0.10 c8=0.03 c9=0.05 c10=0.05 c11=0.03 c12=0.07 c13=0.03 c14=0.03 c15=0.04
        symbols: 94/128 active  c0=5/8(H=1.34) c1=6/8(H=1.41) c2=6/8(H=1.07) c3=6/8(H=1.25) c4=5/8(H=1.28) c5=7/8(H=0.81) c6=6/8(H=1.36) c7=7/8(H=1.12) c8=6/8(H=1.04) c9=6/8(H=1.33) c10=5/8(H=1.48) c11=6/8(H=1.38) c12=6/8(H=1.54) c13=5/8(H=1.11) c14=6/8(H=1.26) c15=6/8(H=1.65)
E0122 [5upd]  ep_rew=14.2790  rew_20=13.4731  L_geo=0.6716  L_rew=0.0005  L_chart=1.7884  L_crit=0.3095  L_bnd=0.4958  lr=0.0010  dt=19.96s
        recon=0.2190  vq=0.2743  code_H=1.3283  code_px=3.7921  ch_usage=0.0893  rtr_mrg=0.0033  enc_gn=6.6973
        ctrl=0.0001  tex=0.0298  im_rew=0.0390  im_ret=0.5630  value=0.0208  wm_gn=0.2283
        z_norm=0.6483  z_max=0.9288  jump=0.0000  cons=0.0535  sol=0.9973  e_var=0.0008  ch_ent=2.6609  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5630  dret=0.5454  term=0.0205  bnd=0.0324  chart_acc=0.4133  chart_ent=1.7563  rw_drift=0.0000
        v_err=1.4980  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8081
        bnd_x=0.0338  bell=0.0405  bell_s=0.0277  rtg_e=1.4980  rtg_b=-1.4980  cal_e=1.4980  u_l2=0.1022  cov_n=0.0010
        col=12.3761  smp=0.0028  enc_t=0.2582  bnd_t=0.0406  wm_t=1.0774  crt_t=0.0092  diag_t=0.5908
        charts: 16/16 active  c0=0.04 c1=0.07 c2=0.11 c3=0.10 c4=0.03 c5=0.02 c6=0.04 c7=0.09 c8=0.02 c9=0.09 c10=0.05 c11=0.10 c12=0.06 c13=0.03 c14=0.06 c15=0.07
        symbols: 102/128 active  c0=4/8(H=0.92) c1=8/8(H=1.68) c2=6/8(H=0.52) c3=5/8(H=1.23) c4=5/8(H=1.06) c5=7/8(H=1.62) c6=6/8(H=1.13) c7=7/8(H=1.55) c8=7/8(H=1.42) c9=6/8(H=1.25) c10=7/8(H=1.35) c11=8/8(H=0.95) c12=7/8(H=1.79) c13=5/8(H=1.18) c14=7/8(H=1.55) c15=7/8(H=1.80)
E0123 [5upd]  ep_rew=11.3484  rew_20=13.0215  L_geo=0.7214  L_rew=0.0004  L_chart=1.7355  L_crit=0.2559  L_bnd=0.5150  lr=0.0010  dt=19.13s
        recon=0.1839  vq=0.2845  code_H=1.2054  code_px=3.3712  ch_usage=0.0795  rtr_mrg=0.0008  enc_gn=5.7622
        ctrl=0.0001  tex=0.0273  im_rew=0.0372  im_ret=0.5450  value=0.0295  wm_gn=0.1879
        z_norm=0.6951  z_max=0.9636  jump=0.0000  cons=0.0486  sol=0.9952  e_var=0.0025  ch_ent=2.6037  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5450  dret=0.5206  term=0.0284  bnd=0.0470  chart_acc=0.4517  chart_ent=1.7728  rw_drift=0.0000
        v_err=1.2082  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6713
        bnd_x=0.0612  bell=0.0338  bell_s=0.0224  rtg_e=1.2082  rtg_b=-1.2080  cal_e=1.2080  u_l2=0.1111  cov_n=0.0014
        col=11.5033  smp=0.0042  enc_t=0.2580  bnd_t=0.0409  wm_t=1.0806  crt_t=0.0091  diag_t=0.6090
        charts: 16/16 active  c0=0.09 c1=0.14 c2=0.07 c3=0.14 c4=0.02 c5=0.10 c6=0.04 c7=0.06 c8=0.04 c9=0.07 c10=0.06 c11=0.01 c12=0.05 c13=0.06 c14=0.04 c15=0.05
        symbols: 82/128 active  c0=4/8(H=0.67) c1=5/8(H=1.06) c2=5/8(H=0.60) c3=4/8(H=0.25) c4=6/8(H=1.50) c5=3/8(H=0.69) c6=6/8(H=1.02) c7=8/8(H=1.56) c8=3/8(H=0.62) c9=6/8(H=1.38) c10=6/8(H=1.19) c11=4/8(H=1.33) c12=7/8(H=1.66) c13=4/8(H=0.65) c14=5/8(H=1.11) c15=6/8(H=1.60)
E0124 [5upd]  ep_rew=14.3688  rew_20=13.1535  L_geo=0.6556  L_rew=0.0005  L_chart=1.6940  L_crit=0.2724  L_bnd=0.5587  lr=0.0010  dt=18.76s
        recon=0.2014  vq=0.2821  code_H=1.2436  code_px=3.4733  ch_usage=0.0767  rtr_mrg=0.0027  enc_gn=5.8709
        ctrl=0.0001  tex=0.0296  im_rew=0.0403  im_ret=0.5915  value=0.0323  wm_gn=0.2037
        z_norm=0.6332  z_max=0.9454  jump=0.0000  cons=0.0511  sol=0.9975  e_var=0.0013  ch_ent=2.6396  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5915  dret=0.5638  term=0.0322  bnd=0.0491  chart_acc=0.4642  chart_ent=1.7452  rw_drift=0.0000
        v_err=1.4405  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5162
        bnd_x=0.0551  bell=0.0407  bell_s=0.0364  rtg_e=1.4405  rtg_b=-1.4404  cal_e=1.4404  u_l2=0.0882  cov_n=0.0014
        col=11.4273  smp=0.0033  enc_t=0.2545  bnd_t=0.0398  wm_t=1.0340  crt_t=0.0088  diag_t=0.5809
        charts: 16/16 active  c0=0.07 c1=0.08 c2=0.01 c3=0.13 c4=0.02 c5=0.04 c6=0.05 c7=0.06 c8=0.06 c9=0.11 c10=0.08 c11=0.10 c12=0.06 c13=0.03 c14=0.04 c15=0.05
        symbols: 89/128 active  c0=4/8(H=0.80) c1=6/8(H=1.53) c2=4/8(H=0.86) c3=5/8(H=0.43) c4=3/8(H=0.86) c5=6/8(H=1.04) c6=7/8(H=1.38) c7=6/8(H=0.87) c8=5/8(H=1.24) c9=7/8(H=1.47) c10=6/8(H=1.29) c11=6/8(H=1.12) c12=6/8(H=1.21) c13=6/8(H=1.62) c14=6/8(H=1.49) c15=6/8(H=1.51)
E0125 [5upd]  ep_rew=19.7778  rew_20=13.3552  L_geo=0.8075  L_rew=0.0008  L_chart=1.9041  L_crit=0.3154  L_bnd=0.5962  lr=0.0010  dt=18.61s
        recon=0.2098  vq=0.2417  code_H=1.3195  code_px=3.7628  ch_usage=0.0738  rtr_mrg=0.0033  enc_gn=5.6106
        ctrl=0.0001  tex=0.0297  im_rew=0.0375  im_ret=0.5508  value=0.0296  wm_gn=0.2110
        z_norm=0.6436  z_max=0.9612  jump=0.0000  cons=0.0537  sol=0.9962  e_var=0.0006  ch_ent=2.6513  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5508  dret=0.5250  term=0.0300  bnd=0.0491  chart_acc=0.3904  chart_ent=1.9048  rw_drift=0.0000
        v_err=1.2909  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2561
        bnd_x=0.0549  bell=0.0364  bell_s=0.0294  rtg_e=1.2909  rtg_b=-1.2908  cal_e=1.2908  u_l2=0.0722  cov_n=0.0014
        col=11.2374  smp=0.0038  enc_t=0.2555  bnd_t=0.0400  wm_t=1.0392  crt_t=0.0092  diag_t=0.5830
        charts: 16/16 active  c0=0.08 c1=0.07 c2=0.04 c3=0.10 c4=0.01 c5=0.07 c6=0.05 c7=0.04 c8=0.03 c9=0.09 c10=0.08 c11=0.09 c12=0.03 c13=0.13 c14=0.04 c15=0.06
        symbols: 102/128 active  c0=6/8(H=1.52) c1=6/8(H=1.50) c2=7/8(H=1.48) c3=8/8(H=1.18) c4=5/8(H=1.41) c5=7/8(H=0.70) c6=6/8(H=1.16) c7=6/8(H=1.45) c8=7/8(H=1.81) c9=5/8(H=0.71) c10=7/8(H=1.52) c11=7/8(H=0.86) c12=6/8(H=1.52) c13=6/8(H=1.37) c14=5/8(H=0.87) c15=8/8(H=1.58)
E0126 [5upd]  ep_rew=20.7463  rew_20=14.0177  L_geo=0.7646  L_rew=0.0010  L_chart=1.7807  L_crit=0.2694  L_bnd=0.6387  lr=0.0010  dt=19.15s
        recon=0.2049  vq=0.2621  code_H=1.3561  code_px=3.8985  ch_usage=0.0964  rtr_mrg=0.0029  enc_gn=4.8451
        ctrl=0.0001  tex=0.0267  im_rew=0.0335  im_ret=0.4884  value=0.0225  wm_gn=0.1980
        z_norm=0.7023  z_max=0.9413  jump=0.0000  cons=0.0421  sol=0.9978  e_var=0.0010  ch_ent=2.5942  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4884  dret=0.4688  term=0.0229  bnd=0.0420  chart_acc=0.4604  chart_ent=1.7261  rw_drift=0.0000
        v_err=1.2391  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1380
        bnd_x=0.0471  bell=0.0350  bell_s=0.0370  rtg_e=1.2391  rtg_b=-1.2391  cal_e=1.2391  u_l2=0.0562  cov_n=0.0011
        col=11.3075  smp=0.0021  enc_t=0.2881  bnd_t=0.0409  wm_t=1.1025  crt_t=0.0095  diag_t=0.5742
        charts: 16/16 active  c0=0.08 c1=0.05 c2=0.07 c3=0.07 c4=0.01 c5=0.17 c6=0.02 c7=0.06 c8=0.03 c9=0.13 c10=0.06 c11=0.03 c12=0.03 c13=0.07 c14=0.06 c15=0.05
        symbols: 99/128 active  c0=6/8(H=1.30) c1=7/8(H=1.24) c2=6/8(H=1.09) c3=6/8(H=0.71) c4=3/8(H=0.69) c5=6/8(H=0.95) c6=6/8(H=0.86) c7=6/8(H=1.45) c8=8/8(H=1.47) c9=6/8(H=0.37) c10=7/8(H=1.67) c11=5/8(H=0.90) c12=6/8(H=1.39) c13=7/8(H=1.45) c14=8/8(H=0.77) c15=6/8(H=1.48)
E0127 [5upd]  ep_rew=15.1355  rew_20=13.6109  L_geo=0.6526  L_rew=0.0007  L_chart=1.5438  L_crit=0.3038  L_bnd=0.5877  lr=0.0010  dt=18.48s
        recon=0.1714  vq=0.3242  code_H=1.1692  code_px=3.2241  ch_usage=0.0656  rtr_mrg=0.0026  enc_gn=6.1557
        ctrl=0.0001  tex=0.0266  im_rew=0.0399  im_ret=0.5780  value=0.0225  wm_gn=0.1608
        z_norm=0.6865  z_max=0.9478  jump=0.0000  cons=0.0598  sol=0.9994  e_var=0.0010  ch_ent=2.6093  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5780  dret=0.5589  term=0.0222  bnd=0.0341  chart_acc=0.5254  chart_ent=1.6159  rw_drift=0.0000
        v_err=1.3547  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2782
        bnd_x=0.0395  bell=0.0383  bell_s=0.0295  rtg_e=1.3547  rtg_b=-1.3547  cal_e=1.3547  u_l2=0.0573  cov_n=0.0012
        col=11.1605  smp=0.0031  enc_t=0.2537  bnd_t=0.0400  wm_t=1.0328  crt_t=0.0087  diag_t=0.5726
        charts: 16/16 active  c0=0.07 c1=0.04 c2=0.09 c3=0.14 c4=0.04 c5=0.08 c6=0.06 c7=0.04 c8=0.01 c9=0.14 c10=0.06 c11=0.03 c12=0.03 c13=0.08 c14=0.04 c15=0.05
        symbols: 88/128 active  c0=5/8(H=1.09) c1=6/8(H=1.24) c2=5/8(H=0.99) c3=5/8(H=1.31) c4=6/8(H=1.13) c5=5/8(H=1.00) c6=5/8(H=1.38) c7=7/8(H=1.58) c8=4/8(H=1.04) c9=8/8(H=1.56) c10=4/8(H=1.35) c11=4/8(H=0.77) c12=6/8(H=1.39) c13=5/8(H=1.42) c14=6/8(H=1.18) c15=7/8(H=1.60)
E0128 [5upd]  ep_rew=14.6275  rew_20=14.7387  L_geo=0.6821  L_rew=0.0007  L_chart=1.6389  L_crit=0.2934  L_bnd=0.5625  lr=0.0010  dt=18.49s
        recon=0.1879  vq=0.3135  code_H=1.2732  code_px=3.5801  ch_usage=0.0581  rtr_mrg=0.0044  enc_gn=5.6977
        ctrl=0.0001  tex=0.0276  im_rew=0.0367  im_ret=0.5366  value=0.0273  wm_gn=0.1842
        z_norm=0.6852  z_max=0.9528  jump=0.0000  cons=0.0585  sol=0.9999  e_var=0.0010  ch_ent=2.6382  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5366  dret=0.5135  term=0.0269  bnd=0.0450  chart_acc=0.4546  chart_ent=1.7031  rw_drift=0.0000
        v_err=1.1505  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4899
        bnd_x=0.0502  bell=0.0337  bell_s=0.0343  rtg_e=1.1505  rtg_b=-1.1504  cal_e=1.1504  u_l2=0.0589  cov_n=0.0016
        col=11.1717  smp=0.0026  enc_t=0.2539  bnd_t=0.0401  wm_t=1.0312  crt_t=0.0087  diag_t=0.5796
        charts: 16/16 active  c0=0.05 c1=0.07 c2=0.09 c3=0.05 c4=0.01 c5=0.14 c6=0.08 c7=0.08 c8=0.01 c9=0.06 c10=0.09 c11=0.02 c12=0.04 c13=0.05 c14=0.08 c15=0.07
        symbols: 86/128 active  c0=4/8(H=0.96) c1=5/8(H=0.84) c2=6/8(H=1.31) c3=5/8(H=0.92) c4=6/8(H=1.42) c5=5/8(H=0.98) c6=6/8(H=1.56) c7=7/8(H=1.53) c8=4/8(H=0.94) c9=5/8(H=1.19) c10=5/8(H=1.27) c11=7/8(H=1.62) c12=5/8(H=1.17) c13=3/8(H=1.07) c14=6/8(H=0.93) c15=7/8(H=1.71)
E0129 [5upd]  ep_rew=14.2956  rew_20=15.5509  L_geo=0.6635  L_rew=0.0008  L_chart=1.7575  L_crit=0.3047  L_bnd=0.5669  lr=0.0010  dt=18.53s
        recon=0.2036  vq=0.2867  code_H=1.2467  code_px=3.4829  ch_usage=0.0338  rtr_mrg=0.0029  enc_gn=5.9119
        ctrl=0.0001  tex=0.0284  im_rew=0.0392  im_ret=0.5704  value=0.0255  wm_gn=0.2536
        z_norm=0.6550  z_max=0.9240  jump=0.0000  cons=0.0669  sol=0.9961  e_var=0.0008  ch_ent=2.6358  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5704  dret=0.5490  term=0.0249  bnd=0.0390  chart_acc=0.4183  chart_ent=1.7722  rw_drift=0.0000
        v_err=1.4829  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3397
        bnd_x=0.0452  bell=0.0409  bell_s=0.0302  rtg_e=1.4829  rtg_b=-1.4828  cal_e=1.4828  u_l2=0.0704  cov_n=0.0014
        col=11.2203  smp=0.0046  enc_t=0.2543  bnd_t=0.0398  wm_t=1.0319  crt_t=0.0085  diag_t=0.5697
        charts: 16/16 active  c0=0.04 c1=0.06 c2=0.09 c3=0.12 c4=0.08 c5=0.06 c6=0.10 c7=0.05 c8=0.01 c9=0.13 c10=0.03 c11=0.03 c12=0.04 c13=0.06 c14=0.04 c15=0.06
        symbols: 80/128 active  c0=5/8(H=1.14) c1=7/8(H=0.84) c2=7/8(H=1.64) c3=5/8(H=0.86) c4=4/8(H=1.05) c5=6/8(H=0.90) c6=5/8(H=1.45) c7=7/8(H=1.57) c8=3/8(H=0.43) c9=6/8(H=1.39) c10=4/8(H=1.19) c11=4/8(H=1.00) c12=5/8(H=1.25) c13=2/8(H=0.27) c14=5/8(H=1.18) c15=5/8(H=1.21)
E0130 [5upd]  ep_rew=14.2179  rew_20=15.1011  L_geo=0.7260  L_rew=0.0004  L_chart=1.8781  L_crit=0.2622  L_bnd=0.6749  lr=0.0010  dt=18.53s
        recon=0.2163  vq=0.2786  code_H=1.2991  code_px=3.6940  ch_usage=0.1045  rtr_mrg=0.0029  enc_gn=12.0006
        ctrl=0.0001  tex=0.0321  im_rew=0.0410  im_ret=0.5901  value=0.0198  wm_gn=0.3236
        z_norm=0.6228  z_max=0.9183  jump=0.0000  cons=0.0472  sol=0.9951  e_var=0.0006  ch_ent=2.5165  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5901  dret=0.5731  term=0.0198  bnd=0.0297  chart_acc=0.4058  chart_ent=1.7820  rw_drift=0.0000
        v_err=1.4646  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0440
        bnd_x=0.0323  bell=0.0400  bell_s=0.0196  rtg_e=1.4646  rtg_b=-1.4645  cal_e=1.4645  u_l2=0.0741  cov_n=0.0010
        col=11.2623  smp=0.0039  enc_t=0.2534  bnd_t=0.0399  wm_t=1.0238  crt_t=0.0087  diag_t=0.5729
        charts: 16/16 active  c0=0.11 c1=0.03 c2=0.01 c3=0.13 c4=0.03 c5=0.04 c6=0.08 c7=0.03 c8=0.04 c9=0.03 c10=0.03 c11=0.19 c12=0.03 c13=0.07 c14=0.10 c15=0.04
        symbols: 95/128 active  c0=5/8(H=1.12) c1=5/8(H=1.29) c2=5/8(H=1.41) c3=7/8(H=1.70) c4=4/8(H=1.20) c5=7/8(H=1.03) c6=4/8(H=1.09) c7=7/8(H=1.66) c8=8/8(H=1.38) c9=5/8(H=1.39) c10=4/8(H=1.09) c11=8/8(H=1.41) c12=7/8(H=1.56) c13=7/8(H=1.45) c14=7/8(H=1.34) c15=5/8(H=1.42)
E0131 [5upd]  ep_rew=7.2913  rew_20=14.5545  L_geo=0.7059  L_rew=0.0006  L_chart=1.8389  L_crit=0.3097  L_bnd=0.6316  lr=0.0010  dt=18.50s
        recon=0.2599  vq=0.2676  code_H=1.3999  code_px=4.0671  ch_usage=0.1414  rtr_mrg=0.0043  enc_gn=5.6836
        ctrl=0.0001  tex=0.0335  im_rew=0.0428  im_ret=0.6118  value=0.0157  wm_gn=0.2345
        z_norm=0.5906  z_max=0.9284  jump=0.0000  cons=0.0570  sol=0.9979  e_var=0.0005  ch_ent=2.6014  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6118  dret=0.5985  term=0.0155  bnd=0.0223  chart_acc=0.3950  chart_ent=1.8155  rw_drift=0.0000
        v_err=1.3927  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2689
        bnd_x=0.0248  bell=0.0386  bell_s=0.0203  rtg_e=1.3927  rtg_b=-1.3927  cal_e=1.3927  u_l2=0.0712  cov_n=0.0007
        col=11.1072  smp=0.0046  enc_t=0.2534  bnd_t=0.0397  wm_t=1.0486  crt_t=0.0086  diag_t=0.5733
        charts: 16/16 active  c0=0.12 c1=0.09 c2=0.02 c3=0.18 c4=0.02 c5=0.07 c6=0.06 c7=0.07 c8=0.02 c9=0.05 c10=0.05 c11=0.02 c12=0.07 c13=0.05 c14=0.05 c15=0.05
        symbols: 92/128 active  c0=4/8(H=0.93) c1=4/8(H=0.84) c2=6/8(H=1.33) c3=7/8(H=0.98) c4=4/8(H=1.00) c5=7/8(H=1.52) c6=6/8(H=1.43) c7=7/8(H=1.12) c8=6/8(H=1.10) c9=4/8(H=0.88) c10=7/8(H=1.66) c11=6/8(H=1.52) c12=7/8(H=1.57) c13=6/8(H=1.54) c14=6/8(H=1.33) c15=5/8(H=1.08)
E0132 [5upd]  ep_rew=16.1887  rew_20=14.7153  L_geo=0.6781  L_rew=0.0003  L_chart=1.8059  L_crit=0.2524  L_bnd=0.5962  lr=0.0010  dt=18.46s
        recon=0.2112  vq=0.2516  code_H=1.2259  code_px=3.4175  ch_usage=0.0930  rtr_mrg=0.0028  enc_gn=6.8758
        ctrl=0.0001  tex=0.0294  im_rew=0.0301  im_ret=0.4372  value=0.0191  wm_gn=0.2021
        z_norm=0.6228  z_max=0.9626  jump=0.0000  cons=0.0393  sol=0.9987  e_var=0.0008  ch_ent=2.5091  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4372  dret=0.4206  term=0.0193  bnd=0.0395  chart_acc=0.4546  chart_ent=1.7943  rw_drift=0.0000
        v_err=1.1708  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2451
        bnd_x=0.0478  bell=0.0317  bell_s=0.0134  rtg_e=1.1708  rtg_b=-1.1708  cal_e=1.1708  u_l2=0.0790  cov_n=0.0007
        col=11.1990  smp=0.0037  enc_t=0.2531  bnd_t=0.0398  wm_t=1.0229  crt_t=0.0087  diag_t=0.5716
        charts: 16/16 active  c0=0.19 c1=0.08 c2=0.04 c3=0.15 c4=0.02 c5=0.10 c6=0.05 c7=0.06 c8=0.02 c9=0.02 c10=0.05 c11=0.01 c12=0.05 c13=0.08 c14=0.03 c15=0.06
        symbols: 83/128 active  c0=5/8(H=1.40) c1=4/8(H=1.28) c2=4/8(H=1.03) c3=7/8(H=0.83) c4=4/8(H=0.83) c5=5/8(H=1.10) c6=6/8(H=1.13) c7=8/8(H=1.66) c8=4/8(H=1.17) c9=4/8(H=1.18) c10=5/8(H=1.29) c11=5/8(H=1.35) c12=7/8(H=1.75) c13=5/8(H=0.75) c14=5/8(H=1.09) c15=5/8(H=1.43)
E0133 [5upd]  ep_rew=8.4287  rew_20=13.8300  L_geo=0.7199  L_rew=0.0010  L_chart=1.9260  L_crit=0.3044  L_bnd=0.6911  lr=0.0010  dt=18.41s
        recon=0.2545  vq=0.2229  code_H=1.2764  code_px=3.6126  ch_usage=0.1046  rtr_mrg=0.0032  enc_gn=6.7812
        ctrl=0.0001  tex=0.0313  im_rew=0.0382  im_ret=0.5548  value=0.0246  wm_gn=0.1966
        z_norm=0.6162  z_max=0.8896  jump=0.0000  cons=0.0575  sol=0.9984  e_var=0.0005  ch_ent=2.6752  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5548  dret=0.5340  term=0.0242  bnd=0.0389  chart_acc=0.4050  chart_ent=1.9235  rw_drift=0.0000
        v_err=1.5044  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0449
        bnd_x=0.0457  bell=0.0409  bell_s=0.0320  rtg_e=1.5044  rtg_b=-1.5043  cal_e=1.5043  u_l2=0.0863  cov_n=0.0010
        col=11.1418  smp=0.0040  enc_t=0.2554  bnd_t=0.0401  wm_t=1.0219  crt_t=0.0086  diag_t=0.5638
        charts: 16/16 active  c0=0.16 c1=0.09 c2=0.05 c3=0.08 c4=0.04 c5=0.05 c6=0.05 c7=0.06 c8=0.02 c9=0.06 c10=0.05 c11=0.05 c12=0.05 c13=0.07 c14=0.06 c15=0.05
        symbols: 105/128 active  c0=6/8(H=1.40) c1=6/8(H=1.40) c2=7/8(H=1.39) c3=6/8(H=0.98) c4=7/8(H=1.55) c5=7/8(H=1.07) c6=6/8(H=1.03) c7=8/8(H=1.43) c8=7/8(H=1.51) c9=5/8(H=1.43) c10=6/8(H=1.47) c11=7/8(H=1.02) c12=8/8(H=1.58) c13=6/8(H=1.22) c14=7/8(H=1.37) c15=6/8(H=1.66)
E0134 [5upd]  ep_rew=11.0236  rew_20=13.7938  L_geo=0.7808  L_rew=0.0008  L_chart=1.9724  L_crit=0.3180  L_bnd=0.7580  lr=0.0010  dt=18.41s
        recon=0.2496  vq=0.2360  code_H=1.3146  code_px=3.7387  ch_usage=0.0685  rtr_mrg=0.0030  enc_gn=4.9202
        ctrl=0.0001  tex=0.0314  im_rew=0.0368  im_ret=0.5324  value=0.0197  wm_gn=0.3609
        z_norm=0.6134  z_max=0.9711  jump=0.0000  cons=0.0595  sol=0.9937  e_var=0.0004  ch_ent=2.5588  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5324  dret=0.5155  term=0.0197  bnd=0.0329  chart_acc=0.3746  chart_ent=1.9355  rw_drift=0.0000
        v_err=1.1748  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1777
        bnd_x=0.0380  bell=0.0319  bell_s=0.0138  rtg_e=1.1748  rtg_b=-1.1748  cal_e=1.1748  u_l2=0.0906  cov_n=0.0009
        col=11.1565  smp=0.0040  enc_t=0.2554  bnd_t=0.0399  wm_t=1.0170  crt_t=0.0086  diag_t=0.5782
        charts: 16/16 active  c0=0.22 c1=0.09 c2=0.07 c3=0.09 c4=0.02 c5=0.07 c6=0.03 c7=0.04 c8=0.03 c9=0.03 c10=0.07 c11=0.03 c12=0.03 c13=0.09 c14=0.03 c15=0.07
        symbols: 92/128 active  c0=5/8(H=1.19) c1=7/8(H=1.46) c2=8/8(H=1.50) c3=7/8(H=0.94) c4=5/8(H=1.54) c5=5/8(H=1.12) c6=6/8(H=1.48) c7=5/8(H=1.11) c8=5/8(H=1.28) c9=5/8(H=1.10) c10=6/8(H=1.31) c11=6/8(H=1.27) c12=5/8(H=1.37) c13=8/8(H=1.66) c14=4/8(H=1.20) c15=5/8(H=1.59)
E0135 [5upd]  ep_rew=39.5230  rew_20=15.2777  L_geo=0.7551  L_rew=0.0009  L_chart=1.9197  L_crit=0.3096  L_bnd=0.7204  lr=0.0010  dt=18.34s
        recon=0.2457  vq=0.2673  code_H=1.2841  code_px=3.6434  ch_usage=0.0492  rtr_mrg=0.0023  enc_gn=7.0556
        ctrl=0.0001  tex=0.0275  im_rew=0.0376  im_ret=0.5390  value=0.0158  wm_gn=0.5014
        z_norm=0.6498  z_max=0.9346  jump=0.0000  cons=0.0745  sol=0.9932  e_var=0.0008  ch_ent=2.5721  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5390  dret=0.5259  term=0.0153  bnd=0.0250  chart_acc=0.4704  chart_ent=1.8444  rw_drift=0.0000
        v_err=1.3200  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0085
        bnd_x=0.0293  bell=0.0356  bell_s=0.0160  rtg_e=1.3200  rtg_b=-1.3200  cal_e=1.3200  u_l2=0.1063  cov_n=0.0009
        col=11.0890  smp=0.0030  enc_t=0.2531  bnd_t=0.0394  wm_t=1.0202  crt_t=0.0086  diag_t=0.5770
        charts: 16/16 active  c0=0.17 c1=0.15 c2=0.03 c3=0.06 c4=0.04 c5=0.08 c6=0.04 c7=0.02 c8=0.03 c9=0.07 c10=0.03 c11=0.02 c12=0.03 c13=0.08 c14=0.08 c15=0.05
        symbols: 99/128 active  c0=7/8(H=1.43) c1=7/8(H=1.37) c2=6/8(H=1.03) c3=5/8(H=0.39) c4=5/8(H=1.30) c5=6/8(H=0.48) c6=6/8(H=0.87) c7=6/8(H=1.72) c8=7/8(H=1.52) c9=8/8(H=1.72) c10=5/8(H=1.24) c11=7/8(H=1.45) c12=6/8(H=1.38) c13=6/8(H=1.47) c14=7/8(H=1.29) c15=5/8(H=1.21)
E0136 [5upd]  ep_rew=14.9696  rew_20=15.0926  L_geo=0.7571  L_rew=0.0007  L_chart=1.7829  L_crit=0.2977  L_bnd=0.6108  lr=0.0010  dt=18.54s
        recon=0.2286  vq=0.2934  code_H=1.3015  code_px=3.6890  ch_usage=0.0659  rtr_mrg=0.0029  enc_gn=8.1664
        ctrl=0.0001  tex=0.0280  im_rew=0.0365  im_ret=0.5273  value=0.0195  wm_gn=0.1711
        z_norm=0.6705  z_max=0.9442  jump=0.0000  cons=0.0524  sol=1.0019  e_var=0.0013  ch_ent=2.5800  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5273  dret=0.5110  term=0.0190  bnd=0.0319  chart_acc=0.4454  chart_ent=1.8455  rw_drift=0.0000
        v_err=1.3677  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3551
        bnd_x=0.0445  bell=0.0381  bell_s=0.0257  rtg_e=1.3677  rtg_b=-1.3677  cal_e=1.3677  u_l2=0.1135  cov_n=0.0009
        col=11.1488  smp=0.0035  enc_t=0.2531  bnd_t=0.0398  wm_t=1.0508  crt_t=0.0085  diag_t=0.5641
        charts: 16/16 active  c0=0.09 c1=0.07 c2=0.05 c3=0.16 c4=0.04 c5=0.10 c6=0.04 c7=0.05 c8=0.01 c9=0.09 c10=0.11 c11=0.05 c12=0.02 c13=0.06 c14=0.01 c15=0.04
        symbols: 102/128 active  c0=7/8(H=0.98) c1=6/8(H=1.26) c2=6/8(H=1.05) c3=6/8(H=1.12) c4=6/8(H=1.56) c5=6/8(H=1.02) c6=5/8(H=1.06) c7=7/8(H=0.84) c8=6/8(H=1.53) c9=6/8(H=0.95) c10=6/8(H=1.52) c11=7/8(H=1.16) c12=6/8(H=1.18) c13=8/8(H=1.35) c14=7/8(H=1.70) c15=7/8(H=1.44)
E0137 [5upd]  ep_rew=24.6196  rew_20=15.7288  L_geo=0.6167  L_rew=0.0004  L_chart=1.6673  L_crit=0.2797  L_bnd=0.5875  lr=0.0010  dt=18.40s
        recon=0.2100  vq=0.3215  code_H=1.2105  code_px=3.3618  ch_usage=0.1331  rtr_mrg=0.0017  enc_gn=7.5838
        ctrl=0.0001  tex=0.0279  im_rew=0.0430  im_ret=0.6198  value=0.0211  wm_gn=0.2200
        z_norm=0.6734  z_max=0.9607  jump=0.0000  cons=0.0635  sol=0.9921  e_var=0.0007  ch_ent=2.6451  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6198  dret=0.6018  term=0.0209  bnd=0.0298  chart_acc=0.4950  chart_ent=1.8052  rw_drift=0.0000
        v_err=1.6965  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4551
        bnd_x=0.0367  bell=0.0455  bell_s=0.0373  rtg_e=1.6965  rtg_b=-1.6964  cal_e=1.6964  u_l2=0.1058  cov_n=0.0012
        col=11.1433  smp=0.0028  enc_t=0.2541  bnd_t=0.0395  wm_t=1.0193  crt_t=0.0087  diag_t=0.5808
        charts: 16/16 active  c0=0.13 c1=0.07 c2=0.07 c3=0.03 c4=0.11 c5=0.08 c6=0.06 c7=0.05 c8=0.04 c9=0.08 c10=0.07 c11=0.07 c12=0.02 c13=0.06 c14=0.02 c15=0.03
        symbols: 103/128 active  c0=6/8(H=1.35) c1=5/8(H=1.50) c2=7/8(H=1.02) c3=4/8(H=0.65) c4=6/8(H=1.34) c5=8/8(H=0.85) c6=6/8(H=1.27) c7=7/8(H=1.06) c8=7/8(H=1.09) c9=6/8(H=1.27) c10=7/8(H=1.06) c11=7/8(H=1.51) c12=7/8(H=1.54) c13=7/8(H=1.21) c14=7/8(H=1.57) c15=6/8(H=1.39)
E0138 [5upd]  ep_rew=15.4214  rew_20=15.9589  L_geo=0.6706  L_rew=0.0012  L_chart=1.7584  L_crit=0.3296  L_bnd=0.6870  lr=0.0010  dt=14.26s
        recon=0.2194  vq=0.2637  code_H=1.2577  code_px=3.5253  ch_usage=0.0745  rtr_mrg=0.0027  enc_gn=8.3732
        ctrl=0.0001  tex=0.0292  im_rew=0.0392  im_ret=0.5654  value=0.0192  wm_gn=0.3328
        z_norm=0.6428  z_max=0.9579  jump=0.0000  cons=0.0557  sol=0.9958  e_var=0.0007  ch_ent=2.4965  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5654  dret=0.5488  term=0.0194  bnd=0.0303  chart_acc=0.4558  chart_ent=1.7649  rw_drift=0.0000
        v_err=1.2758  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0435
        bnd_x=0.0344  bell=0.0355  bell_s=0.0290  rtg_e=1.2758  rtg_b=-1.2757  cal_e=1.2757  u_l2=0.1154  cov_n=0.0016
        col=8.9125  smp=0.0026  enc_t=0.2253  bnd_t=0.0344  wm_t=0.7141  crt_t=0.0060  diag_t=0.3951
        charts: 16/16 active  c0=0.16 c1=0.09 c2=0.04 c3=0.15 c4=0.02 c5=0.15 c6=0.04 c7=0.03 c8=0.01 c9=0.02 c10=0.06 c11=0.05 c12=0.02 c13=0.08 c14=0.02 c15=0.04
        symbols: 96/128 active  c0=5/8(H=1.31) c1=5/8(H=1.33) c2=8/8(H=1.66) c3=5/8(H=1.03) c4=4/8(H=1.26) c5=8/8(H=0.98) c6=6/8(H=1.35) c7=6/8(H=1.64) c8=7/8(H=1.66) c9=4/8(H=0.70) c10=6/8(H=1.34) c11=8/8(H=1.18) c12=4/8(H=1.29) c13=8/8(H=1.37) c14=6/8(H=1.41) c15=6/8(H=1.54)
E0139 [5upd]  ep_rew=14.3748  rew_20=15.9239  L_geo=0.6814  L_rew=0.0006  L_chart=1.7238  L_crit=0.3083  L_bnd=0.7189  lr=0.0010  dt=12.55s
        recon=0.2161  vq=0.2497  code_H=1.2768  code_px=3.5906  ch_usage=0.0571  rtr_mrg=0.0021  enc_gn=6.0015
        ctrl=0.0001  tex=0.0308  im_rew=0.0368  im_ret=0.5268  value=0.0139  wm_gn=0.2875
        z_norm=0.6169  z_max=0.9472  jump=0.0000  cons=0.0572  sol=0.9987  e_var=0.0004  ch_ent=2.5999  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5268  dret=0.5151  term=0.0137  bnd=0.0229  chart_acc=0.4954  chart_ent=1.7037  rw_drift=0.0000
        v_err=1.2760  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0007
        bnd_x=0.0256  bell=0.0347  bell_s=0.0147  rtg_e=1.2760  rtg_b=-1.2760  cal_e=1.2760  u_l2=0.1038  cov_n=0.0013
        col=7.1075  smp=0.0026  enc_t=0.2258  bnd_t=0.0344  wm_t=0.7277  crt_t=0.0059  diag_t=0.4158
        charts: 16/16 active  c0=0.10 c1=0.12 c2=0.04 c3=0.11 c4=0.01 c5=0.09 c6=0.05 c7=0.03 c8=0.06 c9=0.04 c10=0.04 c11=0.09 c12=0.02 c13=0.12 c14=0.03 c15=0.05
        symbols: 87/128 active  c0=4/8(H=1.12) c1=5/8(H=1.31) c2=6/8(H=0.87) c3=6/8(H=1.29) c4=4/8(H=1.21) c5=7/8(H=1.30) c6=7/8(H=1.47) c7=4/8(H=0.98) c8=6/8(H=1.18) c9=5/8(H=0.77) c10=5/8(H=0.98) c11=7/8(H=1.37) c12=4/8(H=1.08) c13=7/8(H=1.42) c14=5/8(H=1.07) c15=5/8(H=1.43)
E0140 [5upd]  ep_rew=11.4058  rew_20=14.4784  L_geo=0.6555  L_rew=0.0005  L_chart=1.7478  L_crit=0.2891  L_bnd=0.6553  lr=0.0010  dt=15.51s
        recon=0.2308  vq=0.2398  code_H=1.2622  code_px=3.5595  ch_usage=0.1059  rtr_mrg=0.0027  enc_gn=6.6251
        ctrl=0.0001  tex=0.0307  im_rew=0.0383  im_ret=0.5440  value=0.0089  wm_gn=0.2552
        z_norm=0.6193  z_max=0.9411  jump=0.0000  cons=0.0550  sol=0.9991  e_var=0.0006  ch_ent=2.6117  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5440  dret=0.5364  term=0.0088  bnd=0.0142  chart_acc=0.4800  chart_ent=1.7249  rw_drift=0.0000
        v_err=1.4381  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1897
        bnd_x=0.0173  bell=0.0404  bell_s=0.0340  rtg_e=1.4381  rtg_b=-1.4381  cal_e=1.4381  u_l2=0.0925  cov_n=0.0009
        col=7.3766  smp=0.0046  enc_t=0.2584  bnd_t=0.0413  wm_t=1.1726  crt_t=0.0095  diag_t=0.6553
        charts: 16/16 active  c0=0.08 c1=0.08 c2=0.02 c3=0.07 c4=0.03 c5=0.07 c6=0.05 c7=0.02 c8=0.02 c9=0.12 c10=0.09 c11=0.14 c12=0.02 c13=0.08 c14=0.06 c15=0.04
        symbols: 84/128 active  c0=4/8(H=0.89) c1=6/8(H=1.28) c2=6/8(H=1.51) c3=4/8(H=0.94) c4=5/8(H=1.40) c5=7/8(H=1.32) c6=6/8(H=1.54) c7=5/8(H=1.51) c8=5/8(H=1.40) c9=4/8(H=0.93) c10=5/8(H=1.23) c11=6/8(H=0.95) c12=4/8(H=1.37) c13=5/8(H=1.16) c14=7/8(H=1.29) c15=5/8(H=1.42)
E0141 [5upd]  ep_rew=14.3595  rew_20=14.4372  L_geo=0.7974  L_rew=0.0007  L_chart=1.8757  L_crit=0.3321  L_bnd=0.5763  lr=0.0010  dt=19.25s
        recon=0.2397  vq=0.2229  code_H=1.3716  code_px=3.9503  ch_usage=0.0817  rtr_mrg=0.0019  enc_gn=6.8328
        ctrl=0.0001  tex=0.0314  im_rew=0.0401  im_ret=0.5724  value=0.0127  wm_gn=0.4067
        z_norm=0.6131  z_max=0.9557  jump=0.0000  cons=0.0597  sol=0.9966  e_var=0.0005  ch_ent=2.5534  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5724  dret=0.5612  term=0.0131  bnd=0.0200  chart_acc=0.3975  chart_ent=1.7892  rw_drift=0.0000
        v_err=1.5726  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4070
        bnd_x=0.0230  bell=0.0432  bell_s=0.0372  rtg_e=1.5726  rtg_b=-1.5726  cal_e=1.5726  u_l2=0.0951  cov_n=0.0014
        col=11.8658  smp=0.0045  enc_t=0.2545  bnd_t=0.0416  wm_t=1.0409  crt_t=0.0088  diag_t=0.5804
        charts: 16/16 active  c0=0.17 c1=0.04 c2=0.05 c3=0.17 c4=0.03 c5=0.03 c6=0.08 c7=0.04 c8=0.01 c9=0.07 c10=0.05 c11=0.08 c12=0.03 c13=0.04 c14=0.06 c15=0.06
        symbols: 93/128 active  c0=6/8(H=1.06) c1=4/8(H=1.11) c2=5/8(H=0.64) c3=7/8(H=1.00) c4=5/8(H=1.32) c5=7/8(H=1.30) c6=7/8(H=1.57) c7=8/8(H=1.40) c8=4/8(H=1.12) c9=4/8(H=0.89) c10=4/8(H=1.02) c11=6/8(H=1.25) c12=7/8(H=1.50) c13=5/8(H=1.24) c14=6/8(H=1.35) c15=8/8(H=1.68)
E0142 [5upd]  ep_rew=21.4911  rew_20=14.7868  L_geo=0.6912  L_rew=0.0006  L_chart=1.7737  L_crit=0.2922  L_bnd=0.5452  lr=0.0010  dt=18.69s
        recon=0.2058  vq=0.2619  code_H=1.3431  code_px=3.8373  ch_usage=0.0760  rtr_mrg=0.0034  enc_gn=5.7554
        ctrl=0.0001  tex=0.0283  im_rew=0.0413  im_ret=0.5912  value=0.0156  wm_gn=0.2160
        z_norm=0.6787  z_max=0.9594  jump=0.0000  cons=0.0677  sol=0.9954  e_var=0.0031  ch_ent=2.5336  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5912  dret=0.5782  term=0.0152  bnd=0.0226  chart_acc=0.4267  chart_ent=1.8590  rw_drift=0.0000
        v_err=1.3802  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5582
        bnd_x=0.0268  bell=0.0400  bell_s=0.0395  rtg_e=1.3802  rtg_b=-1.3802  cal_e=1.3802  u_l2=0.1143  cov_n=0.0015
        col=11.3326  smp=0.0035  enc_t=0.2551  bnd_t=0.0400  wm_t=1.0375  crt_t=0.0090  diag_t=0.5779
        charts: 16/16 active  c0=0.17 c1=0.09 c2=0.07 c3=0.13 c4=0.01 c5=0.02 c6=0.04 c7=0.03 c8=0.02 c9=0.10 c10=0.02 c11=0.10 c12=0.04 c13=0.02 c14=0.08 c15=0.06
        symbols: 101/128 active  c0=7/8(H=1.46) c1=7/8(H=0.91) c2=6/8(H=1.15) c3=5/8(H=0.89) c4=6/8(H=1.54) c5=7/8(H=0.89) c6=5/8(H=1.22) c7=7/8(H=1.32) c8=5/8(H=1.35) c9=7/8(H=1.34) c10=5/8(H=1.44) c11=7/8(H=1.41) c12=7/8(H=1.46) c13=6/8(H=1.34) c14=6/8(H=1.46) c15=8/8(H=1.53)
E0143 [5upd]  ep_rew=14.1238  rew_20=14.7078  L_geo=0.6289  L_rew=0.0004  L_chart=1.6381  L_crit=0.2973  L_bnd=0.5575  lr=0.0010  dt=18.61s
        recon=0.1869  vq=0.3121  code_H=1.2469  code_px=3.4937  ch_usage=0.1084  rtr_mrg=0.0032  enc_gn=6.5667
        ctrl=0.0001  tex=0.0290  im_rew=0.0369  im_ret=0.5290  value=0.0140  wm_gn=0.3078
        z_norm=0.6641  z_max=0.9598  jump=0.0000  cons=0.0640  sol=0.9983  e_var=0.0007  ch_ent=2.6083  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5290  dret=0.5168  term=0.0142  bnd=0.0236  chart_acc=0.5454  chart_ent=1.7182  rw_drift=0.0000
        v_err=1.2993  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6510
        bnd_x=0.0283  bell=0.0351  bell_s=0.0191  rtg_e=1.2993  rtg_b=-1.2993  cal_e=1.2993  u_l2=0.1355  cov_n=0.0011
        col=11.2739  smp=0.0041  enc_t=0.2549  bnd_t=0.0405  wm_t=1.0345  crt_t=0.0087  diag_t=0.5702
        charts: 16/16 active  c0=0.08 c1=0.04 c2=0.09 c3=0.14 c4=0.01 c5=0.08 c6=0.04 c7=0.01 c8=0.04 c9=0.06 c10=0.05 c11=0.12 c12=0.03 c13=0.06 c14=0.09 c15=0.06
        symbols: 91/128 active  c0=4/8(H=0.87) c1=6/8(H=1.06) c2=6/8(H=1.05) c3=7/8(H=1.38) c4=3/8(H=0.83) c5=7/8(H=1.22) c6=5/8(H=1.12) c7=4/8(H=0.88) c8=7/8(H=1.59) c9=5/8(H=1.23) c10=7/8(H=1.80) c11=8/8(H=1.56) c12=5/8(H=1.13) c13=5/8(H=1.26) c14=7/8(H=1.42) c15=5/8(H=1.54)
E0144 [5upd]  ep_rew=13.6413  rew_20=14.6076  L_geo=0.6456  L_rew=0.0007  L_chart=1.6604  L_crit=0.2903  L_bnd=0.5365  lr=0.0010  dt=18.54s
        recon=0.2023  vq=0.3426  code_H=1.3825  code_px=3.9961  ch_usage=0.0888  rtr_mrg=0.0024  enc_gn=7.3811
        ctrl=0.0001  tex=0.0272  im_rew=0.0390  im_ret=0.5564  value=0.0133  wm_gn=0.1531
        z_norm=0.6837  z_max=0.9289  jump=0.0000  cons=0.0518  sol=0.9988  e_var=0.0010  ch_ent=2.4754  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5564  dret=0.5450  term=0.0132  bnd=0.0208  chart_acc=0.5096  chart_ent=1.6946  rw_drift=0.0000
        v_err=1.3153  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6563
        bnd_x=0.0255  bell=0.0362  bell_s=0.0258  rtg_e=1.3153  rtg_b=-1.3153  cal_e=1.3153  u_l2=0.1391  cov_n=0.0008
        col=11.2315  smp=0.0046  enc_t=0.2555  bnd_t=0.0397  wm_t=1.0292  crt_t=0.0086  diag_t=0.5706
        charts: 16/16 active  c0=0.14 c1=0.07 c2=0.10 c3=0.22 c4=0.02 c5=0.07 c6=0.07 c7=0.02 c8=0.02 c9=0.09 c10=0.02 c11=0.05 c12=0.02 c13=0.04 c14=0.03 c15=0.03
        symbols: 91/128 active  c0=5/8(H=0.60) c1=5/8(H=1.20) c2=7/8(H=0.94) c3=7/8(H=1.50) c4=4/8(H=0.91) c5=7/8(H=1.37) c6=7/8(H=1.20) c7=4/8(H=1.02) c8=7/8(H=1.32) c9=6/8(H=1.41) c10=4/8(H=1.27) c11=5/8(H=0.93) c12=5/8(H=1.36) c13=7/8(H=1.39) c14=6/8(H=0.80) c15=5/8(H=1.20)
E0145 [5upd]  ep_rew=14.0266  rew_20=14.6795  L_geo=0.6882  L_rew=0.0006  L_chart=1.6217  L_crit=0.2611  L_bnd=0.6312  lr=0.0010  dt=18.64s
        recon=0.1811  vq=0.3251  code_H=1.3695  code_px=3.9416  ch_usage=0.0646  rtr_mrg=0.0029  enc_gn=6.0072
        ctrl=0.0001  tex=0.0278  im_rew=0.0398  im_ret=0.5718  value=0.0166  wm_gn=0.4380
        z_norm=0.6850  z_max=0.9214  jump=0.0000  cons=0.0529  sol=0.9966  e_var=0.0011  ch_ent=2.5007  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5718  dret=0.5573  term=0.0169  bnd=0.0260  chart_acc=0.5346  chart_ent=1.5837  rw_drift=0.0000
        v_err=1.3473  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3997
        bnd_x=0.0284  bell=0.0380  bell_s=0.0295  rtg_e=1.3473  rtg_b=-1.3473  cal_e=1.3473  u_l2=0.1292  cov_n=0.0009
        col=11.1905  smp=0.0033  enc_t=0.2530  bnd_t=0.0402  wm_t=1.0576  crt_t=0.0087  diag_t=0.5763
        charts: 16/16 active  c0=0.13 c1=0.01 c2=0.07 c3=0.17 c4=0.01 c5=0.06 c6=0.02 c7=0.06 c8=0.03 c9=0.13 c10=0.02 c11=0.11 c12=0.03 c13=0.07 c14=0.04 c15=0.04
        symbols: 98/128 active  c0=7/8(H=1.36) c1=4/8(H=0.81) c2=7/8(H=0.79) c3=7/8(H=1.00) c4=3/8(H=0.94) c5=5/8(H=1.04) c6=5/8(H=1.22) c7=7/8(H=1.35) c8=7/8(H=1.58) c9=7/8(H=0.96) c10=5/8(H=1.23) c11=8/8(H=1.22) c12=7/8(H=1.36) c13=6/8(H=1.53) c14=8/8(H=1.27) c15=5/8(H=1.28)
E0146 [5upd]  ep_rew=14.1676  rew_20=14.9761  L_geo=0.6624  L_rew=0.0004  L_chart=1.6239  L_crit=0.3105  L_bnd=0.5983  lr=0.0010  dt=18.98s
        recon=0.1879  vq=0.2947  code_H=1.2357  code_px=3.4596  ch_usage=0.0986  rtr_mrg=0.0011  enc_gn=7.2864
        ctrl=0.0001  tex=0.0294  im_rew=0.0355  im_ret=0.5145  value=0.0213  wm_gn=0.2262
        z_norm=0.6339  z_max=0.9813  jump=0.0000  cons=0.0570  sol=0.9986  e_var=0.0071  ch_ent=2.6205  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5145  dret=0.4961  term=0.0214  bnd=0.0371  chart_acc=0.5208  chart_ent=1.6059  rw_drift=0.0000
        v_err=1.4242  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4910
        bnd_x=0.0464  bell=0.0384  bell_s=0.0304  rtg_e=1.4242  rtg_b=-1.4242  cal_e=1.4242  u_l2=0.1178  cov_n=0.0008
        col=11.6750  smp=0.0044  enc_t=0.2548  bnd_t=0.0403  wm_t=1.0282  crt_t=0.0087  diag_t=0.5770
        charts: 16/16 active  c0=0.09 c1=0.07 c2=0.03 c3=0.18 c4=0.03 c5=0.06 c6=0.02 c7=0.08 c8=0.03 c9=0.04 c10=0.05 c11=0.07 c12=0.04 c13=0.08 c14=0.04 c15=0.08
        symbols: 114/128 active  c0=5/8(H=0.83) c1=8/8(H=1.08) c2=6/8(H=0.87) c3=7/8(H=1.25) c4=6/8(H=1.31) c5=7/8(H=0.96) c6=7/8(H=1.67) c7=7/8(H=1.49) c8=7/8(H=1.58) c9=8/8(H=1.42) c10=6/8(H=1.61) c11=8/8(H=1.38) c12=8/8(H=1.80) c13=8/8(H=1.36) c14=8/8(H=0.91) c15=8/8(H=1.44)
E0147 [5upd]  ep_rew=13.5114  rew_20=14.0502  L_geo=0.6739  L_rew=0.0004  L_chart=1.7626  L_crit=0.2633  L_bnd=0.5590  lr=0.0010  dt=18.34s
        recon=0.2065  vq=0.2355  code_H=1.3148  code_px=3.7323  ch_usage=0.1797  rtr_mrg=0.0031  enc_gn=6.3470
        ctrl=0.0001  tex=0.0314  im_rew=0.0365  im_ret=0.5345  value=0.0272  wm_gn=0.1809
        z_norm=0.6157  z_max=0.9172  jump=0.0000  cons=0.0696  sol=0.9961  e_var=0.0006  ch_ent=2.5686  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5345  dret=0.5110  term=0.0273  bnd=0.0459  chart_acc=0.4754  chart_ent=1.7839  rw_drift=0.0000
        v_err=1.3168  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6038
        bnd_x=0.0614  bell=0.0360  bell_s=0.0143  rtg_e=1.3168  rtg_b=-1.3167  cal_e=1.3167  u_l2=0.0950  cov_n=0.0010
        col=11.1042  smp=0.0037  enc_t=0.2536  bnd_t=0.0399  wm_t=1.0152  crt_t=0.0086  diag_t=0.5783
        charts: 16/16 active  c0=0.07 c1=0.04 c2=0.02 c3=0.17 c4=0.01 c5=0.06 c6=0.03 c7=0.05 c8=0.02 c9=0.13 c10=0.05 c11=0.12 c12=0.04 c13=0.07 c14=0.05 c15=0.07
        symbols: 86/128 active  c0=5/8(H=1.17) c1=6/8(H=1.06) c2=5/8(H=1.05) c3=7/8(H=1.54) c4=4/8(H=1.16) c5=5/8(H=0.44) c6=5/8(H=1.46) c7=4/8(H=0.96) c8=4/8(H=1.20) c9=7/8(H=1.41) c10=6/8(H=1.31) c11=8/8(H=1.07) c12=5/8(H=1.03) c13=4/8(H=1.19) c14=5/8(H=1.31) c15=6/8(H=1.42)
E0148 [5upd]  ep_rew=14.0610  rew_20=13.9003  L_geo=0.7341  L_rew=0.0005  L_chart=1.8518  L_crit=0.2403  L_bnd=0.5553  lr=0.0010  dt=18.34s
        recon=0.2247  vq=0.1984  code_H=1.3009  code_px=3.6799  ch_usage=0.1455  rtr_mrg=0.0032  enc_gn=5.3637
        ctrl=0.0001  tex=0.0287  im_rew=0.0361  im_ret=0.5305  value=0.0297  wm_gn=0.3259
        z_norm=0.6540  z_max=0.9468  jump=0.0000  cons=0.0653  sol=1.0036  e_var=0.0008  ch_ent=2.4907  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5305  dret=0.5049  term=0.0297  bnd=0.0506  chart_acc=0.4296  chart_ent=1.8126  rw_drift=0.0000
        v_err=1.1632  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5702
        bnd_x=0.0575  bell=0.0334  bell_s=0.0250  rtg_e=1.1632  rtg_b=-1.1631  cal_e=1.1631  u_l2=0.0785  cov_n=0.0018
        col=11.0718  smp=0.0029  enc_t=0.2550  bnd_t=0.0402  wm_t=1.0229  crt_t=0.0086  diag_t=0.5683
        charts: 16/16 active  c0=0.14 c1=0.02 c2=0.06 c3=0.06 c4=0.01 c5=0.08 c6=0.03 c7=0.05 c8=0.02 c9=0.23 c10=0.10 c11=0.05 c12=0.03 c13=0.03 c14=0.05 c15=0.05
        symbols: 94/128 active  c0=6/8(H=1.45) c1=7/8(H=1.35) c2=6/8(H=0.80) c3=6/8(H=1.21) c4=5/8(H=1.41) c5=6/8(H=0.46) c6=6/8(H=1.17) c7=8/8(H=1.60) c8=5/8(H=1.20) c9=7/8(H=1.09) c10=6/8(H=1.49) c11=4/8(H=1.05) c12=7/8(H=1.50) c13=4/8(H=1.35) c14=5/8(H=1.35) c15=6/8(H=1.66)
E0149 [5upd]  ep_rew=20.7861  rew_20=14.0703  L_geo=0.7707  L_rew=0.0004  L_chart=2.0354  L_crit=0.2928  L_bnd=0.5782  lr=0.0010  dt=18.39s
        recon=0.2865  vq=0.1952  code_H=1.3617  code_px=3.9084  ch_usage=0.0650  rtr_mrg=0.0039  enc_gn=6.7517
        ctrl=0.0001  tex=0.0311  im_rew=0.0388  im_ret=0.5589  value=0.0191  wm_gn=0.2668
        z_norm=0.5928  z_max=0.9483  jump=0.0000  cons=0.0761  sol=1.0017  e_var=0.0008  ch_ent=2.5549  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5589  dret=0.5428  term=0.0187  bnd=0.0297  chart_acc=0.3567  chart_ent=1.9655  rw_drift=0.0000
        v_err=1.2765  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4886
        bnd_x=0.0355  bell=0.0352  bell_s=0.0177  rtg_e=1.2765  rtg_b=-1.2765  cal_e=1.2765  u_l2=0.0614  cov_n=0.0007
        col=11.0944  smp=0.0046  enc_t=0.2546  bnd_t=0.0400  wm_t=1.0267  crt_t=0.0088  diag_t=0.5802
        charts: 16/16 active  c0=0.11 c1=0.09 c2=0.11 c3=0.12 c4=0.02 c5=0.03 c6=0.05 c7=0.08 c8=0.02 c9=0.16 c10=0.04 c11=0.03 c12=0.03 c13=0.05 c14=0.02 c15=0.03
        symbols: 95/128 active  c0=7/8(H=1.59) c1=7/8(H=1.35) c2=5/8(H=0.86) c3=8/8(H=1.73) c4=4/8(H=1.32) c5=7/8(H=1.54) c6=6/8(H=1.47) c7=7/8(H=1.49) c8=5/8(H=1.21) c9=7/8(H=1.56) c10=5/8(H=1.36) c11=8/8(H=1.79) c12=4/8(H=0.85) c13=6/8(H=1.33) c14=4/8(H=1.00) c15=5/8(H=1.33)
E0150 [5upd]  ep_rew=14.2764  rew_20=13.9827  L_geo=0.6254  L_rew=0.0010  L_chart=1.8250  L_crit=0.3024  L_bnd=0.7849  lr=0.0010  dt=18.51s
        recon=0.2230  vq=0.2274  code_H=1.2603  code_px=3.5498  ch_usage=0.0620  rtr_mrg=0.0032  enc_gn=7.5524
        ctrl=0.0001  tex=0.0301  im_rew=0.0429  im_ret=0.6152  value=0.0171  wm_gn=0.1579
        z_norm=0.6115  z_max=0.9531  jump=0.0000  cons=0.0617  sol=0.9983  e_var=0.0007  ch_ent=2.5750  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6152  dret=0.6009  term=0.0167  bnd=0.0238  chart_acc=0.4004  chart_ent=1.8928  rw_drift=0.0000
        v_err=1.2936  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3594
        bnd_x=0.0274  bell=0.0363  bell_s=0.0189  rtg_e=1.2936  rtg_b=-1.2936  cal_e=1.2936  u_l2=0.0535  cov_n=0.0009
        col=11.1260  smp=0.0037  enc_t=0.2542  bnd_t=0.0404  wm_t=1.0444  crt_t=0.0085  diag_t=0.5777
        charts: 16/16 active  c0=0.11 c1=0.18 c2=0.04 c3=0.06 c4=0.02 c5=0.03 c6=0.05 c7=0.08 c8=0.03 c9=0.13 c10=0.06 c11=0.06 c12=0.03 c13=0.03 c14=0.05 c15=0.05
        symbols: 90/128 active  c0=6/8(H=1.16) c1=7/8(H=1.08) c2=6/8(H=0.91) c3=6/8(H=1.17) c4=6/8(H=1.52) c5=6/8(H=1.06) c6=6/8(H=1.07) c7=7/8(H=1.35) c8=4/8(H=0.86) c9=7/8(H=1.12) c10=5/8(H=1.48) c11=6/8(H=1.01) c12=4/8(H=0.74) c13=4/8(H=1.26) c14=5/8(H=1.08) c15=5/8(H=1.46)
  EVAL  reward=11.6 +/- 3.4  len=300
E0151 [5upd]  ep_rew=16.4301  rew_20=14.1575  L_geo=0.6439  L_rew=0.0005  L_chart=1.7064  L_crit=0.2742  L_bnd=0.8176  lr=0.0010  dt=17.97s
        recon=0.2088  vq=0.2487  code_H=1.2998  code_px=3.7009  ch_usage=0.0543  rtr_mrg=0.0025  enc_gn=6.1149
        ctrl=0.0001  tex=0.0292  im_rew=0.0374  im_ret=0.5412  value=0.0221  wm_gn=0.1593
        z_norm=0.6534  z_max=0.9327  jump=0.0000  cons=0.0508  sol=0.9999  e_var=0.0008  ch_ent=2.4386  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5412  dret=0.5228  term=0.0214  bnd=0.0351  chart_acc=0.4563  chart_ent=1.7069  rw_drift=0.0000
        v_err=1.3355  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.5658
        bnd_x=0.0423  bell=0.0365  bell_s=0.0157  rtg_e=1.3355  rtg_b=-1.3354  cal_e=1.3354  u_l2=0.0433  cov_n=0.0012
        col=10.7690  smp=0.0031  enc_t=0.2533  bnd_t=0.0400  wm_t=1.0115  crt_t=0.0086  diag_t=0.5682
        charts: 16/16 active  c0=0.15 c1=0.04 c2=0.02 c3=0.17 c4=0.03 c5=0.09 c6=0.04 c7=0.04 c8=0.01 c9=0.19 c10=0.06 c11=0.01 c12=0.03 c13=0.03 c14=0.04 c15=0.05
        symbols: 83/128 active  c0=5/8(H=1.45) c1=4/8(H=0.43) c2=4/8(H=1.33) c3=6/8(H=0.95) c4=6/8(H=1.45) c5=6/8(H=0.68) c6=5/8(H=1.05) c7=6/8(H=1.34) c8=2/8(H=0.38) c9=7/8(H=1.32) c10=5/8(H=1.35) c11=5/8(H=1.36) c12=6/8(H=1.50) c13=5/8(H=1.46) c14=5/8(H=0.87) c15=6/8(H=1.55)
E0152 [5upd]  ep_rew=13.6013  rew_20=14.4761  L_geo=0.6475  L_rew=0.0005  L_chart=1.6930  L_crit=0.2542  L_bnd=0.7140  lr=0.0010  dt=18.22s
        recon=0.1978  vq=0.2506  code_H=1.3150  code_px=3.7394  ch_usage=0.1134  rtr_mrg=0.0053  enc_gn=5.2457
        ctrl=0.0001  tex=0.0286  im_rew=0.0355  im_ret=0.5205  value=0.0269  wm_gn=0.1678
        z_norm=0.6340  z_max=0.9695  jump=0.0000  cons=0.0739  sol=0.9957  e_var=0.0006  ch_ent=2.6016  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5205  dret=0.4965  term=0.0278  bnd=0.0482  chart_acc=0.4629  chart_ent=1.7185  rw_drift=0.0000
        v_err=1.4231  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2237
        bnd_x=0.0537  bell=0.0410  bell_s=0.0444  rtg_e=1.4231  rtg_b=-1.4230  cal_e=1.4230  u_l2=0.0311  cov_n=0.0014
        col=11.0036  smp=0.0043  enc_t=0.2530  bnd_t=0.0397  wm_t=1.0140  crt_t=0.0088  diag_t=0.5678
        charts: 16/16 active  c0=0.14 c1=0.06 c2=0.05 c3=0.12 c4=0.01 c5=0.06 c6=0.04 c7=0.02 c8=0.02 c9=0.10 c10=0.03 c11=0.09 c12=0.03 c13=0.06 c14=0.09 c15=0.08
        symbols: 108/128 active  c0=5/8(H=0.89) c1=5/8(H=0.87) c2=6/8(H=1.11) c3=7/8(H=1.46) c4=7/8(H=1.80) c5=8/8(H=0.83) c6=5/8(H=1.39) c7=8/8(H=1.72) c8=7/8(H=1.71) c9=6/8(H=0.93) c10=6/8(H=1.38) c11=8/8(H=1.62) c12=8/8(H=1.70) c13=7/8(H=1.34) c14=7/8(H=1.11) c15=8/8(H=1.71)
E0153 [5upd]  ep_rew=14.1792  rew_20=14.5060  L_geo=0.6517  L_rew=0.0004  L_chart=1.7673  L_crit=0.2819  L_bnd=0.5499  lr=0.0010  dt=16.60s
        recon=0.2281  vq=0.2682  code_H=1.3482  code_px=3.8535  ch_usage=0.0946  rtr_mrg=0.0023  enc_gn=6.8503
        ctrl=0.0001  tex=0.0272  im_rew=0.0363  im_ret=0.5310  value=0.0276  wm_gn=0.1895
        z_norm=0.6619  z_max=0.9512  jump=0.0000  cons=0.0650  sol=0.9941  e_var=0.0017  ch_ent=2.5869  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5310  dret=0.5081  term=0.0266  bnd=0.0450  chart_acc=0.4600  chart_ent=1.7496  rw_drift=0.0000
        v_err=1.4081  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4935
        bnd_x=0.0522  bell=0.0385  bell_s=0.0274  rtg_e=1.4081  rtg_b=-1.4080  cal_e=1.4080  u_l2=0.0295  cov_n=0.0018
        col=11.0221  smp=0.0032  enc_t=0.2332  bnd_t=0.0352  wm_t=0.7504  crt_t=0.0065  diag_t=0.3919
        charts: 16/16 active  c0=0.15 c1=0.06 c2=0.07 c3=0.06 c4=0.02 c5=0.13 c6=0.02 c7=0.05 c8=0.03 c9=0.13 c10=0.04 c11=0.06 c12=0.04 c13=0.04 c14=0.07 c15=0.05
        symbols: 90/128 active  c0=6/8(H=1.46) c1=4/8(H=0.27) c2=8/8(H=1.65) c3=6/8(H=1.00) c4=2/8(H=0.16) c5=8/8(H=1.14) c6=5/8(H=1.22) c7=5/8(H=1.14) c8=5/8(H=1.01) c9=5/8(H=0.96) c10=5/8(H=1.24) c11=5/8(H=1.11) c12=5/8(H=1.46) c13=8/8(H=1.75) c14=7/8(H=1.28) c15=6/8(H=1.59)
E0154 [5upd]  ep_rew=15.1076  rew_20=14.1011  L_geo=0.6695  L_rew=0.0005  L_chart=1.6057  L_crit=0.2758  L_bnd=0.5361  lr=0.0010  dt=12.42s
        recon=0.1962  vq=0.2888  code_H=1.2835  code_px=3.6299  ch_usage=0.1103  rtr_mrg=0.0021  enc_gn=9.1397
        ctrl=0.0001  tex=0.0289  im_rew=0.0447  im_ret=0.6429  value=0.0212  wm_gn=0.2293
        z_norm=0.6286  z_max=0.9576  jump=0.0000  cons=0.0568  sol=0.9981  e_var=0.0011  ch_ent=2.6076  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6429  dret=0.6248  term=0.0210  bnd=0.0289  chart_acc=0.5254  chart_ent=1.6987  rw_drift=0.0000
        v_err=1.4638  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6629
        bnd_x=0.0376  bell=0.0402  bell_s=0.0207  rtg_e=1.4638  rtg_b=-1.4638  cal_e=1.4638  u_l2=0.0516  cov_n=0.0017
        col=7.0475  smp=0.0031  enc_t=0.2298  bnd_t=0.0351  wm_t=0.7117  crt_t=0.0058  diag_t=0.4038
        charts: 16/16 active  c0=0.10 c1=0.12 c2=0.02 c3=0.14 c4=0.03 c5=0.04 c6=0.11 c7=0.03 c8=0.02 c9=0.09 c10=0.05 c11=0.05 c12=0.02 c13=0.06 c14=0.07 c15=0.04
        symbols: 93/128 active  c0=5/8(H=0.84) c1=5/8(H=1.27) c2=5/8(H=1.00) c3=8/8(H=1.05) c4=6/8(H=1.23) c5=6/8(H=1.00) c6=7/8(H=1.42) c7=6/8(H=1.41) c8=5/8(H=0.83) c9=7/8(H=1.47) c10=5/8(H=1.40) c11=4/8(H=0.91) c12=5/8(H=1.41) c13=7/8(H=1.47) c14=6/8(H=1.33) c15=6/8(H=1.27)
E0155 [5upd]  ep_rew=15.1693  rew_20=15.6391  L_geo=0.7525  L_rew=0.0004  L_chart=1.8427  L_crit=0.2978  L_bnd=0.5397  lr=0.0010  dt=12.77s
        recon=0.2165  vq=0.2658  code_H=1.2975  code_px=3.6710  ch_usage=0.0869  rtr_mrg=0.0014  enc_gn=9.7334
        ctrl=0.0001  tex=0.0296  im_rew=0.0450  im_ret=0.6437  value=0.0163  wm_gn=0.6014
        z_norm=0.6257  z_max=0.9385  jump=0.0000  cons=0.0717  sol=0.9974  e_var=0.0006  ch_ent=2.5875  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6437  dret=0.6295  term=0.0165  bnd=0.0225  chart_acc=0.3821  chart_ent=1.8614  rw_drift=0.0000
        v_err=1.4649  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6569
        bnd_x=0.0261  bell=0.0411  bell_s=0.0306  rtg_e=1.4649  rtg_b=-1.4649  cal_e=1.4649  u_l2=0.0616  cov_n=0.0015
        col=7.2907  smp=0.0033  enc_t=0.2481  bnd_t=0.0345  wm_t=0.6989  crt_t=0.0061  diag_t=0.4830
        charts: 16/16 active  c0=0.10 c1=0.08 c2=0.04 c3=0.12 c4=0.04 c5=0.04 c6=0.08 c7=0.08 c8=0.03 c9=0.17 c10=0.03 c11=0.06 c12=0.04 c13=0.02 c14=0.05 c15=0.03
        symbols: 93/128 active  c0=5/8(H=1.12) c1=6/8(H=1.37) c2=5/8(H=0.92) c3=7/8(H=1.71) c4=6/8(H=1.01) c5=5/8(H=0.53) c6=8/8(H=1.76) c7=6/8(H=1.23) c8=4/8(H=1.20) c9=8/8(H=1.39) c10=4/8(H=1.21) c11=6/8(H=1.46) c12=7/8(H=1.49) c13=5/8(H=1.23) c14=6/8(H=1.46) c15=5/8(H=1.08)
E0156 [5upd]  ep_rew=28.7484  rew_20=15.8947  L_geo=0.7217  L_rew=0.0002  L_chart=1.8887  L_crit=0.2626  L_bnd=0.5033  lr=0.0010  dt=19.92s
        recon=0.2534  vq=0.2354  code_H=1.3962  code_px=4.0898  ch_usage=0.1387  rtr_mrg=0.0036  enc_gn=7.0396
        ctrl=0.0001  tex=0.0332  im_rew=0.0357  im_ret=0.5088  value=0.0111  wm_gn=0.1978
        z_norm=0.5946  z_max=0.9199  jump=0.0000  cons=0.0889  sol=0.9970  e_var=0.0003  ch_ent=2.6740  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5088  dret=0.4999  term=0.0104  bnd=0.0178  chart_acc=0.3508  chart_ent=1.8013  rw_drift=0.0000
        v_err=1.3283  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7005
        bnd_x=0.0210  bell=0.0361  bell_s=0.0185  rtg_e=1.3283  rtg_b=-1.3283  cal_e=1.3283  u_l2=0.0630  cov_n=0.0007
        col=12.4209  smp=0.0042  enc_t=0.2584  bnd_t=0.0406  wm_t=1.0606  crt_t=0.0089  diag_t=0.5831
        charts: 16/16 active  c0=0.09 c1=0.07 c2=0.09 c3=0.08 c4=0.02 c5=0.02 c6=0.07 c7=0.05 c8=0.03 c9=0.11 c10=0.06 c11=0.09 c12=0.04 c13=0.07 c14=0.05 c15=0.07
        symbols: 92/128 active  c0=4/8(H=1.20) c1=5/8(H=1.25) c2=6/8(H=1.08) c3=5/8(H=0.70) c4=4/8(H=1.08) c5=6/8(H=1.71) c6=7/8(H=1.74) c7=7/8(H=1.53) c8=4/8(H=1.27) c9=8/8(H=1.37) c10=5/8(H=1.37) c11=8/8(H=1.42) c12=5/8(H=1.37) c13=7/8(H=1.16) c14=5/8(H=1.36) c15=6/8(H=1.57)
E0157 [5upd]  ep_rew=14.1330  rew_20=16.4779  L_geo=0.7750  L_rew=0.0008  L_chart=1.9892  L_crit=0.3291  L_bnd=0.5499  lr=0.0010  dt=18.63s
        recon=0.3033  vq=0.2044  code_H=1.4098  code_px=4.1173  ch_usage=0.0631  rtr_mrg=0.0016  enc_gn=6.0929
        ctrl=0.0001  tex=0.0286  im_rew=0.0367  im_ret=0.5226  value=0.0105  wm_gn=0.1648
        z_norm=0.6425  z_max=0.9469  jump=0.0000  cons=0.0498  sol=0.9983  e_var=0.0006  ch_ent=2.6263  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5226  dret=0.5136  term=0.0104  bnd=0.0174  chart_acc=0.3417  chart_ent=1.9688  rw_drift=0.0000
        v_err=1.1572  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5546
        bnd_x=0.0233  bell=0.0331  bell_s=0.0398  rtg_e=1.1572  rtg_b=-1.1572  cal_e=1.1572  u_l2=0.0707  cov_n=0.0009
        col=11.3403  smp=0.0041  enc_t=0.2540  bnd_t=0.0398  wm_t=1.0240  crt_t=0.0086  diag_t=0.5887
        charts: 16/16 active  c0=0.11 c1=0.03 c2=0.08 c3=0.05 c4=0.01 c5=0.08 c6=0.04 c7=0.07 c8=0.02 c9=0.10 c10=0.14 c11=0.07 c12=0.03 c13=0.07 c14=0.04 c15=0.06
        symbols: 87/128 active  c0=6/8(H=1.59) c1=3/8(H=0.72) c2=6/8(H=1.22) c3=4/8(H=1.08) c4=5/8(H=1.20) c5=7/8(H=1.42) c6=5/8(H=1.24) c7=5/8(H=1.33) c8=6/8(H=1.45) c9=5/8(H=1.28) c10=7/8(H=1.71) c11=6/8(H=1.48) c12=6/8(H=1.56) c13=6/8(H=1.58) c14=5/8(H=1.52) c15=5/8(H=1.47)
E0158 [5upd]  ep_rew=14.1086  rew_20=16.7692  L_geo=0.7760  L_rew=0.0006  L_chart=1.9854  L_crit=0.3035  L_bnd=0.5757  lr=0.0010  dt=18.57s
        recon=0.2864  vq=0.2247  code_H=1.4009  code_px=4.0634  ch_usage=0.0716  rtr_mrg=0.0043  enc_gn=5.8857
        ctrl=0.0001  tex=0.0311  im_rew=0.0414  im_ret=0.5899  value=0.0124  wm_gn=0.1888
        z_norm=0.5773  z_max=0.9640  jump=0.0000  cons=0.0720  sol=0.9985  e_var=0.0005  ch_ent=2.6113  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5899  dret=0.5792  term=0.0125  bnd=0.0186  chart_acc=0.3917  chart_ent=1.9296  rw_drift=0.0000
        v_err=1.6015  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5170
        bnd_x=0.0236  bell=0.0457  bell_s=0.0351  rtg_e=1.6015  rtg_b=-1.6015  cal_e=1.6015  u_l2=0.0823  cov_n=0.0014
        col=11.2571  smp=0.0028  enc_t=0.2539  bnd_t=0.0400  wm_t=1.0316  crt_t=0.0085  diag_t=0.5739
        charts: 16/16 active  c0=0.04 c1=0.04 c2=0.05 c3=0.09 c4=0.05 c5=0.09 c6=0.08 c7=0.02 c8=0.05 c9=0.08 c10=0.03 c11=0.20 c12=0.03 c13=0.04 c14=0.06 c15=0.05
        symbols: 96/128 active  c0=4/8(H=0.78) c1=6/8(H=0.76) c2=6/8(H=1.14) c3=5/8(H=0.76) c4=7/8(H=1.59) c5=6/8(H=1.31) c6=5/8(H=1.34) c7=8/8(H=1.75) c8=6/8(H=1.48) c9=7/8(H=1.52) c10=6/8(H=1.11) c11=8/8(H=1.66) c12=5/8(H=1.43) c13=6/8(H=1.07) c14=6/8(H=1.58) c15=5/8(H=1.07)
E0159 [5upd]  ep_rew=34.1128  rew_20=17.7533  L_geo=0.7406  L_rew=0.0004  L_chart=1.8363  L_crit=0.3153  L_bnd=0.5128  lr=0.0010  dt=18.43s
        recon=0.2461  vq=0.2250  code_H=1.4501  code_px=4.2735  ch_usage=0.0790  rtr_mrg=0.0033  enc_gn=6.3076
        ctrl=0.0000  tex=0.0299  im_rew=0.0367  im_ret=0.5195  value=0.0070  wm_gn=0.2771
        z_norm=0.6178  z_max=0.9026  jump=0.0000  cons=0.0604  sol=0.9957  e_var=0.0008  ch_ent=2.5290  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5195  dret=0.5135  term=0.0069  bnd=0.0116  chart_acc=0.4000  chart_ent=1.9119  rw_drift=0.0000
        v_err=1.3654  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7426
        bnd_x=0.0138  bell=0.0367  bell_s=0.0154  rtg_e=1.3654  rtg_b=-1.3654  cal_e=1.3654  u_l2=0.1037  cov_n=0.0006
        col=11.1333  smp=0.0036  enc_t=0.2540  bnd_t=0.0399  wm_t=1.0274  crt_t=0.0088  diag_t=0.5817
        charts: 16/16 active  c0=0.06 c1=0.11 c2=0.04 c3=0.10 c4=0.01 c5=0.03 c6=0.10 c7=0.06 c8=0.03 c9=0.19 c10=0.08 c11=0.09 c12=0.01 c13=0.04 c14=0.03 c15=0.02
        symbols: 95/128 active  c0=6/8(H=1.24) c1=7/8(H=1.60) c2=5/8(H=0.51) c3=4/8(H=1.19) c4=5/8(H=1.21) c5=6/8(H=1.19) c6=6/8(H=1.44) c7=6/8(H=1.30) c8=6/8(H=1.57) c9=4/8(H=0.62) c10=6/8(H=1.69) c11=7/8(H=1.46) c12=7/8(H=1.80) c13=7/8(H=1.39) c14=6/8(H=0.88) c15=7/8(H=1.22)
E0160 [5upd]  ep_rew=15.1154  rew_20=16.1232  L_geo=0.4870  L_rew=0.0004  L_chart=1.4514  L_crit=0.3020  L_bnd=0.5859  lr=0.0010  dt=18.51s
        recon=0.1960  vq=0.3031  code_H=1.2301  code_px=3.4479  ch_usage=0.1267  rtr_mrg=0.0010  enc_gn=10.5599
        ctrl=0.0000  tex=0.0265  im_rew=0.0397  im_ret=0.5641  value=0.0098  wm_gn=0.1866
        z_norm=0.6883  z_max=0.9673  jump=0.0000  cons=0.0497  sol=0.9964  e_var=0.0016  ch_ent=2.3732  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5641  dret=0.5557  term=0.0098  bnd=0.0151  chart_acc=0.5771  chart_ent=1.6905  rw_drift=0.0000
        v_err=1.3308  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6204
        bnd_x=0.0190  bell=0.0364  bell_s=0.0199  rtg_e=1.3308  rtg_b=-1.3308  cal_e=1.3308  u_l2=0.1247  cov_n=0.0006
        col=11.1351  smp=0.0038  enc_t=0.2526  bnd_t=0.0398  wm_t=1.0494  crt_t=0.0086  diag_t=0.5563
        charts: 16/16 active  c0=0.20 c1=0.04 c2=0.10 c3=0.10 c4=0.06 c5=0.07 c6=0.05 c7=0.00 c8=0.01 c9=0.20 c10=0.04 c11=0.05 c12=0.01 c13=0.03 c14=0.03 c15=0.01
        symbols: 71/128 active  c0=5/8(H=0.93) c1=5/8(H=0.93) c2=5/8(H=0.96) c3=6/8(H=1.35) c4=4/8(H=0.83) c5=7/8(H=0.92) c6=5/8(H=1.24) c7=3/8(H=0.88) c8=6/8(H=1.15) c9=4/8(H=0.83) c10=3/8(H=0.58) c11=4/8(H=0.97) c12=3/8(H=0.52) c13=5/8(H=1.17) c14=4/8(H=0.68) c15=2/8(H=0.55)
E0161 [5upd]  ep_rew=9.0535  rew_20=14.9806  L_geo=0.6878  L_rew=0.0004  L_chart=1.7755  L_crit=0.2443  L_bnd=0.6518  lr=0.0010  dt=18.31s
        recon=0.2294  vq=0.2520  code_H=1.3074  code_px=3.7017  ch_usage=0.0292  rtr_mrg=0.0029  enc_gn=5.8454
        ctrl=0.0001  tex=0.0318  im_rew=0.0358  im_ret=0.5171  value=0.0191  wm_gn=0.1677
        z_norm=0.6059  z_max=0.9685  jump=0.0000  cons=0.0633  sol=0.9993  e_var=0.0005  ch_ent=2.6328  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5171  dret=0.5009  term=0.0188  bnd=0.0322  chart_acc=0.4308  chart_ent=1.7892  rw_drift=0.0000
        v_err=1.2982  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1716
        bnd_x=0.0345  bell=0.0354  bell_s=0.0128  rtg_e=1.2982  rtg_b=-1.2982  cal_e=1.2982  u_l2=0.1162  cov_n=0.0010
        col=11.0374  smp=0.0030  enc_t=0.2532  bnd_t=0.0403  wm_t=1.0228  crt_t=0.0086  diag_t=0.5789
        charts: 16/16 active  c0=0.07 c1=0.13 c2=0.07 c3=0.06 c4=0.02 c5=0.07 c6=0.06 c7=0.03 c8=0.05 c9=0.10 c10=0.05 c11=0.14 c12=0.01 c13=0.06 c14=0.05 c15=0.05
        symbols: 82/128 active  c0=5/8(H=0.72) c1=6/8(H=1.60) c2=6/8(H=1.19) c3=4/8(H=1.33) c4=4/8(H=0.95) c5=6/8(H=0.66) c6=5/8(H=1.08) c7=4/8(H=1.16) c8=6/8(H=0.93) c9=6/8(H=1.09) c10=4/8(H=1.01) c11=5/8(H=0.97) c12=4/8(H=1.23) c13=7/8(H=1.52) c14=4/8(H=1.29) c15=6/8(H=1.35)
E0162 [5upd]  ep_rew=15.2235  rew_20=14.1504  L_geo=0.7020  L_rew=0.0005  L_chart=2.1164  L_crit=0.3097  L_bnd=0.7341  lr=0.0010  dt=18.39s
        recon=0.2677  vq=0.2287  code_H=1.3884  code_px=4.0169  ch_usage=0.0819  rtr_mrg=0.0024  enc_gn=5.5042
        ctrl=0.0002  tex=0.0304  im_rew=0.0369  im_ret=0.5370  value=0.0234  wm_gn=1.0779
        z_norm=0.6233  z_max=0.9456  jump=0.0000  cons=0.0766  sol=0.9946  e_var=0.0005  ch_ent=2.6656  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5370  dret=0.5169  term=0.0233  bnd=0.0388  chart_acc=0.4213  chart_ent=1.8484  rw_drift=0.0000
        v_err=1.3729  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0466
        bnd_x=0.0419  bell=0.0376  bell_s=0.0288  rtg_e=1.3729  rtg_b=-1.3728  cal_e=1.3728  u_l2=0.0947  cov_n=0.0017
        col=11.1492  smp=0.0042  enc_t=0.2529  bnd_t=0.0398  wm_t=1.0185  crt_t=0.0086  diag_t=0.5740
        charts: 16/16 active  c0=0.10 c1=0.04 c2=0.11 c3=0.10 c4=0.02 c5=0.06 c6=0.03 c7=0.03 c8=0.07 c9=0.06 c10=0.04 c11=0.08 c12=0.04 c13=0.08 c14=0.09 c15=0.06
        symbols: 98/128 active  c0=4/8(H=0.73) c1=4/8(H=0.67) c2=8/8(H=1.26) c3=4/8(H=1.22) c4=4/8(H=0.90) c5=8/8(H=1.54) c6=5/8(H=1.13) c7=6/8(H=1.37) c8=7/8(H=1.12) c9=5/8(H=1.13) c10=7/8(H=1.36) c11=6/8(H=1.35) c12=7/8(H=1.56) c13=8/8(H=1.58) c14=8/8(H=1.61) c15=7/8(H=1.58)
E0163 [5upd]  ep_rew=14.5032  rew_20=13.9994  L_geo=0.5440  L_rew=0.0004  L_chart=1.6818  L_crit=0.2746  L_bnd=0.7774  lr=0.0010  dt=18.34s
        recon=0.2049  vq=0.2480  code_H=1.2738  code_px=3.5838  ch_usage=0.0827  rtr_mrg=0.0023  enc_gn=7.2247
        ctrl=0.0001  tex=0.0309  im_rew=0.0414  im_ret=0.5977  value=0.0213  wm_gn=0.1962
        z_norm=0.6248  z_max=0.9672  jump=0.0000  cons=0.0570  sol=0.9971  e_var=0.0006  ch_ent=2.5938  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5977  dret=0.5796  term=0.0211  bnd=0.0313  chart_acc=0.5200  chart_ent=1.7676  rw_drift=0.0000
        v_err=1.3538  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2417
        bnd_x=0.0340  bell=0.0368  bell_s=0.0128  rtg_e=1.3538  rtg_b=-1.3538  cal_e=1.3538  u_l2=0.0918  cov_n=0.0013
        col=11.0950  smp=0.0031  enc_t=0.2530  bnd_t=0.0413  wm_t=1.0180  crt_t=0.0085  diag_t=0.5708
        charts: 16/16 active  c0=0.15 c1=0.06 c2=0.07 c3=0.11 c4=0.01 c5=0.03 c6=0.08 c7=0.03 c8=0.02 c9=0.06 c10=0.02 c11=0.06 c12=0.05 c13=0.08 c14=0.11 c15=0.08
        symbols: 81/128 active  c0=6/8(H=1.13) c1=5/8(H=0.98) c2=6/8(H=1.14) c3=7/8(H=1.13) c4=4/8(H=1.29) c5=6/8(H=0.72) c6=6/8(H=1.53) c7=6/8(H=1.30) c8=4/8(H=0.86) c9=5/8(H=1.12) c10=3/8(H=1.09) c11=5/8(H=1.21) c12=4/8(H=0.89) c13=5/8(H=1.25) c14=5/8(H=1.25) c15=4/8(H=1.17)
E0164 [5upd]  ep_rew=14.8045  rew_20=13.0532  L_geo=0.6936  L_rew=0.0013  L_chart=1.8342  L_crit=0.3689  L_bnd=0.7487  lr=0.0010  dt=18.75s
        recon=0.2301  vq=0.2430  code_H=1.3709  code_px=3.9822  ch_usage=0.0543  rtr_mrg=0.0021  enc_gn=6.9972
        ctrl=0.0001  tex=0.0305  im_rew=0.0458  im_ret=0.6570  value=0.0187  wm_gn=0.1983
        z_norm=0.6276  z_max=0.9567  jump=0.0000  cons=0.0686  sol=0.9986  e_var=0.0005  ch_ent=2.6178  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6570  dret=0.6411  term=0.0184  bnd=0.0247  chart_acc=0.4113  chart_ent=1.7828  rw_drift=0.0000
        v_err=1.3186  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2624
        bnd_x=0.0298  bell=0.0383  bell_s=0.0476  rtg_e=1.3186  rtg_b=-1.3186  cal_e=1.3186  u_l2=0.0842  cov_n=0.0010
        col=11.1451  smp=0.0034  enc_t=0.2561  bnd_t=0.0407  wm_t=1.0836  crt_t=0.0091  diag_t=0.5865
        charts: 16/16 active  c0=0.08 c1=0.05 c2=0.12 c3=0.12 c4=0.03 c5=0.10 c6=0.04 c7=0.03 c8=0.02 c9=0.13 c10=0.03 c11=0.05 c12=0.03 c13=0.04 c14=0.06 c15=0.05
        symbols: 93/128 active  c0=6/8(H=1.03) c1=6/8(H=1.20) c2=7/8(H=1.36) c3=7/8(H=1.38) c4=6/8(H=1.03) c5=7/8(H=1.08) c6=5/8(H=1.24) c7=8/8(H=1.48) c8=4/8(H=1.13) c9=8/8(H=1.47) c10=4/8(H=1.14) c11=4/8(H=1.05) c12=5/8(H=1.33) c13=6/8(H=1.55) c14=5/8(H=0.88) c15=5/8(H=1.32)
E0165 [5upd]  ep_rew=12.4249  rew_20=12.7254  L_geo=0.6795  L_rew=0.0004  L_chart=1.8364  L_crit=0.3024  L_bnd=0.7738  lr=0.0010  dt=18.43s
        recon=0.2582  vq=0.2471  code_H=1.3673  code_px=3.9419  ch_usage=0.0861  rtr_mrg=0.0017  enc_gn=5.3029
        ctrl=0.0001  tex=0.0253  im_rew=0.0314  im_ret=0.4583  value=0.0217  wm_gn=0.5469
        z_norm=0.6881  z_max=0.9535  jump=0.0000  cons=0.0846  sol=0.9946  e_var=0.0009  ch_ent=2.5799  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4583  dret=0.4396  term=0.0217  bnd=0.0425  chart_acc=0.4029  chart_ent=1.8216  rw_drift=0.0000
        v_err=1.3343  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.2424
        bnd_x=0.0507  bell=0.0362  bell_s=0.0211  rtg_e=1.3343  rtg_b=-1.3343  cal_e=1.3343  u_l2=0.0778  cov_n=0.0009
        col=11.0680  smp=0.0045  enc_t=0.2531  bnd_t=0.0400  wm_t=1.0429  crt_t=0.0086  diag_t=0.5677
        charts: 16/16 active  c0=0.01 c1=0.03 c2=0.08 c3=0.04 c4=0.04 c5=0.02 c6=0.04 c7=0.05 c8=0.02 c9=0.15 c10=0.06 c11=0.07 c12=0.09 c13=0.09 c14=0.09 c15=0.12
        symbols: 86/128 active  c0=2/8(H=0.26) c1=4/8(H=1.01) c2=6/8(H=1.28) c3=3/8(H=0.50) c4=5/8(H=1.11) c5=5/8(H=1.28) c6=6/8(H=1.12) c7=7/8(H=1.59) c8=3/8(H=0.95) c9=6/8(H=1.48) c10=4/8(H=1.23) c11=6/8(H=1.32) c12=8/8(H=1.56) c13=5/8(H=0.73) c14=8/8(H=1.48) c15=8/8(H=1.75)
E0166 [5upd]  ep_rew=15.6946  rew_20=13.7676  L_geo=0.6876  L_rew=0.0010  L_chart=1.8147  L_crit=0.3301  L_bnd=0.6826  lr=0.0010  dt=18.30s
        recon=0.2347  vq=0.2540  code_H=1.3363  code_px=3.8125  ch_usage=0.0399  rtr_mrg=0.0029  enc_gn=7.9072
        ctrl=0.0001  tex=0.0270  im_rew=0.0382  im_ret=0.5563  value=0.0264  wm_gn=0.4120
        z_norm=0.6487  z_max=0.9613  jump=0.0000  cons=0.0657  sol=0.9995  e_var=0.0009  ch_ent=2.6253  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5563  dret=0.5340  term=0.0260  bnd=0.0418  chart_acc=0.3929  chart_ent=1.8375  rw_drift=0.0000
        v_err=1.4389  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0182
        bnd_x=0.0499  bell=0.0396  bell_s=0.0266  rtg_e=1.4389  rtg_b=-1.4389  cal_e=1.4389  u_l2=0.0759  cov_n=0.0012
        col=11.0807  smp=0.0049  enc_t=0.2535  bnd_t=0.0396  wm_t=1.0136  crt_t=0.0085  diag_t=0.5714
        charts: 16/16 active  c0=0.10 c1=0.09 c2=0.05 c3=0.07 c4=0.01 c5=0.04 c6=0.06 c7=0.04 c8=0.03 c9=0.17 c10=0.05 c11=0.07 c12=0.03 c13=0.07 c14=0.06 c15=0.07
        symbols: 97/128 active  c0=4/8(H=1.07) c1=6/8(H=1.24) c2=6/8(H=1.01) c3=5/8(H=1.58) c4=5/8(H=1.15) c5=7/8(H=0.81) c6=5/8(H=1.35) c7=8/8(H=1.68) c8=6/8(H=1.43) c9=5/8(H=0.87) c10=6/8(H=1.58) c11=5/8(H=1.22) c12=7/8(H=1.72) c13=8/8(H=1.41) c14=6/8(H=1.52) c15=8/8(H=1.71)
E0167 [5upd]  ep_rew=14.2866  rew_20=14.0683  L_geo=0.7690  L_rew=0.0008  L_chart=1.8398  L_crit=0.3175  L_bnd=0.5571  lr=0.0010  dt=18.21s
        recon=0.2346  vq=0.2757  code_H=1.3216  code_px=3.7571  ch_usage=0.0359  rtr_mrg=0.0026  enc_gn=5.9898
        ctrl=0.0001  tex=0.0278  im_rew=0.0460  im_ret=0.6700  value=0.0310  wm_gn=0.1508
        z_norm=0.6574  z_max=0.9364  jump=0.0000  cons=0.0742  sol=0.9990  e_var=0.0010  ch_ent=2.6935  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6700  dret=0.6435  term=0.0308  bnd=0.0412  chart_acc=0.3908  chart_ent=1.8712  rw_drift=0.0000
        v_err=1.3225  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5202
        bnd_x=0.0486  bell=0.0361  bell_s=0.0226  rtg_e=1.3225  rtg_b=-1.3224  cal_e=1.3224  u_l2=0.0682  cov_n=0.0016
        col=10.9960  smp=0.0035  enc_t=0.2528  bnd_t=0.0398  wm_t=1.0134  crt_t=0.0086  diag_t=0.5686
        charts: 16/16 active  c0=0.09 c1=0.06 c2=0.08 c3=0.08 c4=0.02 c5=0.04 c6=0.03 c7=0.04 c8=0.04 c9=0.07 c10=0.07 c11=0.11 c12=0.05 c13=0.06 c14=0.07 c15=0.08
        symbols: 95/128 active  c0=5/8(H=1.34) c1=5/8(H=0.73) c2=8/8(H=1.45) c3=5/8(H=1.38) c4=4/8(H=0.98) c5=6/8(H=1.31) c6=5/8(H=1.28) c7=7/8(H=1.39) c8=7/8(H=1.37) c9=4/8(H=0.42) c10=6/8(H=1.27) c11=5/8(H=1.11) c12=8/8(H=1.47) c13=8/8(H=1.61) c14=6/8(H=1.39) c15=6/8(H=1.11)
E0168 [5upd]  ep_rew=14.8136  rew_20=14.0840  L_geo=0.5927  L_rew=0.0004  L_chart=1.5872  L_crit=0.2610  L_bnd=0.5119  lr=0.0010  dt=18.34s
        recon=0.1901  vq=0.2850  code_H=1.2337  code_px=3.4422  ch_usage=0.1115  rtr_mrg=0.0026  enc_gn=7.9480
        ctrl=0.0001  tex=0.0285  im_rew=0.0367  im_ret=0.5425  value=0.0349  wm_gn=0.1642
        z_norm=0.6573  z_max=0.9658  jump=0.0000  cons=0.0880  sol=0.9896  e_var=0.0006  ch_ent=2.6612  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5425  dret=0.5140  term=0.0331  bnd=0.0554  chart_acc=0.5038  chart_ent=1.6651  rw_drift=0.0000
        v_err=1.3795  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7711
        bnd_x=0.0663  bell=0.0401  bell_s=0.0339  rtg_e=1.3795  rtg_b=-1.3794  cal_e=1.3794  u_l2=0.0520  cov_n=0.0018
        col=11.1094  smp=0.0040  enc_t=0.2534  bnd_t=0.0401  wm_t=1.0175  crt_t=0.0086  diag_t=0.5606
        charts: 16/16 active  c0=0.11 c1=0.03 c2=0.09 c3=0.04 c4=0.04 c5=0.13 c6=0.07 c7=0.05 c8=0.04 c9=0.11 c10=0.05 c11=0.05 c12=0.05 c13=0.09 c14=0.03 c15=0.04
        symbols: 92/128 active  c0=7/8(H=1.30) c1=5/8(H=1.03) c2=6/8(H=1.35) c3=5/8(H=1.06) c4=7/8(H=1.75) c5=8/8(H=1.37) c6=5/8(H=0.89) c7=6/8(H=1.33) c8=6/8(H=1.15) c9=5/8(H=0.96) c10=5/8(H=1.29) c11=5/8(H=0.93) c12=5/8(H=1.12) c13=6/8(H=0.95) c14=6/8(H=1.41) c15=5/8(H=1.45)
E0169 [5upd]  ep_rew=14.5530  rew_20=13.9426  L_geo=0.6720  L_rew=0.0004  L_chart=1.8562  L_crit=0.2763  L_bnd=0.4848  lr=0.0010  dt=18.31s
        recon=0.2265  vq=0.2323  code_H=1.4052  code_px=4.0812  ch_usage=0.0791  rtr_mrg=0.0020  enc_gn=5.1094
        ctrl=0.0001  tex=0.0310  im_rew=0.0344  im_ret=0.5067  value=0.0296  wm_gn=0.2434
        z_norm=0.6252  z_max=0.9674  jump=0.0000  cons=0.0715  sol=1.0019  e_var=0.0005  ch_ent=2.5381  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5067  dret=0.4817  term=0.0291  bnd=0.0519  chart_acc=0.4604  chart_ent=1.8360  rw_drift=0.0000
        v_err=1.2759  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7300
        bnd_x=0.0561  bell=0.0347  bell_s=0.0155  rtg_e=1.2759  rtg_b=-1.2758  cal_e=1.2758  u_l2=0.0515  cov_n=0.0015
        col=11.0817  smp=0.0041  enc_t=0.2538  bnd_t=0.0400  wm_t=1.0152  crt_t=0.0085  diag_t=0.5697
        charts: 16/16 active  c0=0.06 c1=0.01 c2=0.20 c3=0.12 c4=0.02 c5=0.11 c6=0.05 c7=0.05 c8=0.03 c9=0.02 c10=0.05 c11=0.07 c12=0.04 c13=0.05 c14=0.06 c15=0.06
        symbols: 79/128 active  c0=5/8(H=1.21) c1=3/8(H=0.81) c2=6/8(H=1.55) c3=5/8(H=1.03) c4=3/8(H=0.89) c5=7/8(H=1.58) c6=5/8(H=1.28) c7=5/8(H=1.33) c8=6/8(H=1.44) c9=4/8(H=0.96) c10=5/8(H=1.35) c11=6/8(H=1.57) c12=4/8(H=1.12) c13=5/8(H=1.17) c14=5/8(H=1.36) c15=5/8(H=1.31)
E0170 [5upd]  ep_rew=14.3133  rew_20=13.8892  L_geo=0.6159  L_rew=0.0004  L_chart=1.8082  L_crit=0.2678  L_bnd=0.5699  lr=0.0010  dt=18.47s
        recon=0.2296  vq=0.2474  code_H=1.3870  code_px=4.0227  ch_usage=0.1117  rtr_mrg=0.0037  enc_gn=6.4982
        ctrl=0.0001  tex=0.0301  im_rew=0.0425  im_ret=0.6070  value=0.0150  wm_gn=0.3275
        z_norm=0.6260  z_max=0.9392  jump=0.0000  cons=0.0679  sol=0.9982  e_var=0.0005  ch_ent=2.6014  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6070  dret=0.5944  term=0.0147  bnd=0.0212  chart_acc=0.4725  chart_ent=1.6858  rw_drift=0.0000
        v_err=1.3877  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4868
        bnd_x=0.0230  bell=0.0384  bell_s=0.0241  rtg_e=1.3877  rtg_b=-1.3877  cal_e=1.3877  u_l2=0.0432  cov_n=0.0013
        col=11.1073  smp=0.0035  enc_t=0.2546  bnd_t=0.0397  wm_t=1.0416  crt_t=0.0085  diag_t=0.5722
        charts: 16/16 active  c0=0.18 c1=0.05 c2=0.10 c3=0.05 c4=0.03 c5=0.08 c6=0.09 c7=0.05 c8=0.03 c9=0.09 c10=0.05 c11=0.06 c12=0.03 c13=0.02 c14=0.07 c15=0.03
        symbols: 88/128 active  c0=6/8(H=1.51) c1=5/8(H=1.49) c2=8/8(H=1.61) c3=5/8(H=1.24) c4=6/8(H=1.58) c5=4/8(H=0.75) c6=5/8(H=1.23) c7=5/8(H=1.17) c8=6/8(H=1.22) c9=7/8(H=1.04) c10=5/8(H=1.27) c11=4/8(H=1.26) c12=6/8(H=1.38) c13=5/8(H=1.33) c14=6/8(H=1.40) c15=5/8(H=1.21)
E0171 [5upd]  ep_rew=14.2123  rew_20=13.5126  L_geo=0.5793  L_rew=0.0004  L_chart=1.7077  L_crit=0.2689  L_bnd=0.4779  lr=0.0010  dt=18.24s
        recon=0.2285  vq=0.2579  code_H=1.3661  code_px=3.9281  ch_usage=0.0722  rtr_mrg=0.0024  enc_gn=4.4583
        ctrl=0.0000  tex=0.0314  im_rew=0.0391  im_ret=0.5498  value=0.0032  wm_gn=0.2013
        z_norm=0.6144  z_max=0.9434  jump=0.0000  cons=0.0769  sol=0.9925  e_var=0.0003  ch_ent=2.5441  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5498  dret=0.5472  term=0.0031  bnd=0.0048  chart_acc=0.4471  chart_ent=1.7935  rw_drift=0.0000
        v_err=1.4325  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7632
        bnd_x=0.0053  bell=0.0404  bell_s=0.0345  rtg_e=1.4325  rtg_b=-1.4325  cal_e=1.4325  u_l2=0.0512  cov_n=0.0002
        col=11.0443  smp=0.0038  enc_t=0.2532  bnd_t=0.0399  wm_t=1.0122  crt_t=0.0085  diag_t=0.5591
        charts: 16/16 active  c0=0.19 c1=0.13 c2=0.05 c3=0.08 c4=0.04 c5=0.01 c6=0.09 c7=0.05 c8=0.03 c9=0.10 c10=0.04 c11=0.05 c12=0.06 c13=0.02 c14=0.03 c15=0.03
        symbols: 100/128 active  c0=5/8(H=0.92) c1=8/8(H=1.10) c2=5/8(H=0.90) c3=6/8(H=1.47) c4=7/8(H=1.49) c5=5/8(H=1.34) c6=6/8(H=1.48) c7=8/8(H=1.07) c8=8/8(H=1.26) c9=5/8(H=0.92) c10=7/8(H=1.54) c11=5/8(H=1.15) c12=6/8(H=1.11) c13=6/8(H=1.13) c14=6/8(H=1.15) c15=7/8(H=1.08)
E0172 [5upd]  ep_rew=14.6302  rew_20=13.3274  L_geo=0.6475  L_rew=0.0005  L_chart=1.7813  L_crit=0.2801  L_bnd=0.4886  lr=0.0010  dt=18.35s
        recon=0.2220  vq=0.2490  code_H=1.3034  code_px=3.6996  ch_usage=0.0426  rtr_mrg=0.0013  enc_gn=7.1034
        ctrl=0.0000  tex=0.0267  im_rew=0.0327  im_ret=0.4642  value=0.0076  wm_gn=0.2111
        z_norm=0.6843  z_max=0.9276  jump=0.0000  cons=0.0655  sol=0.9966  e_var=0.0011  ch_ent=2.5891  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4642  dret=0.4575  term=0.0077  bnd=0.0146  chart_acc=0.3983  chart_ent=1.8599  rw_drift=0.0000
        v_err=1.2827  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8156
        bnd_x=0.0167  bell=0.0367  bell_s=0.0377  rtg_e=1.2827  rtg_b=-1.2827  cal_e=1.2827  u_l2=0.0705  cov_n=0.0006
        col=11.2060  smp=0.0029  enc_t=0.2522  bnd_t=0.0394  wm_t=1.0034  crt_t=0.0084  diag_t=0.5624
        charts: 16/16 active  c0=0.05 c1=0.05 c2=0.14 c3=0.02 c4=0.02 c5=0.15 c6=0.05 c7=0.08 c8=0.01 c9=0.09 c10=0.09 c11=0.03 c12=0.07 c13=0.03 c14=0.07 c15=0.06
        symbols: 82/128 active  c0=4/8(H=1.07) c1=7/8(H=1.39) c2=7/8(H=1.57) c3=3/8(H=0.84) c4=4/8(H=1.27) c5=5/8(H=0.97) c6=5/8(H=1.24) c7=7/8(H=1.59) c8=4/8(H=1.13) c9=6/8(H=1.36) c10=5/8(H=1.46) c11=3/8(H=0.93) c12=6/8(H=1.64) c13=5/8(H=1.15) c14=6/8(H=1.44) c15=5/8(H=1.36)
E0173 [5upd]  ep_rew=12.4011  rew_20=13.2257  L_geo=0.6559  L_rew=0.0007  L_chart=1.8375  L_crit=0.3018  L_bnd=0.4725  lr=0.0010  dt=18.22s
        recon=0.2123  vq=0.2474  code_H=1.3777  code_px=3.9700  ch_usage=0.0519  rtr_mrg=0.0026  enc_gn=6.7351
        ctrl=0.0001  tex=0.0283  im_rew=0.0424  im_ret=0.6103  value=0.0187  wm_gn=0.2465
        z_norm=0.6421  z_max=0.9669  jump=0.0000  cons=0.0572  sol=0.9933  e_var=0.0005  ch_ent=2.6273  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6103  dret=0.5939  term=0.0190  bnd=0.0276  chart_acc=0.4279  chart_ent=1.8142  rw_drift=0.0000
        v_err=1.3942  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7712
        bnd_x=0.0310  bell=0.0376  bell_s=0.0141  rtg_e=1.3942  rtg_b=-1.3941  cal_e=1.3941  u_l2=0.0922  cov_n=0.0014
        col=11.0180  smp=0.0032  enc_t=0.2528  bnd_t=0.0394  wm_t=1.0098  crt_t=0.0086  diag_t=0.5850
        charts: 16/16 active  c0=0.08 c1=0.10 c2=0.12 c3=0.12 c4=0.04 c5=0.05 c6=0.07 c7=0.07 c8=0.01 c9=0.07 c10=0.03 c11=0.07 c12=0.07 c13=0.01 c14=0.05 c15=0.04
        symbols: 91/128 active  c0=3/8(H=0.32) c1=7/8(H=1.32) c2=7/8(H=1.31) c3=5/8(H=0.94) c4=5/8(H=1.34) c5=6/8(H=1.02) c6=6/8(H=1.31) c7=8/8(H=1.22) c8=4/8(H=1.30) c9=5/8(H=1.08) c10=5/8(H=1.39) c11=5/8(H=1.10) c12=8/8(H=1.62) c13=5/8(H=1.27) c14=5/8(H=1.25) c15=7/8(H=1.31)
E0174 [5upd]  ep_rew=20.6234  rew_20=13.7652  L_geo=0.5855  L_rew=0.0003  L_chart=1.6940  L_crit=0.2789  L_bnd=0.4981  lr=0.0010  dt=18.24s
        recon=0.1937  vq=0.2902  code_H=1.2247  code_px=3.4164  ch_usage=0.1179  rtr_mrg=0.0040  enc_gn=11.4605
        ctrl=0.0001  tex=0.0270  im_rew=0.0405  im_ret=0.5885  value=0.0256  wm_gn=0.2591
        z_norm=0.6589  z_max=0.9420  jump=0.0000  cons=0.0572  sol=1.0010  e_var=0.0009  ch_ent=2.6608  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5885  dret=0.5668  term=0.0252  bnd=0.0383  chart_acc=0.4658  chart_ent=1.7714  rw_drift=0.0000
        v_err=1.3290  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7874
        bnd_x=0.0456  bell=0.0366  bell_s=0.0239  rtg_e=1.3290  rtg_b=-1.3290  cal_e=1.3290  u_l2=0.0918  cov_n=0.0013
        col=11.0216  smp=0.0030  enc_t=0.2531  bnd_t=0.0398  wm_t=1.0150  crt_t=0.0086  diag_t=0.5641
        charts: 16/16 active  c0=0.05 c1=0.02 c2=0.04 c3=0.08 c4=0.05 c5=0.13 c6=0.03 c7=0.08 c8=0.06 c9=0.05 c10=0.06 c11=0.13 c12=0.04 c13=0.04 c14=0.08 c15=0.05
        symbols: 101/128 active  c0=5/8(H=1.25) c1=6/8(H=1.59) c2=7/8(H=1.25) c3=6/8(H=1.34) c4=7/8(H=1.55) c5=6/8(H=0.86) c6=4/8(H=1.34) c7=7/8(H=0.99) c8=6/8(H=0.95) c9=6/8(H=1.23) c10=4/8(H=1.05) c11=6/8(H=1.63) c12=8/8(H=1.57) c13=7/8(H=1.55) c14=8/8(H=1.60) c15=8/8(H=1.30)
E0175 [5upd]  ep_rew=14.2395  rew_20=14.3685  L_geo=0.6821  L_rew=0.0003  L_chart=1.8631  L_crit=0.2758  L_bnd=0.6389  lr=0.0010  dt=16.64s
        recon=0.2103  vq=0.2368  code_H=1.4141  code_px=4.1249  ch_usage=0.0756  rtr_mrg=0.0029  enc_gn=5.6190
        ctrl=0.0001  tex=0.0289  im_rew=0.0330  im_ret=0.4811  value=0.0235  wm_gn=0.3658
        z_norm=0.6177  z_max=0.9713  jump=0.0000  cons=0.0726  sol=1.0000  e_var=0.0010  ch_ent=2.6662  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4811  dret=0.4613  term=0.0231  bnd=0.0430  chart_acc=0.4363  chart_ent=1.8219  rw_drift=0.0000
        v_err=1.4072  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.0400
        bnd_x=0.0448  bell=0.0382  bell_s=0.0253  rtg_e=1.4072  rtg_b=-1.4072  cal_e=1.4072  u_l2=0.0764  cov_n=0.0017
        col=11.0123  smp=0.0029  enc_t=0.2344  bnd_t=0.0355  wm_t=0.7598  crt_t=0.0061  diag_t=0.3885
        charts: 16/16 active  c0=0.13 c1=0.05 c2=0.08 c3=0.12 c4=0.02 c5=0.04 c6=0.03 c7=0.03 c8=0.05 c9=0.07 c10=0.05 c11=0.10 c12=0.04 c13=0.07 c14=0.07 c15=0.05
        symbols: 90/128 active  c0=6/8(H=1.27) c1=4/8(H=0.59) c2=6/8(H=1.29) c3=4/8(H=0.86) c4=4/8(H=0.92) c5=7/8(H=1.26) c6=5/8(H=1.48) c7=5/8(H=0.74) c8=5/8(H=1.28) c9=6/8(H=1.43) c10=5/8(H=1.13) c11=6/8(H=1.60) c12=6/8(H=1.29) c13=7/8(H=1.52) c14=8/8(H=1.49) c15=6/8(H=1.26)
E0176 [5upd]  ep_rew=15.7612  rew_20=14.3154  L_geo=0.7226  L_rew=0.0001  L_chart=1.8922  L_crit=0.2398  L_bnd=0.7703  lr=0.0010  dt=12.52s
        recon=0.2333  vq=0.2425  code_H=1.3948  code_px=4.0381  ch_usage=0.0894  rtr_mrg=0.0025  enc_gn=7.4203
        ctrl=0.0000  tex=0.0289  im_rew=0.0364  im_ret=0.5121  value=0.0035  wm_gn=0.3364
        z_norm=0.6260  z_max=0.9566  jump=0.0000  cons=0.0792  sol=0.9931  e_var=0.0005  ch_ent=2.6025  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5121  dret=0.5091  term=0.0036  bnd=0.0060  chart_acc=0.4038  chart_ent=1.8482  rw_drift=0.0000
        v_err=1.3384  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.3969
        bnd_x=0.0068  bell=0.0362  bell_s=0.0123  rtg_e=1.3384  rtg_b=-1.3384  cal_e=1.3384  u_l2=0.0561  cov_n=0.0005
        col=7.1720  smp=0.0031  enc_t=0.2256  bnd_t=0.0344  wm_t=0.7174  crt_t=0.0059  diag_t=0.3745
        charts: 16/16 active  c0=0.07 c1=0.12 c2=0.01 c3=0.08 c4=0.02 c5=0.06 c6=0.10 c7=0.01 c8=0.05 c9=0.09 c10=0.04 c11=0.11 c12=0.03 c13=0.09 c14=0.09 c15=0.05
        symbols: 92/128 active  c0=4/8(H=0.79) c1=8/8(H=1.25) c2=5/8(H=1.26) c3=5/8(H=1.56) c4=5/8(H=1.02) c5=7/8(H=1.28) c6=6/8(H=1.52) c7=6/8(H=1.53) c8=6/8(H=1.44) c9=6/8(H=0.73) c10=5/8(H=1.23) c11=6/8(H=1.66) c12=4/8(H=1.10) c13=8/8(H=1.56) c14=5/8(H=1.25) c15=6/8(H=1.09)
E0177 [5upd]  ep_rew=14.0012  rew_20=14.3236  L_geo=0.6345  L_rew=0.0007  L_chart=1.6840  L_crit=0.3007  L_bnd=0.7083  lr=0.0010  dt=12.68s
        recon=0.2096  vq=0.2734  code_H=1.3881  code_px=4.0226  ch_usage=0.0943  rtr_mrg=0.0028  enc_gn=6.6119
        ctrl=0.0000  tex=0.0267  im_rew=0.0409  im_ret=0.5724  value=0.0000  wm_gn=0.3464
        z_norm=0.6813  z_max=0.9636  jump=0.0000  cons=0.0770  sol=0.9982  e_var=0.0007  ch_ent=2.5901  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5724  dret=0.5724  term=0.0000  bnd=0.0000  chart_acc=0.4704  chart_ent=1.7625  rw_drift=0.0000
        v_err=1.3058  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0722
        bnd_x=0.0001  bell=0.0373  bell_s=0.0381  rtg_e=1.3058  rtg_b=-1.3058  cal_e=1.3058  u_l2=0.0551  cov_n=0.0000
        col=7.2800  smp=0.0034  enc_t=0.2259  bnd_t=0.0345  wm_t=0.7193  crt_t=0.0061  diag_t=0.4143
        charts: 16/16 active  c0=0.17 c1=0.11 c2=0.04 c3=0.06 c4=0.03 c5=0.04 c6=0.09 c7=0.02 c8=0.03 c9=0.11 c10=0.06 c11=0.05 c12=0.01 c13=0.07 c14=0.06 c15=0.04
        symbols: 88/128 active  c0=4/8(H=0.87) c1=6/8(H=1.17) c2=5/8(H=1.17) c3=6/8(H=1.36) c4=6/8(H=1.50) c5=6/8(H=1.42) c6=6/8(H=1.13) c7=6/8(H=1.62) c8=4/8(H=1.00) c9=5/8(H=1.10) c10=8/8(H=1.90) c11=4/8(H=0.95) c12=4/8(H=1.18) c13=7/8(H=1.59) c14=6/8(H=1.22) c15=5/8(H=1.56)
E0178 [5upd]  ep_rew=14.6148  rew_20=13.9737  L_geo=0.6912  L_rew=0.0007  L_chart=1.7734  L_crit=0.3474  L_bnd=0.6408  lr=0.0010  dt=12.99s
        recon=0.2164  vq=0.2971  code_H=1.3761  code_px=3.9641  ch_usage=0.1010  rtr_mrg=0.0025  enc_gn=8.0491
        ctrl=0.0002  tex=0.0288  im_rew=0.0483  im_ret=0.6861  value=0.0122  wm_gn=0.2627
        z_norm=0.6515  z_max=0.9699  jump=0.0000  cons=0.0686  sol=1.0038  e_var=0.0005  ch_ent=2.6120  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6861  dret=0.6759  term=0.0119  bnd=0.0151  chart_acc=0.4192  chart_ent=1.8040  rw_drift=0.0000
        v_err=1.5798  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1665
        bnd_x=0.0185  bell=0.0457  bell_s=0.0425  rtg_e=1.5798  rtg_b=-1.5798  cal_e=1.5798  u_l2=0.0586  cov_n=0.0024
        col=7.5005  smp=0.0016  enc_t=0.2311  bnd_t=0.0347  wm_t=0.7304  crt_t=0.0063  diag_t=0.4139
        charts: 16/16 active  c0=0.10 c1=0.05 c2=0.06 c3=0.08 c4=0.09 c5=0.04 c6=0.03 c7=0.05 c8=0.06 c9=0.05 c10=0.06 c11=0.18 c12=0.02 c13=0.08 c14=0.03 c15=0.03
        symbols: 87/128 active  c0=5/8(H=1.04) c1=5/8(H=0.76) c2=7/8(H=1.28) c3=5/8(H=1.12) c4=6/8(H=1.27) c5=7/8(H=1.58) c6=5/8(H=1.40) c7=6/8(H=1.32) c8=5/8(H=1.08) c9=6/8(H=1.23) c10=5/8(H=1.23) c11=5/8(H=1.19) c12=4/8(H=1.27) c13=6/8(H=1.48) c14=5/8(H=1.35) c15=5/8(H=1.40)
E0179 [5upd]  ep_rew=13.0553  rew_20=13.6208  L_geo=0.6947  L_rew=0.0004  L_chart=1.8408  L_crit=0.3006  L_bnd=0.5314  lr=0.0010  dt=12.86s
        recon=0.1747  vq=0.2953  code_H=1.2945  code_px=3.6623  ch_usage=0.0659  rtr_mrg=0.0021  enc_gn=7.7237
        ctrl=0.0002  tex=0.0277  im_rew=0.0422  im_ret=0.6015  value=0.0128  wm_gn=0.5112
        z_norm=0.6717  z_max=0.9810  jump=0.0000  cons=0.0659  sol=0.9983  e_var=0.0007  ch_ent=2.5365  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6015  dret=0.5906  term=0.0127  bnd=0.0185  chart_acc=0.3721  chart_ent=1.7617  rw_drift=0.0000
        v_err=1.3955  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5251
        bnd_x=0.0223  bell=0.0384  bell_s=0.0181  rtg_e=1.3955  rtg_b=-1.3955  cal_e=1.3955  u_l2=0.0690  cov_n=0.0030
        col=7.4797  smp=0.0037  enc_t=0.2261  bnd_t=0.0355  wm_t=0.7195  crt_t=0.0061  diag_t=0.3818
        charts: 16/16 active  c0=0.20 c1=0.05 c2=0.04 c3=0.09 c4=0.02 c5=0.00 c6=0.03 c7=0.02 c8=0.08 c9=0.11 c10=0.04 c11=0.07 c12=0.03 c13=0.07 c14=0.06 c15=0.08
        symbols: 80/128 active  c0=6/8(H=1.40) c1=7/8(H=1.39) c2=3/8(H=0.45) c3=5/8(H=1.07) c4=6/8(H=1.60) c5=5/8(H=1.46) c6=5/8(H=1.18) c7=4/8(H=1.17) c8=3/8(H=0.69) c9=6/8(H=1.00) c10=4/8(H=1.22) c11=6/8(H=1.51) c12=6/8(H=1.41) c13=4/8(H=1.24) c14=4/8(H=1.06) c15=6/8(H=1.72)
E0180 [5upd]  ep_rew=8.8662  rew_20=12.8846  L_geo=0.6378  L_rew=0.0003  L_chart=1.6181  L_crit=0.3311  L_bnd=0.5103  lr=0.0010  dt=13.30s
        recon=0.1829  vq=0.3219  code_H=1.2815  code_px=3.6203  ch_usage=0.0239  rtr_mrg=0.0019  enc_gn=5.5291
        ctrl=0.0001  tex=0.0285  im_rew=0.0384  im_ret=0.5438  value=0.0076  wm_gn=0.2119
        z_norm=0.6579  z_max=0.9867  jump=0.0000  cons=0.0769  sol=1.0017  e_var=0.0004  ch_ent=2.6721  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5438  dret=0.5374  term=0.0075  bnd=0.0120  chart_acc=0.4254  chart_ent=1.7380  rw_drift=0.0000
        v_err=1.5086  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8028
        bnd_x=0.0148  bell=0.0410  bell_s=0.0293  rtg_e=1.5086  rtg_b=-1.5086  cal_e=1.5086  u_l2=0.0877  cov_n=0.0017
        col=7.6655  smp=0.0030  enc_t=0.2541  bnd_t=0.0341  wm_t=0.7363  crt_t=0.0062  diag_t=0.4263
        charts: 16/16 active  c0=0.11 c1=0.04 c2=0.02 c3=0.10 c4=0.04 c5=0.04 c6=0.10 c7=0.06 c8=0.06 c9=0.07 c10=0.05 c11=0.10 c12=0.03 c13=0.07 c14=0.06 c15=0.05
        symbols: 100/128 active  c0=6/8(H=1.05) c1=5/8(H=1.25) c2=6/8(H=1.44) c3=6/8(H=1.52) c4=7/8(H=1.15) c5=5/8(H=0.57) c6=6/8(H=1.41) c7=7/8(H=1.13) c8=5/8(H=1.36) c9=4/8(H=0.88) c10=6/8(H=1.17) c11=6/8(H=1.43) c12=8/8(H=1.47) c13=7/8(H=1.14) c14=8/8(H=1.72) c15=8/8(H=1.56)
  EVAL  reward=11.9 +/- 3.7  len=300
E0181 [5upd]  ep_rew=14.3061  rew_20=12.5991  L_geo=0.6062  L_rew=0.0009  L_chart=1.6607  L_crit=0.3272  L_bnd=0.4551  lr=0.0010  dt=12.85s
        recon=0.1796  vq=0.2844  code_H=1.2647  code_px=3.5492  ch_usage=0.0328  rtr_mrg=0.0024  enc_gn=6.3144
        ctrl=0.0001  tex=0.0288  im_rew=0.0478  im_ret=0.6733  value=0.0059  wm_gn=0.2637
        z_norm=0.6453  z_max=0.9095  jump=0.0000  cons=0.0611  sol=0.9981  e_var=0.0006  ch_ent=2.6492  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6733  dret=0.6682  term=0.0059  bnd=0.0077  chart_acc=0.5033  chart_ent=1.7895  rw_drift=0.0000
        v_err=1.4174  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9050
        bnd_x=0.0091  bell=0.0398  bell_s=0.0334  rtg_e=1.4174  rtg_b=-1.4174  cal_e=1.4174  u_l2=0.1217  cov_n=0.0011
        col=7.3705  smp=0.0024  enc_t=0.2284  bnd_t=0.0350  wm_t=0.7340  crt_t=0.0063  diag_t=0.4042
        charts: 16/16 active  c0=0.04 c1=0.04 c2=0.09 c3=0.08 c4=0.06 c5=0.05 c6=0.13 c7=0.06 c8=0.03 c9=0.07 c10=0.05 c11=0.06 c12=0.07 c13=0.02 c14=0.13 c15=0.02
        symbols: 91/128 active  c0=4/8(H=0.79) c1=5/8(H=1.30) c2=6/8(H=1.23) c3=6/8(H=1.06) c4=6/8(H=1.42) c5=6/8(H=0.89) c6=7/8(H=1.37) c7=8/8(H=1.27) c8=3/8(H=0.45) c9=6/8(H=1.56) c10=4/8(H=1.04) c11=7/8(H=1.07) c12=6/8(H=1.46) c13=4/8(H=0.97) c14=8/8(H=1.42) c15=5/8(H=1.27)
E0182 [5upd]  ep_rew=13.9584  rew_20=12.3244  L_geo=0.6415  L_rew=0.0005  L_chart=1.7721  L_crit=0.3001  L_bnd=0.4694  lr=0.0010  dt=13.27s
        recon=0.1748  vq=0.2478  code_H=1.3033  code_px=3.6954  ch_usage=0.1081  rtr_mrg=0.0023  enc_gn=5.2343
        ctrl=0.0001  tex=0.0287  im_rew=0.0449  im_ret=0.6350  value=0.0072  wm_gn=0.1963
        z_norm=0.6355  z_max=0.9599  jump=0.0000  cons=0.0889  sol=0.9969  e_var=0.0006  ch_ent=2.6650  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6350  dret=0.6290  term=0.0070  bnd=0.0095  chart_acc=0.4392  chart_ent=1.7270  rw_drift=0.0000
        v_err=1.4173  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8854
        bnd_x=0.0123  bell=0.0386  bell_s=0.0199  rtg_e=1.4173  rtg_b=-1.4173  cal_e=1.4173  u_l2=0.1291  cov_n=0.0012
        col=7.6940  smp=0.0024  enc_t=0.2308  bnd_t=0.0348  wm_t=0.7535  crt_t=0.0065  diag_t=0.3886
        charts: 16/16 active  c0=0.10 c1=0.04 c2=0.07 c3=0.08 c4=0.04 c5=0.07 c6=0.06 c7=0.07 c8=0.01 c9=0.07 c10=0.04 c11=0.06 c12=0.06 c13=0.03 c14=0.15 c15=0.05
        symbols: 91/128 active  c0=5/8(H=1.27) c1=5/8(H=0.95) c2=6/8(H=1.36) c3=6/8(H=1.09) c4=7/8(H=1.46) c5=6/8(H=1.32) c6=6/8(H=1.19) c7=7/8(H=1.47) c8=4/8(H=1.27) c9=5/8(H=1.31) c10=5/8(H=1.33) c11=5/8(H=1.33) c12=8/8(H=1.74) c13=5/8(H=1.24) c14=6/8(H=1.25) c15=5/8(H=1.48)
E0183 [5upd]  ep_rew=14.2164  rew_20=12.7909  L_geo=0.6050  L_rew=0.0004  L_chart=1.7717  L_crit=0.2759  L_bnd=0.4623  lr=0.0010  dt=13.40s
        recon=0.2064  vq=0.2311  code_H=1.3356  code_px=3.8100  ch_usage=0.0821  rtr_mrg=0.0039  enc_gn=9.1217
        ctrl=0.0001  tex=0.0315  im_rew=0.0385  im_ret=0.5453  value=0.0082  wm_gn=0.2294
        z_norm=0.6053  z_max=0.9736  jump=0.0000  cons=0.0670  sol=0.9979  e_var=0.0003  ch_ent=2.6786  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5453  dret=0.5384  term=0.0081  bnd=0.0129  chart_acc=0.4629  chart_ent=1.8056  rw_drift=0.0000
        v_err=1.4693  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8368
        bnd_x=0.0164  bell=0.0403  bell_s=0.0243  rtg_e=1.4693  rtg_b=-1.4693  cal_e=1.4693  u_l2=0.1079  cov_n=0.0013
        col=7.8540  smp=0.0034  enc_t=0.2311  bnd_t=0.0350  wm_t=0.7430  crt_t=0.0066  diag_t=0.4128
        charts: 16/16 active  c0=0.07 c1=0.04 c2=0.08 c3=0.08 c4=0.07 c5=0.06 c6=0.07 c7=0.03 c8=0.04 c9=0.06 c10=0.05 c11=0.16 c12=0.03 c13=0.06 c14=0.06 c15=0.04
        symbols: 93/128 active  c0=5/8(H=1.09) c1=7/8(H=1.60) c2=7/8(H=1.40) c3=5/8(H=0.59) c4=6/8(H=1.29) c5=6/8(H=1.25) c6=5/8(H=1.51) c7=7/8(H=1.66) c8=6/8(H=1.33) c9=4/8(H=0.93) c10=5/8(H=1.39) c11=6/8(H=1.30) c12=6/8(H=1.65) c13=6/8(H=1.17) c14=6/8(H=1.21) c15=6/8(H=1.48)
E0184 [5upd]  ep_rew=14.2934  rew_20=12.8512  L_geo=0.7463  L_rew=0.0007  L_chart=1.9524  L_crit=0.2758  L_bnd=0.4406  lr=0.0010  dt=19.05s
        recon=0.2159  vq=0.2317  code_H=1.4024  code_px=4.0861  ch_usage=0.0442  rtr_mrg=0.0030  enc_gn=5.9313
        ctrl=0.0001  tex=0.0319  im_rew=0.0399  im_ret=0.5640  value=0.0051  wm_gn=0.2337
        z_norm=0.6248  z_max=0.9584  jump=0.0000  cons=0.0777  sol=0.9974  e_var=0.0048  ch_ent=2.6820  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5640  dret=0.5594  term=0.0054  bnd=0.0083  chart_acc=0.3650  chart_ent=1.8663  rw_drift=0.0000
        v_err=1.3674  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8602
        bnd_x=0.0107  bell=0.0373  bell_s=0.0197  rtg_e=1.3674  rtg_b=-1.3674  cal_e=1.3674  u_l2=0.0875  cov_n=0.0009
        col=11.2666  smp=0.0035  enc_t=0.2624  bnd_t=0.0416  wm_t=1.1078  crt_t=0.0094  diag_t=0.6018
        charts: 16/16 active  c0=0.05 c1=0.04 c2=0.04 c3=0.10 c4=0.02 c5=0.06 c6=0.08 c7=0.04 c8=0.07 c9=0.12 c10=0.06 c11=0.09 c12=0.04 c13=0.08 c14=0.05 c15=0.06
        symbols: 90/128 active  c0=3/8(H=0.73) c1=3/8(H=0.43) c2=6/8(H=1.03) c3=5/8(H=0.42) c4=5/8(H=1.34) c5=5/8(H=1.24) c6=5/8(H=1.35) c7=7/8(H=1.31) c8=5/8(H=1.08) c9=4/8(H=0.87) c10=6/8(H=1.01) c11=6/8(H=1.19) c12=7/8(H=1.22) c13=8/8(H=1.62) c14=8/8(H=1.46) c15=7/8(H=1.26)
E0185 [5upd]  ep_rew=7.2354  rew_20=13.1688  L_geo=0.6449  L_rew=0.0004  L_chart=1.6777  L_crit=0.2817  L_bnd=0.4755  lr=0.0010  dt=19.78s
        recon=0.1930  vq=0.2702  code_H=1.2850  code_px=3.6287  ch_usage=0.0887  rtr_mrg=0.0039  enc_gn=5.8679
        ctrl=0.0001  tex=0.0280  im_rew=0.0378  im_ret=0.5350  value=0.0064  wm_gn=0.1663
        z_norm=0.6657  z_max=0.9444  jump=0.0000  cons=0.0723  sol=0.9982  e_var=0.0217  ch_ent=2.5934  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5350  dret=0.5294  term=0.0065  bnd=0.0106  chart_acc=0.4775  chart_ent=1.7447  rw_drift=0.0000
        v_err=1.2782  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8795
        bnd_x=0.0124  bell=0.0342  bell_s=0.0122  rtg_e=1.2782  rtg_b=-1.2782  cal_e=1.2782  u_l2=0.0811  cov_n=0.0010
        col=12.1642  smp=0.0048  enc_t=0.2567  bnd_t=0.0407  wm_t=1.0864  crt_t=0.0090  diag_t=0.5842
        charts: 16/16 active  c0=0.15 c1=0.02 c2=0.11 c3=0.09 c4=0.01 c5=0.02 c6=0.07 c7=0.06 c8=0.02 c9=0.07 c10=0.04 c11=0.07 c12=0.04 c13=0.05 c14=0.11 c15=0.06
        symbols: 79/128 active  c0=4/8(H=0.94) c1=2/8(H=0.43) c2=6/8(H=1.19) c3=5/8(H=0.47) c4=4/8(H=1.06) c5=5/8(H=1.05) c6=5/8(H=1.05) c7=7/8(H=1.45) c8=5/8(H=1.30) c9=4/8(H=0.82) c10=6/8(H=1.44) c11=5/8(H=1.37) c12=5/8(H=1.11) c13=4/8(H=1.15) c14=6/8(H=1.60) c15=6/8(H=1.68)
E0186 [5upd]  ep_rew=13.6947  rew_20=13.4192  L_geo=0.6898  L_rew=0.0002  L_chart=1.8038  L_crit=0.2866  L_bnd=0.4539  lr=0.0010  dt=18.96s
        recon=0.1799  vq=0.2719  code_H=1.2547  code_px=3.5273  ch_usage=0.0797  rtr_mrg=0.0019  enc_gn=7.9916
        ctrl=0.0001  tex=0.0284  im_rew=0.0389  im_ret=0.5530  value=0.0107  wm_gn=0.2145
        z_norm=0.6667  z_max=0.9544  jump=0.0000  cons=0.0695  sol=0.9997  e_var=0.0009  ch_ent=2.5690  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5530  dret=0.5440  term=0.0105  bnd=0.0166  chart_acc=0.3758  chart_ent=1.8110  rw_drift=0.0000
        v_err=1.3393  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8985
        bnd_x=0.0193  bell=0.0367  bell_s=0.0141  rtg_e=1.3393  rtg_b=-1.3393  cal_e=1.3393  u_l2=0.0673  cov_n=0.0014
        col=11.5387  smp=0.0048  enc_t=0.2558  bnd_t=0.0406  wm_t=1.0485  crt_t=0.0089  diag_t=0.5824
        charts: 16/16 active  c0=0.04 c1=0.01 c2=0.05 c3=0.12 c4=0.01 c5=0.08 c6=0.10 c7=0.05 c8=0.02 c9=0.10 c10=0.03 c11=0.16 c12=0.07 c13=0.03 c14=0.06 c15=0.06
        symbols: 74/128 active  c0=4/8(H=0.66) c1=4/8(H=0.91) c2=6/8(H=1.06) c3=4/8(H=0.76) c4=4/8(H=1.09) c5=4/8(H=0.66) c6=4/8(H=1.35) c7=5/8(H=0.80) c8=5/8(H=0.80) c9=5/8(H=1.02) c10=3/8(H=0.93) c11=7/8(H=0.72) c12=4/8(H=0.63) c13=5/8(H=1.02) c14=5/8(H=1.29) c15=5/8(H=0.90)
E0187 [5upd]  ep_rew=7.4378  rew_20=13.3820  L_geo=0.5626  L_rew=0.0005  L_chart=1.7024  L_crit=0.2797  L_bnd=0.4081  lr=0.0010  dt=18.74s
        recon=0.1988  vq=0.2598  code_H=1.3278  code_px=3.8171  ch_usage=0.0343  rtr_mrg=0.0020  enc_gn=5.1606
        ctrl=0.0001  tex=0.0295  im_rew=0.0347  im_ret=0.4967  value=0.0133  wm_gn=0.2290
        z_norm=0.6350  z_max=0.9795  jump=0.0000  cons=0.0713  sol=0.9967  e_var=0.0005  ch_ent=2.6652  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4967  dret=0.4851  term=0.0135  bnd=0.0239  chart_acc=0.4508  chart_ent=1.8072  rw_drift=0.0000
        v_err=1.3219  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9312
        bnd_x=0.0271  bell=0.0362  bell_s=0.0173  rtg_e=1.3219  rtg_b=-1.3219  cal_e=1.3219  u_l2=0.0576  cov_n=0.0018
        col=11.3386  smp=0.0027  enc_t=0.2558  bnd_t=0.0405  wm_t=1.0444  crt_t=0.0088  diag_t=0.5835
        charts: 16/16 active  c0=0.14 c1=0.08 c2=0.08 c3=0.06 c4=0.02 c5=0.07 c6=0.08 c7=0.03 c8=0.02 c9=0.09 c10=0.05 c11=0.07 c12=0.04 c13=0.05 c14=0.05 c15=0.06
        symbols: 99/128 active  c0=6/8(H=1.37) c1=7/8(H=1.42) c2=8/8(H=1.27) c3=7/8(H=1.14) c4=4/8(H=1.16) c5=6/8(H=1.30) c6=6/8(H=1.23) c7=7/8(H=1.55) c8=4/8(H=1.16) c9=4/8(H=1.07) c10=6/8(H=1.18) c11=4/8(H=1.26) c12=8/8(H=1.38) c13=7/8(H=1.54) c14=8/8(H=1.35) c15=7/8(H=1.49)
E0188 [5upd]  ep_rew=14.2578  rew_20=13.3072  L_geo=0.6358  L_rew=0.0006  L_chart=1.7155  L_crit=0.3281  L_bnd=0.4332  lr=0.0010  dt=18.66s
        recon=0.2135  vq=0.2565  code_H=1.3536  code_px=3.8772  ch_usage=0.0777  rtr_mrg=0.0018  enc_gn=5.3269
        ctrl=0.0001  tex=0.0288  im_rew=0.0402  im_ret=0.5702  value=0.0082  wm_gn=0.1996
        z_norm=0.6292  z_max=0.9489  jump=0.0000  cons=0.0774  sol=1.0025  e_var=0.0005  ch_ent=2.5754  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5702  dret=0.5631  term=0.0083  bnd=0.0127  chart_acc=0.4467  chart_ent=1.7899  rw_drift=0.0000
        v_err=1.4734  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8898
        bnd_x=0.0137  bell=0.0396  bell_s=0.0196  rtg_e=1.4734  rtg_b=-1.4734  cal_e=1.4734  u_l2=0.0491  cov_n=0.0009
        col=11.2934  smp=0.0022  enc_t=0.2547  bnd_t=0.0403  wm_t=1.0396  crt_t=0.0093  diag_t=0.5748
        charts: 16/16 active  c0=0.14 c1=0.06 c2=0.06 c3=0.12 c4=0.05 c5=0.04 c6=0.09 c7=0.03 c8=0.05 c9=0.16 c10=0.02 c11=0.07 c12=0.03 c13=0.02 c14=0.03 c15=0.04
        symbols: 93/128 active  c0=8/8(H=1.73) c1=8/8(H=1.22) c2=7/8(H=1.14) c3=7/8(H=1.17) c4=8/8(H=1.46) c5=6/8(H=0.95) c6=7/8(H=1.13) c7=4/8(H=0.71) c8=6/8(H=0.76) c9=5/8(H=0.79) c10=5/8(H=0.81) c11=4/8(H=1.01) c12=4/8(H=0.91) c13=4/8(H=0.82) c14=5/8(H=1.24) c15=5/8(H=1.27)
E0189 [5upd]  ep_rew=19.6765  rew_20=14.3174  L_geo=0.7322  L_rew=0.0002  L_chart=1.8689  L_crit=0.2633  L_bnd=0.4470  lr=0.0010  dt=18.98s
        recon=0.1987  vq=0.2614  code_H=1.2581  code_px=3.5364  ch_usage=0.0494  rtr_mrg=0.0013  enc_gn=7.6706
        ctrl=0.0001  tex=0.0261  im_rew=0.0369  im_ret=0.5243  value=0.0092  wm_gn=0.2146
        z_norm=0.6760  z_max=0.9477  jump=0.0000  cons=0.0681  sol=1.0005  e_var=0.0007  ch_ent=2.6520  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5243  dret=0.5163  term=0.0094  bnd=0.0156  chart_acc=0.3979  chart_ent=1.7611  rw_drift=0.0000
        v_err=1.3369  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8914
        bnd_x=0.0183  bell=0.0362  bell_s=0.0169  rtg_e=1.3369  rtg_b=-1.3369  cal_e=1.3369  u_l2=0.0398  cov_n=0.0011
        col=11.3677  smp=0.0043  enc_t=0.2586  bnd_t=0.0409  wm_t=1.0817  crt_t=0.0091  diag_t=0.5863
        charts: 16/16 active  c0=0.11 c1=0.08 c2=0.09 c3=0.04 c4=0.02 c5=0.12 c6=0.09 c7=0.05 c8=0.02 c9=0.09 c10=0.05 c11=0.05 c12=0.02 c13=0.08 c14=0.05 c15=0.05
        symbols: 89/128 active  c0=5/8(H=1.50) c1=5/8(H=1.29) c2=7/8(H=1.62) c3=5/8(H=1.29) c4=5/8(H=0.72) c5=7/8(H=0.81) c6=6/8(H=0.75) c7=5/8(H=0.84) c8=4/8(H=1.03) c9=4/8(H=0.90) c10=5/8(H=1.25) c11=5/8(H=1.30) c12=6/8(H=1.42) c13=8/8(H=1.37) c14=5/8(H=1.20) c15=7/8(H=1.62)
E0190 [5upd]  ep_rew=14.7419  rew_20=14.8957  L_geo=0.6932  L_rew=0.0003  L_chart=1.8007  L_crit=0.3118  L_bnd=0.4017  lr=0.0010  dt=18.79s
        recon=0.2016  vq=0.2493  code_H=1.3561  code_px=3.8846  ch_usage=0.0998  rtr_mrg=0.0019  enc_gn=5.8441
        ctrl=0.0001  tex=0.0294  im_rew=0.0414  im_ret=0.5893  value=0.0125  wm_gn=0.1693
        z_norm=0.6441  z_max=0.9479  jump=0.0000  cons=0.1094  sol=0.9960  e_var=0.0007  ch_ent=2.6970  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5893  dret=0.5790  term=0.0120  bnd=0.0179  chart_acc=0.4279  chart_ent=1.8222  rw_drift=0.0000
        v_err=1.3287  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8941
        bnd_x=0.0206  bell=0.0361  bell_s=0.0120  rtg_e=1.3287  rtg_b=-1.3287  cal_e=1.3287  u_l2=0.0343  cov_n=0.0011
        col=11.2886  smp=0.0036  enc_t=0.2558  bnd_t=0.0403  wm_t=1.0674  crt_t=0.0087  diag_t=0.5723
        charts: 16/16 active  c0=0.07 c1=0.04 c2=0.06 c3=0.10 c4=0.03 c5=0.10 c6=0.05 c7=0.06 c8=0.02 c9=0.07 c10=0.08 c11=0.05 c12=0.07 c13=0.04 c14=0.10 c15=0.05
        symbols: 90/128 active  c0=6/8(H=1.39) c1=4/8(H=0.93) c2=7/8(H=1.51) c3=4/8(H=0.29) c4=7/8(H=1.46) c5=7/8(H=1.18) c6=5/8(H=1.23) c7=7/8(H=1.35) c8=4/8(H=1.10) c9=6/8(H=1.32) c10=5/8(H=1.41) c11=4/8(H=1.11) c12=8/8(H=1.59) c13=5/8(H=1.18) c14=4/8(H=1.02) c15=7/8(H=1.59)
E0191 [5upd]  ep_rew=7.7314  rew_20=14.1794  L_geo=0.6544  L_rew=0.0006  L_chart=1.7188  L_crit=0.2915  L_bnd=0.4332  lr=0.0010  dt=18.48s
        recon=0.1924  vq=0.2481  code_H=1.3919  code_px=4.0507  ch_usage=0.0309  rtr_mrg=0.0018  enc_gn=5.2680
        ctrl=0.0001  tex=0.0268  im_rew=0.0431  im_ret=0.6169  value=0.0163  wm_gn=0.1471
        z_norm=0.6611  z_max=0.9546  jump=0.0000  cons=0.0914  sol=0.9993  e_var=0.0006  ch_ent=2.6798  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6169  dret=0.6027  term=0.0164  bnd=0.0234  chart_acc=0.4346  chart_ent=1.8110  rw_drift=0.0000
        v_err=1.4479  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8720
        bnd_x=0.0273  bell=0.0414  bell_s=0.0370  rtg_e=1.4479  rtg_b=-1.4479  cal_e=1.4479  u_l2=0.0284  cov_n=0.0013
        col=11.1789  smp=0.0042  enc_t=0.2542  bnd_t=0.0397  wm_t=1.0279  crt_t=0.0086  diag_t=0.5721
        charts: 16/16 active  c0=0.13 c1=0.08 c2=0.11 c3=0.05 c4=0.06 c5=0.04 c6=0.05 c7=0.07 c8=0.01 c9=0.06 c10=0.05 c11=0.04 c12=0.07 c13=0.05 c14=0.07 c15=0.07
        symbols: 80/128 active  c0=5/8(H=1.38) c1=6/8(H=1.40) c2=5/8(H=0.95) c3=5/8(H=1.22) c4=6/8(H=1.24) c5=4/8(H=0.69) c6=6/8(H=1.46) c7=7/8(H=1.68) c8=3/8(H=1.04) c9=4/8(H=1.07) c10=5/8(H=1.35) c11=4/8(H=1.20) c12=6/8(H=1.37) c13=4/8(H=0.83) c14=5/8(H=1.23) c15=5/8(H=1.37)
E0192 [5upd]  ep_rew=13.1672  rew_20=14.3127  L_geo=0.7050  L_rew=0.0006  L_chart=1.7548  L_crit=0.3121  L_bnd=0.4527  lr=0.0010  dt=18.49s
        recon=0.1928  vq=0.2740  code_H=1.3435  code_px=3.8333  ch_usage=0.0465  rtr_mrg=0.0036  enc_gn=5.6647
        ctrl=0.0001  tex=0.0296  im_rew=0.0403  im_ret=0.5790  value=0.0179  wm_gn=0.2043
        z_norm=0.6287  z_max=0.9749  jump=0.0000  cons=0.0678  sol=0.9983  e_var=0.0005  ch_ent=2.6656  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5790  dret=0.5639  term=0.0176  bnd=0.0268  chart_acc=0.4508  chart_ent=1.7673  rw_drift=0.0000
        v_err=1.4888  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7544
        bnd_x=0.0298  bell=0.0406  bell_s=0.0299  rtg_e=1.4888  rtg_b=-1.4888  cal_e=1.4888  u_l2=0.0230  cov_n=0.0011
        col=11.1720  smp=0.0042  enc_t=0.2546  bnd_t=0.0401  wm_t=1.0305  crt_t=0.0091  diag_t=0.5766
        charts: 16/16 active  c0=0.04 c1=0.10 c2=0.04 c3=0.16 c4=0.04 c5=0.08 c6=0.06 c7=0.03 c8=0.03 c9=0.06 c10=0.06 c11=0.07 c12=0.04 c13=0.05 c14=0.08 c15=0.05
        symbols: 102/128 active  c0=5/8(H=0.88) c1=7/8(H=1.33) c2=6/8(H=0.90) c3=7/8(H=0.67) c4=5/8(H=0.89) c5=6/8(H=0.59) c6=7/8(H=1.29) c7=7/8(H=1.40) c8=5/8(H=1.19) c9=7/8(H=0.97) c10=7/8(H=1.68) c11=6/8(H=1.03) c12=7/8(H=1.57) c13=7/8(H=1.59) c14=8/8(H=1.41) c15=5/8(H=1.25)
E0193 [5upd]  ep_rew=15.7795  rew_20=14.4493  L_geo=0.6201  L_rew=0.0003  L_chart=1.6767  L_crit=0.2685  L_bnd=0.4643  lr=0.0010  dt=18.43s
        recon=0.1642  vq=0.2587  code_H=1.3209  code_px=3.7772  ch_usage=0.0422  rtr_mrg=0.0030  enc_gn=5.0114
        ctrl=0.0001  tex=0.0284  im_rew=0.0404  im_ret=0.5849  value=0.0226  wm_gn=0.1591
        z_norm=0.6559  z_max=0.9791  jump=0.0000  cons=0.0732  sol=1.0040  e_var=0.0007  ch_ent=2.4603  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5849  dret=0.5656  term=0.0225  bnd=0.0342  chart_acc=0.4588  chart_ent=1.7413  rw_drift=0.0000
        v_err=1.4204  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5579
        bnd_x=0.0374  bell=0.0393  bell_s=0.0223  rtg_e=1.4204  rtg_b=-1.4204  cal_e=1.4204  u_l2=0.0174  cov_n=0.0013
        col=11.1594  smp=0.0035  enc_t=0.2538  bnd_t=0.0396  wm_t=1.0228  crt_t=0.0086  diag_t=0.5752
        charts: 16/16 active  c0=0.17 c1=0.13 c2=0.03 c3=0.16 c4=0.02 c5=0.07 c6=0.03 c7=0.03 c8=0.01 c9=0.04 c10=0.04 c11=0.05 c12=0.04 c13=0.01 c14=0.13 c15=0.03
        symbols: 99/128 active  c0=6/8(H=1.30) c1=6/8(H=0.90) c2=6/8(H=1.17) c3=7/8(H=1.30) c4=7/8(H=1.76) c5=6/8(H=0.64) c6=5/8(H=1.34) c7=7/8(H=1.64) c8=5/8(H=0.90) c9=6/8(H=1.00) c10=5/8(H=1.44) c11=4/8(H=1.05) c12=8/8(H=1.33) c13=6/8(H=1.46) c14=8/8(H=1.50) c15=7/8(H=1.36)
E0194 [5upd]  ep_rew=14.1889  rew_20=13.2497  L_geo=0.5901  L_rew=0.0004  L_chart=1.5979  L_crit=0.2685  L_bnd=0.4785  lr=0.0010  dt=18.55s
        recon=0.1557  vq=0.2721  code_H=1.3050  code_px=3.6920  ch_usage=0.0433  rtr_mrg=0.0024  enc_gn=7.7391
        ctrl=0.0001  tex=0.0278  im_rew=0.0320  im_ret=0.4694  value=0.0257  wm_gn=0.2372
        z_norm=0.6549  z_max=0.9670  jump=0.0000  cons=0.0688  sol=1.0088  e_var=0.0285  ch_ent=2.5789  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4694  dret=0.4477  term=0.0253  bnd=0.0487  chart_acc=0.5113  chart_ent=1.6934  rw_drift=0.0000
        v_err=1.1645  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6664
        bnd_x=0.0542  bell=0.0317  bell_s=0.0129  rtg_e=1.1645  rtg_b=-1.1644  cal_e=1.1644  u_l2=0.0194  cov_n=0.0017
        col=11.1107  smp=0.0041  enc_t=0.2543  bnd_t=0.0400  wm_t=1.0583  crt_t=0.0087  diag_t=0.5685
        charts: 16/16 active  c0=0.14 c1=0.09 c2=0.09 c3=0.06 c4=0.01 c5=0.10 c6=0.11 c7=0.03 c8=0.02 c9=0.06 c10=0.06 c11=0.10 c12=0.01 c13=0.05 c14=0.03 c15=0.04
        symbols: 82/128 active  c0=6/8(H=0.91) c1=5/8(H=1.18) c2=5/8(H=1.26) c3=6/8(H=1.40) c4=5/8(H=1.42) c5=5/8(H=0.47) c6=5/8(H=1.19) c7=4/8(H=1.22) c8=6/8(H=0.98) c9=4/8(H=0.82) c10=6/8(H=1.18) c11=6/8(H=1.19) c12=3/8(H=0.98) c13=6/8(H=1.51) c14=4/8(H=0.83) c15=6/8(H=1.23)
E0195 [5upd]  ep_rew=14.4867  rew_20=13.0034  L_geo=0.6229  L_rew=0.0003  L_chart=1.6200  L_crit=0.3089  L_bnd=0.4350  lr=0.0010  dt=18.42s
        recon=0.1716  vq=0.2680  code_H=1.2877  code_px=3.6417  ch_usage=0.0983  rtr_mrg=0.0024  enc_gn=6.4890
        ctrl=0.0001  tex=0.0292  im_rew=0.0386  im_ret=0.5576  value=0.0204  wm_gn=0.2311
        z_norm=0.6451  z_max=0.9509  jump=0.0000  cons=0.0929  sol=0.9947  e_var=0.0003  ch_ent=2.5838  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5576  dret=0.5397  term=0.0208  bnd=0.0332  chart_acc=0.5133  chart_ent=1.6666  rw_drift=0.0000
        v_err=1.3646  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7789
        bnd_x=0.0377  bell=0.0368  bell_s=0.0149  rtg_e=1.3646  rtg_b=-1.3646  cal_e=1.3646  u_l2=0.0376  cov_n=0.0018
        col=11.1434  smp=0.0043  enc_t=0.2545  bnd_t=0.0400  wm_t=1.0235  crt_t=0.0087  diag_t=0.5693
        charts: 16/16 active  c0=0.18 c1=0.10 c2=0.04 c3=0.08 c4=0.09 c5=0.12 c6=0.04 c7=0.02 c8=0.03 c9=0.07 c10=0.03 c11=0.02 c12=0.03 c13=0.04 c14=0.07 c15=0.05
        symbols: 81/128 active  c0=4/8(H=1.17) c1=6/8(H=1.50) c2=6/8(H=0.87) c3=6/8(H=1.39) c4=6/8(H=1.12) c5=5/8(H=0.97) c6=4/8(H=1.36) c7=5/8(H=1.05) c8=4/8(H=0.56) c9=5/8(H=0.90) c10=5/8(H=1.25) c11=4/8(H=1.15) c12=5/8(H=1.12) c13=7/8(H=1.55) c14=5/8(H=1.30) c15=4/8(H=1.19)
E0196 [5upd]  ep_rew=14.2962  rew_20=13.3576  L_geo=0.7022  L_rew=0.0005  L_chart=1.8823  L_crit=0.2730  L_bnd=0.4609  lr=0.0010  dt=18.46s
        recon=0.1944  vq=0.2395  code_H=1.3765  code_px=3.9682  ch_usage=0.0423  rtr_mrg=0.0044  enc_gn=7.3199
        ctrl=0.0001  tex=0.0278  im_rew=0.0404  im_ret=0.5758  value=0.0120  wm_gn=0.2286
        z_norm=0.6449  z_max=0.9663  jump=0.0000  cons=0.0817  sol=0.9862  e_var=0.0012  ch_ent=2.6341  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5758  dret=0.5660  term=0.0114  bnd=0.0174  chart_acc=0.4296  chart_ent=1.7052  rw_drift=0.0000
        v_err=1.3056  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8085
        bnd_x=0.0205  bell=0.0376  bell_s=0.0388  rtg_e=1.3056  rtg_b=-1.3056  cal_e=1.3056  u_l2=0.0534  cov_n=0.0012
        col=11.1920  smp=0.0051  enc_t=0.2531  bnd_t=0.0401  wm_t=1.0207  crt_t=0.0086  diag_t=0.5812
        charts: 16/16 active  c0=0.13 c1=0.11 c2=0.09 c3=0.09 c4=0.04 c5=0.06 c6=0.03 c7=0.06 c8=0.02 c9=0.09 c10=0.05 c11=0.02 c12=0.05 c13=0.04 c14=0.09 c15=0.04
        symbols: 90/128 active  c0=6/8(H=1.41) c1=6/8(H=1.32) c2=7/8(H=1.54) c3=7/8(H=1.11) c4=6/8(H=1.61) c5=4/8(H=0.74) c6=4/8(H=1.33) c7=8/8(H=1.69) c8=4/8(H=0.98) c9=6/8(H=1.16) c10=5/8(H=1.11) c11=4/8(H=1.09) c12=6/8(H=1.13) c13=6/8(H=0.84) c14=6/8(H=1.41) c15=5/8(H=1.23)
E0197 [5upd]  ep_rew=16.8320  rew_20=14.0479  L_geo=0.6337  L_rew=0.0004  L_chart=1.7306  L_crit=0.2954  L_bnd=0.4174  lr=0.0010  dt=18.45s
        recon=0.1909  vq=0.2645  code_H=1.3756  code_px=3.9854  ch_usage=0.0362  rtr_mrg=0.0025  enc_gn=5.6694
        ctrl=0.0001  tex=0.0307  im_rew=0.0416  im_ret=0.5873  value=0.0059  wm_gn=0.2164
        z_norm=0.6268  z_max=0.9259  jump=0.0000  cons=0.0865  sol=0.9997  e_var=0.0010  ch_ent=2.4811  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5873  dret=0.5821  term=0.0060  bnd=0.0089  chart_acc=0.4608  chart_ent=1.6805  rw_drift=0.0000
        v_err=1.3851  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8236
        bnd_x=0.0105  bell=0.0400  bell_s=0.0296  rtg_e=1.3851  rtg_b=-1.3851  cal_e=1.3851  u_l2=0.0639  cov_n=0.0007
        col=11.1774  smp=0.0040  enc_t=0.2539  bnd_t=0.0401  wm_t=1.0225  crt_t=0.0086  diag_t=0.5693
        charts: 16/16 active  c0=0.21 c1=0.17 c2=0.05 c3=0.07 c4=0.09 c5=0.03 c6=0.04 c7=0.06 c8=0.01 c9=0.05 c10=0.05 c11=0.03 c12=0.01 c13=0.02 c14=0.06 c15=0.04
        symbols: 99/128 active  c0=7/8(H=1.65) c1=6/8(H=1.40) c2=7/8(H=1.37) c3=6/8(H=0.37) c4=8/8(H=1.58) c5=5/8(H=1.06) c6=4/8(H=1.26) c7=6/8(H=0.81) c8=4/8(H=1.14) c9=7/8(H=1.38) c10=7/8(H=1.38) c11=4/8(H=0.93) c12=7/8(H=1.34) c13=8/8(H=1.55) c14=7/8(H=0.90) c15=6/8(H=0.97)
E0198 [5upd]  ep_rew=14.7412  rew_20=13.5167  L_geo=0.6031  L_rew=0.0004  L_chart=1.7130  L_crit=0.2787  L_bnd=0.3778  lr=0.0010  dt=18.32s
        recon=0.1870  vq=0.2738  code_H=1.2749  code_px=3.6049  ch_usage=0.0633  rtr_mrg=0.0024  enc_gn=12.5108
        ctrl=0.0001  tex=0.0287  im_rew=0.0368  im_ret=0.5225  value=0.0094  wm_gn=0.1855
        z_norm=0.6464  z_max=0.9554  jump=0.0000  cons=0.1030  sol=0.9939  e_var=0.0004  ch_ent=2.6977  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5225  dret=0.5146  term=0.0092  bnd=0.0154  chart_acc=0.4692  chart_ent=1.7753  rw_drift=0.0000
        v_err=1.3120  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9156
        bnd_x=0.0177  bell=0.0391  bell_s=0.0395  rtg_e=1.3120  rtg_b=-1.3120  cal_e=1.3120  u_l2=0.0693  cov_n=0.0012
        col=11.0761  smp=0.0046  enc_t=0.2542  bnd_t=0.0397  wm_t=1.0174  crt_t=0.0085  diag_t=0.5758
        charts: 16/16 active  c0=0.07 c1=0.09 c2=0.08 c3=0.05 c4=0.06 c5=0.06 c6=0.06 c7=0.11 c8=0.02 c9=0.07 c10=0.09 c11=0.02 c12=0.06 c13=0.04 c14=0.07 c15=0.07
        symbols: 93/128 active  c0=7/8(H=1.35) c1=5/8(H=1.36) c2=5/8(H=0.98) c3=6/8(H=0.45) c4=7/8(H=1.72) c5=5/8(H=1.35) c6=6/8(H=1.02) c7=6/8(H=1.25) c8=5/8(H=1.14) c9=7/8(H=1.50) c10=6/8(H=1.71) c11=5/8(H=1.13) c12=8/8(H=1.94) c13=5/8(H=1.26) c14=5/8(H=1.22) c15=5/8(H=1.13)
E0199 [5upd]  ep_rew=6.7076  rew_20=13.4115  L_geo=0.6040  L_rew=0.0009  L_chart=1.6776  L_crit=0.3109  L_bnd=0.4281  lr=0.0010  dt=18.49s
        recon=0.1890  vq=0.2594  code_H=1.3715  code_px=3.9503  ch_usage=0.1034  rtr_mrg=0.0022  enc_gn=6.4057
        ctrl=0.0001  tex=0.0283  im_rew=0.0470  im_ret=0.6656  value=0.0096  wm_gn=0.2771
        z_norm=0.6384  z_max=0.9630  jump=0.0000  cons=0.0764  sol=0.9964  e_var=0.0006  ch_ent=2.6415  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6656  dret=0.6574  term=0.0096  bnd=0.0125  chart_acc=0.5075  chart_ent=1.6933  rw_drift=0.0000
        v_err=1.5334  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.9371
        bnd_x=0.0151  bell=0.0428  bell_s=0.0328  rtg_e=1.5334  rtg_b=-1.5334  cal_e=1.5334  u_l2=0.0711  cov_n=0.0010
        col=11.1086  smp=0.0038  enc_t=0.2530  bnd_t=0.0399  wm_t=1.0479  crt_t=0.0085  diag_t=0.5683
        charts: 16/16 active  c0=0.17 c1=0.09 c2=0.06 c3=0.07 c4=0.07 c5=0.08 c6=0.02 c7=0.03 c8=0.08 c9=0.02 c10=0.05 c11=0.04 c12=0.03 c13=0.05 c14=0.07 c15=0.06
        symbols: 98/128 active  c0=7/8(H=1.42) c1=7/8(H=1.56) c2=6/8(H=1.28) c3=5/8(H=1.21) c4=6/8(H=0.86) c5=7/8(H=1.10) c6=4/8(H=1.03) c7=6/8(H=1.34) c8=7/8(H=0.81) c9=5/8(H=1.13) c10=6/8(H=1.35) c11=5/8(H=1.16) c12=8/8(H=1.25) c13=8/8(H=1.50) c14=5/8(H=1.05) c15=6/8(H=1.30)
E0200 [5upd]  ep_rew=13.9361  rew_20=13.3783  L_geo=0.6417  L_rew=0.0004  L_chart=1.7338  L_crit=0.3333  L_bnd=0.4755  lr=0.0010  dt=18.27s
        recon=0.2038  vq=0.2566  code_H=1.3108  code_px=3.7239  ch_usage=0.0712  rtr_mrg=0.0027  enc_gn=6.6257
        ctrl=0.0001  tex=0.0283  im_rew=0.0474  im_ret=0.6749  value=0.0129  wm_gn=0.1859
        z_norm=0.6337  z_max=0.9724  jump=0.0000  cons=0.0909  sol=0.9962  e_var=0.0005  ch_ent=2.6293  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6749  dret=0.6637  term=0.0131  bnd=0.0169  chart_acc=0.4679  chart_ent=1.7702  rw_drift=0.0000
        v_err=1.6395  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8067
        bnd_x=0.0206  bell=0.0466  bell_s=0.0367  rtg_e=1.6395  rtg_b=-1.6395  cal_e=1.6395  u_l2=0.0644  cov_n=0.0010
        col=11.0425  smp=0.0043  enc_t=0.2529  bnd_t=0.0395  wm_t=1.0171  crt_t=0.0086  diag_t=0.5656
        charts: 16/16 active  c0=0.14 c1=0.12 c2=0.04 c3=0.03 c4=0.13 c5=0.03 c6=0.06 c7=0.06 c8=0.05 c9=0.04 c10=0.09 c11=0.06 c12=0.03 c13=0.03 c14=0.05 c15=0.04
        symbols: 91/128 active  c0=5/8(H=1.23) c1=7/8(H=1.49) c2=5/8(H=1.34) c3=5/8(H=1.26) c4=8/8(H=1.69) c5=5/8(H=1.38) c6=7/8(H=1.47) c7=6/8(H=1.18) c8=6/8(H=0.72) c9=4/8(H=1.28) c10=6/8(H=1.48) c11=6/8(H=1.36) c12=5/8(H=1.33) c13=5/8(H=1.25) c14=6/8(H=1.25) c15=5/8(H=1.52)
E0201 [5upd]  ep_rew=15.4263  rew_20=13.9629  L_geo=0.5833  L_rew=0.0006  L_chart=1.5642  L_crit=0.2738  L_bnd=0.4533  lr=0.0010  dt=19.09s
        recon=0.1553  vq=0.2730  code_H=1.2940  code_px=3.6610  ch_usage=0.0474  rtr_mrg=0.0023  enc_gn=4.9914
        ctrl=0.0001  tex=0.0265  im_rew=0.0344  im_ret=0.4971  value=0.0191  wm_gn=0.1983
        z_norm=0.6581  z_max=0.9848  jump=0.0000  cons=0.0829  sol=0.9931  e_var=0.0005  ch_ent=2.5728  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4971  dret=0.4807  term=0.0190  bnd=0.0340  chart_acc=0.4929  chart_ent=1.6593  rw_drift=0.0000
        v_err=1.2584  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6012
        bnd_x=0.0389  bell=0.0342  bell_s=0.0133  rtg_e=1.2584  rtg_b=-1.2584  cal_e=1.2584  u_l2=0.0544  cov_n=0.0014
        col=11.8486  smp=0.0050  enc_t=0.2539  bnd_t=0.0397  wm_t=1.0174  crt_t=0.0086  diag_t=0.5739
        charts: 16/16 active  c0=0.17 c1=0.10 c2=0.04 c3=0.09 c4=0.07 c5=0.12 c6=0.05 c7=0.02 c8=0.04 c9=0.06 c10=0.04 c11=0.03 c12=0.01 c13=0.04 c14=0.09 c15=0.03
        symbols: 85/128 active  c0=6/8(H=1.33) c1=4/8(H=0.97) c2=6/8(H=0.75) c3=6/8(H=1.57) c4=6/8(H=0.91) c5=8/8(H=1.39) c6=4/8(H=1.16) c7=4/8(H=0.48) c8=6/8(H=1.04) c9=4/8(H=1.18) c10=5/8(H=1.29) c11=6/8(H=1.33) c12=5/8(H=1.04) c13=6/8(H=1.41) c14=6/8(H=1.34) c15=3/8(H=0.90)
E0202 [5upd]  ep_rew=14.4045  rew_20=13.6630  L_geo=0.6365  L_rew=0.0011  L_chart=1.7144  L_crit=0.3020  L_bnd=0.5072  lr=0.0010  dt=18.49s
        recon=0.1639  vq=0.2878  code_H=1.3320  code_px=3.7931  ch_usage=0.0553  rtr_mrg=0.0044  enc_gn=6.9755
        ctrl=0.0001  tex=0.0275  im_rew=0.0408  im_ret=0.5897  value=0.0226  wm_gn=0.2352
        z_norm=0.6676  z_max=0.9599  jump=0.0000  cons=0.0816  sol=1.0009  e_var=0.0005  ch_ent=2.6268  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5897  dret=0.5706  term=0.0222  bnd=0.0335  chart_acc=0.4429  chart_ent=1.6660  rw_drift=0.0000
        v_err=1.3535  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.3494
        bnd_x=0.0410  bell=0.0396  bell_s=0.0541  rtg_e=1.3535  rtg_b=-1.3534  cal_e=1.3534  u_l2=0.0475  cov_n=0.0016
        col=11.2179  smp=0.0052  enc_t=0.2548  bnd_t=0.0398  wm_t=1.0244  crt_t=0.0086  diag_t=0.5707
        charts: 16/16 active  c0=0.08 c1=0.03 c2=0.09 c3=0.09 c4=0.06 c5=0.10 c6=0.04 c7=0.03 c8=0.03 c9=0.02 c10=0.05 c11=0.11 c12=0.02 c13=0.06 c14=0.14 c15=0.06
        symbols: 94/128 active  c0=6/8(H=1.32) c1=5/8(H=1.11) c2=6/8(H=1.03) c3=6/8(H=1.28) c4=7/8(H=1.15) c5=7/8(H=1.04) c6=5/8(H=1.37) c7=8/8(H=1.41) c8=6/8(H=1.26) c9=4/8(H=1.09) c10=5/8(H=1.26) c11=4/8(H=1.17) c12=6/8(H=1.47) c13=8/8(H=1.42) c14=6/8(H=1.64) c15=5/8(H=1.13)
E0203 [5upd]  ep_rew=24.2263  rew_20=14.3394  L_geo=0.5978  L_rew=0.0003  L_chart=1.6514  L_crit=0.2978  L_bnd=0.6408  lr=0.0010  dt=18.31s
        recon=0.1697  vq=0.2880  code_H=1.2902  code_px=3.6485  ch_usage=0.0334  rtr_mrg=0.0026  enc_gn=7.3046
        ctrl=0.0002  tex=0.0300  im_rew=0.0415  im_ret=0.5981  value=0.0200  wm_gn=0.2746
        z_norm=0.6325  z_max=0.9499  jump=0.0000  cons=0.0852  sol=0.9999  e_var=0.0003  ch_ent=2.7064  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5981  dret=0.5801  term=0.0209  bnd=0.0310  chart_acc=0.4471  chart_ent=1.6642  rw_drift=0.0000
        v_err=1.5225  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0630
        bnd_x=0.0365  bell=0.0411  bell_s=0.0227  rtg_e=1.5225  rtg_b=-1.5225  cal_e=1.5225  u_l2=0.0464  cov_n=0.0018
        col=11.0729  smp=0.0032  enc_t=0.2534  bnd_t=0.0398  wm_t=1.0162  crt_t=0.0086  diag_t=0.5681
        charts: 16/16 active  c0=0.05 c1=0.07 c2=0.07 c3=0.12 c4=0.07 c5=0.05 c6=0.08 c7=0.02 c8=0.08 c9=0.07 c10=0.05 c11=0.07 c12=0.04 c13=0.04 c14=0.04 c15=0.09
        symbols: 101/128 active  c0=5/8(H=0.40) c1=7/8(H=1.58) c2=6/8(H=1.37) c3=7/8(H=1.21) c4=7/8(H=1.41) c5=6/8(H=1.24) c6=4/8(H=1.14) c7=6/8(H=1.26) c8=5/8(H=1.22) c9=5/8(H=0.41) c10=7/8(H=1.48) c11=5/8(H=1.40) c12=8/8(H=1.45) c13=8/8(H=1.36) c14=7/8(H=1.46) c15=8/8(H=1.49)
E0204 [5upd]  ep_rew=15.4133  rew_20=14.7858  L_geo=0.5399  L_rew=0.0003  L_chart=1.5434  L_crit=0.3317  L_bnd=0.6146  lr=0.0010  dt=18.50s
        recon=0.1621  vq=0.2620  code_H=1.2982  code_px=3.6637  ch_usage=0.0775  rtr_mrg=0.0031  enc_gn=7.0341
        ctrl=0.0001  tex=0.0318  im_rew=0.0416  im_ret=0.5908  value=0.0110  wm_gn=0.2440
        z_norm=0.6058  z_max=0.9392  jump=0.0000  cons=0.0966  sol=0.9967  e_var=0.0002  ch_ent=2.6852  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5908  dret=0.5816  term=0.0107  bnd=0.0158  chart_acc=0.5058  chart_ent=1.6834  rw_drift=0.0000
        v_err=1.4695  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.1234
        bnd_x=0.0189  bell=0.0399  bell_s=0.0177  rtg_e=1.4695  rtg_b=-1.4695  cal_e=1.4695  u_l2=0.0523  cov_n=0.0009
        col=11.0766  smp=0.0034  enc_t=0.2539  bnd_t=0.0405  wm_t=1.0524  crt_t=0.0087  diag_t=0.5748
        charts: 16/16 active  c0=0.12 c1=0.04 c2=0.07 c3=0.09 c4=0.09 c5=0.03 c6=0.07 c7=0.03 c8=0.06 c9=0.07 c10=0.04 c11=0.08 c12=0.07 c13=0.03 c14=0.05 c15=0.05
        symbols: 79/128 active  c0=6/8(H=1.19) c1=5/8(H=1.29) c2=6/8(H=1.23) c3=4/8(H=1.08) c4=8/8(H=1.20) c5=5/8(H=0.80) c6=5/8(H=1.24) c7=6/8(H=1.28) c8=4/8(H=0.29) c9=5/8(H=1.02) c10=3/8(H=0.34) c11=5/8(H=0.82) c12=4/8(H=0.69) c13=5/8(H=1.33) c14=5/8(H=1.44) c15=3/8(H=0.81)
E0205 [5upd]  ep_rew=13.6228  rew_20=14.3250  L_geo=0.6724  L_rew=0.0006  L_chart=1.8346  L_crit=0.2957  L_bnd=0.5473  lr=0.0010  dt=18.39s
        recon=0.1733  vq=0.2345  code_H=1.3581  code_px=3.8986  ch_usage=0.0309  rtr_mrg=0.0066  enc_gn=8.0174
        ctrl=0.0000  tex=0.0293  im_rew=0.0361  im_ret=0.5099  value=0.0049  wm_gn=0.2313
        z_norm=0.6213  z_max=0.9069  jump=0.0000  cons=0.0853  sol=0.9998  e_var=0.0003  ch_ent=2.7105  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5099  dret=0.5057  term=0.0049  bnd=0.0083  chart_acc=0.4288  chart_ent=1.7292  rw_drift=0.0000
        v_err=1.4775  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4113
        bnd_x=0.0094  bell=0.0400  bell_s=0.0259  rtg_e=1.4775  rtg_b=-1.4775  cal_e=1.4775  u_l2=0.0634  cov_n=0.0004
        col=11.1073  smp=0.0040  enc_t=0.2529  bnd_t=0.0399  wm_t=1.0201  crt_t=0.0086  diag_t=0.5941
        charts: 16/16 active  c0=0.03 c1=0.08 c2=0.07 c3=0.09 c4=0.09 c5=0.05 c6=0.08 c7=0.05 c8=0.03 c9=0.09 c10=0.05 c11=0.08 c12=0.04 c13=0.05 c14=0.08 c15=0.04
        symbols: 94/128 active  c0=3/8(H=0.93) c1=5/8(H=1.18) c2=6/8(H=1.21) c3=6/8(H=0.98) c4=7/8(H=1.07) c5=6/8(H=1.40) c6=7/8(H=1.79) c7=8/8(H=1.37) c8=5/8(H=0.87) c9=5/8(H=1.03) c10=5/8(H=1.39) c11=5/8(H=1.32) c12=7/8(H=1.21) c13=7/8(H=1.49) c14=6/8(H=1.53) c15=6/8(H=1.54)
E0206 [5upd]  ep_rew=14.4440  rew_20=14.2428  L_geo=0.5989  L_rew=0.0003  L_chart=1.7747  L_crit=0.2931  L_bnd=0.4400  lr=0.0010  dt=18.81s
        recon=0.1670  vq=0.2513  code_H=1.3447  code_px=3.8405  ch_usage=0.0811  rtr_mrg=0.0038  enc_gn=6.8947
        ctrl=0.0001  tex=0.0294  im_rew=0.0435  im_ret=0.6178  value=0.0109  wm_gn=0.2030
        z_norm=0.6281  z_max=0.9533  jump=0.0000  cons=0.0841  sol=0.9976  e_var=0.0003  ch_ent=2.6817  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6178  dret=0.6087  term=0.0106  bnd=0.0149  chart_acc=0.4279  chart_ent=1.7499  rw_drift=0.0000
        v_err=1.3624  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6530
        bnd_x=0.0172  bell=0.0385  bell_s=0.0345  rtg_e=1.3624  rtg_b=-1.3624  cal_e=1.3624  u_l2=0.0781  cov_n=0.0009
        col=11.6132  smp=0.0038  enc_t=0.2531  bnd_t=0.0396  wm_t=1.0110  crt_t=0.0085  diag_t=0.5650
        charts: 16/16 active  c0=0.06 c1=0.02 c2=0.04 c3=0.10 c4=0.07 c5=0.10 c6=0.05 c7=0.04 c8=0.05 c9=0.08 c10=0.04 c11=0.09 c12=0.08 c13=0.10 c14=0.03 c15=0.05
        symbols: 94/128 active  c0=6/8(H=1.55) c1=4/8(H=1.29) c2=7/8(H=1.59) c3=6/8(H=1.33) c4=6/8(H=1.29) c5=7/8(H=1.03) c6=6/8(H=1.48) c7=7/8(H=1.70) c8=6/8(H=1.00) c9=5/8(H=1.04) c10=5/8(H=1.40) c11=5/8(H=1.27) c12=7/8(H=1.59) c13=6/8(H=0.57) c14=5/8(H=1.37) c15=6/8(H=1.43)
E0207 [5upd]  ep_rew=14.0215  rew_20=13.8894  L_geo=0.6202  L_rew=0.0002  L_chart=1.7391  L_crit=0.2557  L_bnd=0.4490  lr=0.0010  dt=18.20s
        recon=0.1548  vq=0.2508  code_H=1.2734  code_px=3.5989  ch_usage=0.0280  rtr_mrg=0.0035  enc_gn=8.2926
        ctrl=0.0001  tex=0.0313  im_rew=0.0407  im_ret=0.5858  value=0.0186  wm_gn=0.1689
        z_norm=0.6171  z_max=0.8861  jump=0.0000  cons=0.0947  sol=0.9975  e_var=0.0002  ch_ent=2.6909  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5858  dret=0.5701  term=0.0183  bnd=0.0275  chart_acc=0.4542  chart_ent=1.7461  rw_drift=0.0000
        v_err=1.5660  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7433
        bnd_x=0.0295  bell=0.0429  bell_s=0.0174  rtg_e=1.5660  rtg_b=-1.5660  cal_e=1.5660  u_l2=0.1118  cov_n=0.0013
        col=11.0019  smp=0.0031  enc_t=0.2533  bnd_t=0.0397  wm_t=1.0120  crt_t=0.0085  diag_t=0.5630
        charts: 16/16 active  c0=0.08 c1=0.07 c2=0.04 c3=0.10 c4=0.07 c5=0.04 c6=0.05 c7=0.03 c8=0.09 c9=0.05 c10=0.04 c11=0.10 c12=0.10 c13=0.03 c14=0.06 c15=0.04
        symbols: 90/128 active  c0=5/8(H=0.90) c1=4/8(H=0.98) c2=6/8(H=1.45) c3=4/8(H=0.78) c4=6/8(H=1.47) c5=6/8(H=1.10) c6=5/8(H=1.22) c7=8/8(H=1.85) c8=6/8(H=1.32) c9=5/8(H=0.76) c10=5/8(H=1.24) c11=7/8(H=1.49) c12=7/8(H=1.28) c13=5/8(H=0.90) c14=6/8(H=1.20) c15=5/8(H=1.11)
E0208 [5upd]  ep_rew=12.8686  rew_20=13.8624  L_geo=0.6309  L_rew=0.0004  L_chart=1.8515  L_crit=0.2807  L_bnd=0.4520  lr=0.0010  dt=18.32s
        recon=0.1844  vq=0.2467  code_H=1.3348  code_px=3.8107  ch_usage=0.0379  rtr_mrg=0.0035  enc_gn=6.7442
        ctrl=0.0001  tex=0.0299  im_rew=0.0380  im_ret=0.5506  value=0.0214  wm_gn=0.1915
        z_norm=0.6223  z_max=0.9282  jump=0.0000  cons=0.0758  sol=0.9955  e_var=0.0004  ch_ent=2.5682  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5506  dret=0.5324  term=0.0212  bnd=0.0343  chart_acc=0.3979  chart_ent=1.8072  rw_drift=0.0000
        v_err=1.4537  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7709
        bnd_x=0.0399  bell=0.0395  bell_s=0.0143  rtg_e=1.4537  rtg_b=-1.4537  cal_e=1.4537  u_l2=0.1381  cov_n=0.0011
        col=11.0712  smp=0.0029  enc_t=0.2536  bnd_t=0.0393  wm_t=1.0201  crt_t=0.0085  diag_t=0.5705
        charts: 16/16 active  c0=0.08 c1=0.08 c2=0.08 c3=0.13 c4=0.08 c5=0.11 c6=0.03 c7=0.04 c8=0.08 c9=0.01 c10=0.02 c11=0.13 c12=0.02 c13=0.02 c14=0.06 c15=0.03
        symbols: 84/128 active  c0=4/8(H=1.05) c1=5/8(H=1.38) c2=5/8(H=1.33) c3=6/8(H=1.05) c4=6/8(H=1.32) c5=6/8(H=0.94) c6=4/8(H=1.17) c7=6/8(H=0.94) c8=7/8(H=1.04) c9=4/8(H=1.30) c10=7/8(H=1.41) c11=4/8(H=0.93) c12=5/8(H=1.21) c13=5/8(H=1.34) c14=4/8(H=1.09) c15=6/8(H=0.57)
E0209 [5upd]  ep_rew=9.9620  rew_20=13.4494  L_geo=0.5621  L_rew=0.0004  L_chart=1.6651  L_crit=0.2620  L_bnd=0.5056  lr=0.0010  dt=18.48s
        recon=0.1639  vq=0.2495  code_H=1.3077  code_px=3.7209  ch_usage=0.1301  rtr_mrg=0.0046  enc_gn=5.7972
        ctrl=0.0001  tex=0.0304  im_rew=0.0432  im_ret=0.6219  value=0.0204  wm_gn=0.2267
        z_norm=0.6230  z_max=0.9063  jump=0.0000  cons=0.0875  sol=0.9993  e_var=0.0003  ch_ent=2.6247  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6219  dret=0.6048  term=0.0199  bnd=0.0283  chart_acc=0.4900  chart_ent=1.7243  rw_drift=0.0000
        v_err=1.4449  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5228
        bnd_x=0.0333  bell=0.0394  bell_s=0.0205  rtg_e=1.4449  rtg_b=-1.4449  cal_e=1.4449  u_l2=0.1433  cov_n=0.0012
        col=11.0804  smp=0.0045  enc_t=0.2833  bnd_t=0.0399  wm_t=1.0195  crt_t=0.0086  diag_t=0.5773
        charts: 16/16 active  c0=0.08 c1=0.05 c2=0.05 c3=0.09 c4=0.11 c5=0.07 c6=0.07 c7=0.06 c8=0.04 c9=0.03 c10=0.02 c11=0.17 c12=0.06 c13=0.02 c14=0.06 c15=0.03
        symbols: 82/128 active  c0=4/8(H=1.06) c1=4/8(H=1.20) c2=6/8(H=1.19) c3=6/8(H=1.21) c4=8/8(H=1.31) c5=6/8(H=0.59) c6=5/8(H=1.08) c7=7/8(H=1.62) c8=5/8(H=0.81) c9=5/8(H=1.18) c10=4/8(H=1.04) c11=5/8(H=1.02) c12=5/8(H=1.32) c13=4/8(H=0.60) c14=4/8(H=1.17) c15=4/8(H=1.25)
E0210 [5upd]  ep_rew=14.5218  rew_20=14.1282  L_geo=0.5793  L_rew=0.0004  L_chart=1.6292  L_crit=0.2906  L_bnd=0.5638  lr=0.0010  dt=18.23s
        recon=0.1754  vq=0.2653  code_H=1.3096  code_px=3.7243  ch_usage=0.0724  rtr_mrg=0.0037  enc_gn=6.8220
        ctrl=0.0001  tex=0.0295  im_rew=0.0447  im_ret=0.6414  value=0.0187  wm_gn=0.2292
        z_norm=0.6349  z_max=0.9145  jump=0.0000  cons=0.1253  sol=0.9925  e_var=0.0004  ch_ent=2.6509  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6414  dret=0.6257  term=0.0182  bnd=0.0250  chart_acc=0.5292  chart_ent=1.7134  rw_drift=0.0000
        v_err=1.4416  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2711
        bnd_x=0.0311  bell=0.0432  bell_s=0.0446  rtg_e=1.4416  rtg_b=-1.4416  cal_e=1.4416  u_l2=0.1086  cov_n=0.0009
        col=11.0319  smp=0.0043  enc_t=0.2532  bnd_t=0.0398  wm_t=1.0118  crt_t=0.0085  diag_t=0.5626
        charts: 16/16 active  c0=0.06 c1=0.04 c2=0.04 c3=0.03 c4=0.10 c5=0.12 c6=0.02 c7=0.05 c8=0.10 c9=0.08 c10=0.03 c11=0.10 c12=0.05 c13=0.05 c14=0.10 c15=0.04
        symbols: 98/128 active  c0=7/8(H=1.40) c1=5/8(H=1.23) c2=7/8(H=1.16) c3=5/8(H=0.80) c4=7/8(H=1.72) c5=7/8(H=0.90) c6=5/8(H=1.56) c7=7/8(H=1.46) c8=8/8(H=1.48) c9=5/8(H=0.64) c10=5/8(H=1.50) c11=5/8(H=0.66) c12=7/8(H=1.77) c13=6/8(H=1.46) c14=6/8(H=1.34) c15=6/8(H=1.55)
  EVAL  reward=11.4 +/- 1.4  len=300
E0211 [5upd]  ep_rew=14.8310  rew_20=14.9501  L_geo=0.5818  L_rew=0.0004  L_chart=1.6835  L_crit=0.2991  L_bnd=0.6384  lr=0.0010  dt=17.94s
        recon=0.1841  vq=0.2525  code_H=1.3981  code_px=4.0496  ch_usage=0.0284  rtr_mrg=0.0043  enc_gn=5.9085
        ctrl=0.0001  tex=0.0296  im_rew=0.0422  im_ret=0.6056  value=0.0183  wm_gn=0.1770
        z_norm=0.6328  z_max=0.9332  jump=0.0000  cons=0.0975  sol=0.9937  e_var=0.0004  ch_ent=2.6759  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6056  dret=0.5902  term=0.0179  bnd=0.0260  chart_acc=0.4983  chart_ent=1.6662  rw_drift=0.0000
        v_err=1.4750  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.0308
        bnd_x=0.0329  bell=0.0408  bell_s=0.0269  rtg_e=1.4750  rtg_b=-1.4750  cal_e=1.4750  u_l2=0.0790  cov_n=0.0012
        col=10.7133  smp=0.0022  enc_t=0.2544  bnd_t=0.0402  wm_t=1.0136  crt_t=0.0086  diag_t=0.5767
        charts: 16/16 active  c0=0.07 c1=0.02 c2=0.05 c3=0.06 c4=0.06 c5=0.04 c6=0.06 c7=0.05 c8=0.12 c9=0.04 c10=0.07 c11=0.14 c12=0.03 c13=0.05 c14=0.08 c15=0.04
        symbols: 105/128 active  c0=8/8(H=1.55) c1=5/8(H=1.60) c2=5/8(H=1.10) c3=8/8(H=1.28) c4=7/8(H=1.25) c5=6/8(H=1.10) c6=7/8(H=1.54) c7=7/8(H=1.46) c8=8/8(H=1.05) c9=5/8(H=1.36) c10=7/8(H=1.51) c11=6/8(H=1.38) c12=7/8(H=1.43) c13=7/8(H=1.46) c14=6/8(H=1.14) c15=6/8(H=1.51)
E0212 [5upd]  ep_rew=13.8230  rew_20=14.7615  L_geo=0.5273  L_rew=0.0002  L_chart=1.5079  L_crit=0.2765  L_bnd=0.6780  lr=0.0010  dt=18.20s
        recon=0.1509  vq=0.2997  code_H=1.2288  code_px=3.4266  ch_usage=0.0848  rtr_mrg=0.0018  enc_gn=7.9605
        ctrl=0.0001  tex=0.0276  im_rew=0.0341  im_ret=0.4925  value=0.0174  wm_gn=0.1974
        z_norm=0.6620  z_max=0.9444  jump=0.0000  cons=0.0631  sol=0.9967  e_var=0.0004  ch_ent=2.5611  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4925  dret=0.4772  term=0.0178  bnd=0.0320  chart_acc=0.5188  chart_ent=1.5459  rw_drift=0.0000
        v_err=1.3528  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=-0.1933
        bnd_x=0.0383  bell=0.0369  bell_s=0.0262  rtg_e=1.3528  rtg_b=-1.3528  cal_e=1.3528  u_l2=0.0631  cov_n=0.0013
        col=10.9791  smp=0.0026  enc_t=0.2529  bnd_t=0.0394  wm_t=1.0169  crt_t=0.0085  diag_t=0.5625
        charts: 16/16 active  c0=0.18 c1=0.08 c2=0.07 c3=0.06 c4=0.05 c5=0.09 c6=0.02 c7=0.02 c8=0.09 c9=0.02 c10=0.05 c11=0.07 c12=0.01 c13=0.05 c14=0.12 c15=0.02
        symbols: 84/128 active  c0=7/8(H=1.47) c1=8/8(H=1.47) c2=5/8(H=1.37) c3=6/8(H=1.15) c4=5/8(H=1.02) c5=6/8(H=1.00) c6=6/8(H=1.66) c7=3/8(H=0.78) c8=5/8(H=0.80) c9=3/8(H=0.58) c10=6/8(H=1.13) c11=5/8(H=0.84) c12=5/8(H=1.40) c13=5/8(H=0.75) c14=5/8(H=1.09) c15=4/8(H=1.09)
E0213 [5upd]  ep_rew=13.8656  rew_20=14.7483  L_geo=0.5373  L_rew=0.0001  L_chart=1.6420  L_crit=0.2535  L_bnd=0.5599  lr=0.0010  dt=16.63s
        recon=0.1549  vq=0.2584  code_H=1.2832  code_px=3.6157  ch_usage=0.0336  rtr_mrg=0.0023  enc_gn=8.1440
        ctrl=0.0001  tex=0.0298  im_rew=0.0383  im_ret=0.5479  value=0.0135  wm_gn=0.2121
        z_norm=0.6288  z_max=0.9109  jump=0.0000  cons=0.0922  sol=0.9959  e_var=0.0003  ch_ent=2.6887  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5479  dret=0.5365  term=0.0132  bnd=0.0212  chart_acc=0.4538  chart_ent=1.6391  rw_drift=0.0000
        v_err=1.3706  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.2361
        bnd_x=0.0236  bell=0.0376  bell_s=0.0152  rtg_e=1.3706  rtg_b=-1.3706  cal_e=1.3706  u_l2=0.0527  cov_n=0.0010
        col=11.0372  smp=0.0034  enc_t=0.2348  bnd_t=0.0367  wm_t=0.7551  crt_t=0.0063  diag_t=0.3646
        charts: 16/16 active  c0=0.03 c1=0.08 c2=0.08 c3=0.07 c4=0.13 c5=0.04 c6=0.04 c7=0.06 c8=0.04 c9=0.10 c10=0.05 c11=0.07 c12=0.05 c13=0.03 c14=0.08 c15=0.04
        symbols: 99/128 active  c0=4/8(H=1.05) c1=6/8(H=1.37) c2=7/8(H=1.75) c3=7/8(H=1.14) c4=8/8(H=0.94) c5=6/8(H=1.16) c6=5/8(H=1.25) c7=7/8(H=1.49) c8=6/8(H=0.97) c9=5/8(H=0.78) c10=6/8(H=1.29) c11=6/8(H=1.38) c12=8/8(H=1.66) c13=5/8(H=1.25) c14=6/8(H=1.38) c15=7/8(H=1.51)
E0214 [5upd]  ep_rew=13.8943  rew_20=14.6511  L_geo=0.6508  L_rew=0.0002  L_chart=1.8656  L_crit=0.2817  L_bnd=0.4791  lr=0.0010  dt=12.66s
        recon=0.1823  vq=0.2435  code_H=1.2568  code_px=3.5220  ch_usage=0.0621  rtr_mrg=0.0013  enc_gn=7.0267
        ctrl=0.0001  tex=0.0306  im_rew=0.0336  im_ret=0.4771  value=0.0084  wm_gn=0.2358
        z_norm=0.5907  z_max=0.9330  jump=0.0000  cons=0.0825  sol=0.9972  e_var=0.0004  ch_ent=2.6797  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4771  dret=0.4702  term=0.0081  bnd=0.0148  chart_acc=0.4246  chart_ent=1.7069  rw_drift=0.0000
        v_err=1.3650  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6162
        bnd_x=0.0164  bell=0.0366  bell_s=0.0124  rtg_e=1.3650  rtg_b=-1.3650  cal_e=1.3650  u_l2=0.0580  cov_n=0.0007
        col=7.2201  smp=0.0040  enc_t=0.2246  bnd_t=0.0343  wm_t=0.7342  crt_t=0.0059  diag_t=0.3838
        charts: 16/16 active  c0=0.10 c1=0.12 c2=0.08 c3=0.09 c4=0.04 c5=0.03 c6=0.03 c7=0.03 c8=0.06 c9=0.05 c10=0.05 c11=0.07 c12=0.04 c13=0.06 c14=0.09 c15=0.07
        symbols: 84/128 active  c0=5/8(H=1.24) c1=6/8(H=1.30) c2=7/8(H=1.57) c3=5/8(H=1.09) c4=4/8(H=0.94) c5=5/8(H=1.22) c6=5/8(H=1.13) c7=5/8(H=1.17) c8=3/8(H=1.07) c9=5/8(H=1.20) c10=5/8(H=1.51) c11=5/8(H=1.00) c12=7/8(H=1.66) c13=5/8(H=1.12) c14=6/8(H=1.54) c15=6/8(H=1.53)
E0215 [5upd]  ep_rew=13.4912  rew_20=14.5898  L_geo=0.6104  L_rew=0.0004  L_chart=1.7861  L_crit=0.3079  L_bnd=0.4346  lr=0.0010  dt=12.60s
        recon=0.1879  vq=0.2355  code_H=1.3893  code_px=4.0195  ch_usage=0.0333  rtr_mrg=0.0045  enc_gn=5.0030
        ctrl=0.0000  tex=0.0298  im_rew=0.0383  im_ret=0.5409  value=0.0061  wm_gn=0.2516
        z_norm=0.6298  z_max=0.9192  jump=0.0000  cons=0.0939  sol=0.9953  e_var=0.0003  ch_ent=2.6604  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5409  dret=0.5359  term=0.0058  bnd=0.0093  chart_acc=0.4525  chart_ent=1.7533  rw_drift=0.0000
        v_err=1.5041  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8195
        bnd_x=0.0108  bell=0.0415  bell_s=0.0278  rtg_e=1.5041  rtg_b=-1.5041  cal_e=1.5041  u_l2=0.0896  cov_n=0.0005
        col=7.2394  smp=0.0033  enc_t=0.2255  bnd_t=0.0345  wm_t=0.7193  crt_t=0.0061  diag_t=0.3753
        charts: 16/16 active  c0=0.13 c1=0.06 c2=0.08 c3=0.04 c4=0.07 c5=0.07 c6=0.06 c7=0.02 c8=0.02 c9=0.12 c10=0.07 c11=0.04 c12=0.03 c13=0.06 c14=0.08 c15=0.07
        symbols: 88/128 active  c0=5/8(H=1.32) c1=6/8(H=1.50) c2=6/8(H=1.36) c3=4/8(H=0.61) c4=6/8(H=1.31) c5=5/8(H=1.37) c6=6/8(H=1.28) c7=6/8(H=1.61) c8=4/8(H=1.00) c9=6/8(H=1.27) c10=7/8(H=1.83) c11=5/8(H=1.46) c12=6/8(H=1.28) c13=5/8(H=1.11) c14=5/8(H=1.41) c15=6/8(H=1.25)
E0216 [5upd]  ep_rew=14.3309  rew_20=13.4365  L_geo=0.5649  L_rew=0.0002  L_chart=1.6811  L_crit=0.2582  L_bnd=0.3934  lr=0.0010  dt=19.35s
        recon=0.1585  vq=0.2387  code_H=1.3035  code_px=3.6891  ch_usage=0.0599  rtr_mrg=0.0027  enc_gn=7.5257
        ctrl=0.0001  tex=0.0284  im_rew=0.0405  im_ret=0.5757  value=0.0109  wm_gn=0.2072
        z_norm=0.6359  z_max=0.8903  jump=0.0000  cons=0.1101  sol=0.9958  e_var=0.0004  ch_ent=2.6439  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5757  dret=0.5662  term=0.0111  bnd=0.0168  chart_acc=0.4533  chart_ent=1.7725  rw_drift=0.0000
        v_err=1.3647  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8306
        bnd_x=0.0183  bell=0.0375  bell_s=0.0210  rtg_e=1.3647  rtg_b=-1.3647  cal_e=1.3647  u_l2=0.1179  cov_n=0.0010
        col=11.8499  smp=0.0042  enc_t=0.2576  bnd_t=0.0406  wm_t=1.0614  crt_t=0.0090  diag_t=0.5848
        charts: 16/16 active  c0=0.09 c1=0.02 c2=0.09 c3=0.09 c4=0.06 c5=0.06 c6=0.03 c7=0.04 c8=0.05 c9=0.04 c10=0.05 c11=0.09 c12=0.04 c13=0.03 c14=0.15 c15=0.08
        symbols: 77/128 active  c0=5/8(H=1.06) c1=4/8(H=1.11) c2=6/8(H=1.11) c3=3/8(H=0.17) c4=5/8(H=0.80) c5=5/8(H=0.92) c6=4/8(H=0.92) c7=6/8(H=1.49) c8=4/8(H=0.92) c9=5/8(H=0.76) c10=4/8(H=0.83) c11=4/8(H=1.16) c12=7/8(H=1.48) c13=5/8(H=0.87) c14=5/8(H=1.11) c15=5/8(H=1.03)
E0217 [5upd]  ep_rew=14.2721  rew_20=14.0724  L_geo=0.6673  L_rew=0.0006  L_chart=1.8503  L_crit=0.2614  L_bnd=0.4205  lr=0.0010  dt=18.68s
        recon=0.1827  vq=0.2341  code_H=1.4274  code_px=4.1812  ch_usage=0.0388  rtr_mrg=0.0021  enc_gn=6.1472
        ctrl=0.0001  tex=0.0282  im_rew=0.0327  im_ret=0.4679  value=0.0122  wm_gn=0.2095
        z_norm=0.6484  z_max=0.9220  jump=0.0000  cons=0.0979  sol=0.9920  e_var=0.0005  ch_ent=2.6618  ch_act=16.0000  rtr_conf=1.0000
        obj=0.4679  dret=0.4575  term=0.0121  bnd=0.0227  chart_acc=0.4188  chart_ent=1.8166  rw_drift=0.0000
        v_err=1.2405  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8282
        bnd_x=0.0257  bell=0.0352  bell_s=0.0326  rtg_e=1.2405  rtg_b=-1.2405  cal_e=1.2405  u_l2=0.0978  cov_n=0.0011
        col=11.3462  smp=0.0052  enc_t=0.2546  bnd_t=0.0399  wm_t=1.0346  crt_t=0.0088  diag_t=0.5760
        charts: 16/16 active  c0=0.07 c1=0.02 c2=0.05 c3=0.05 c4=0.04 c5=0.14 c6=0.07 c7=0.05 c8=0.05 c9=0.04 c10=0.12 c11=0.06 c12=0.05 c13=0.04 c14=0.09 c15=0.07
        symbols: 87/128 active  c0=6/8(H=1.30) c1=5/8(H=1.51) c2=6/8(H=1.47) c3=3/8(H=0.54) c4=6/8(H=1.00) c5=6/8(H=1.44) c6=5/8(H=0.95) c7=7/8(H=1.43) c8=6/8(H=0.97) c9=5/8(H=1.02) c10=5/8(H=1.44) c11=6/8(H=1.45) c12=7/8(H=1.17) c13=5/8(H=1.24) c14=5/8(H=1.30) c15=4/8(H=1.26)
E0218 [5upd]  ep_rew=15.8090  rew_20=13.8927  L_geo=0.7532  L_rew=0.0004  L_chart=1.8304  L_crit=0.2985  L_bnd=0.4288  lr=0.0010  dt=18.60s
        recon=0.1824  vq=0.2523  code_H=1.3048  code_px=3.6961  ch_usage=0.0344  rtr_mrg=0.0027  enc_gn=5.9305
        ctrl=0.0000  tex=0.0252  im_rew=0.0408  im_ret=0.5792  value=0.0103  wm_gn=0.2169
        z_norm=0.6737  z_max=0.9365  jump=0.0000  cons=0.0831  sol=0.9967  e_var=0.0021  ch_ent=2.5635  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5792  dret=0.5706  term=0.0100  bnd=0.0151  chart_acc=0.3950  chart_ent=1.8408  rw_drift=0.0000
        v_err=1.2978  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.8367
        bnd_x=0.0183  bell=0.0373  bell_s=0.0344  rtg_e=1.2978  rtg_b=-1.2978  cal_e=1.2978  u_l2=0.0775  cov_n=0.0007
        col=11.3021  smp=0.0041  enc_t=0.2547  bnd_t=0.0405  wm_t=1.0263  crt_t=0.0085  diag_t=0.5728
        charts: 16/16 active  c0=0.13 c1=0.05 c2=0.09 c3=0.01 c4=0.06 c5=0.19 c6=0.02 c7=0.04 c8=0.03 c9=0.03 c10=0.06 c11=0.03 c12=0.05 c13=0.04 c14=0.09 c15=0.07
        symbols: 94/128 active  c0=6/8(H=1.35) c1=5/8(H=1.07) c2=5/8(H=1.32) c3=5/8(H=1.22) c4=8/8(H=1.55) c5=6/8(H=1.21) c6=5/8(H=1.32) c7=8/8(H=1.66) c8=5/8(H=0.82) c9=6/8(H=1.30) c10=5/8(H=1.54) c11=7/8(H=1.10) c12=6/8(H=1.49) c13=5/8(H=1.30) c14=6/8(H=1.34) c15=6/8(H=1.53)
E0219 [5upd]  ep_rew=14.9909  rew_20=14.1499  L_geo=0.5464  L_rew=0.0002  L_chart=1.5641  L_crit=0.2884  L_bnd=0.3915  lr=0.0010  dt=20.32s
        recon=0.1714  vq=0.2900  code_H=1.2974  code_px=3.6817  ch_usage=0.0553  rtr_mrg=0.0038  enc_gn=8.7067
        ctrl=0.0001  tex=0.0275  im_rew=0.0440  im_ret=0.6289  value=0.0155  wm_gn=0.1899
        z_norm=0.6692  z_max=0.9066  jump=0.0000  cons=0.1028  sol=0.9996  e_var=0.0004  ch_ent=2.6350  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6289  dret=0.6154  term=0.0157  bnd=0.0219  chart_acc=0.4900  chart_ent=1.7756  rw_drift=0.0000
        v_err=1.4224  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7857
        bnd_x=0.0253  bell=0.0385  bell_s=0.0156  rtg_e=1.4224  rtg_b=-1.4224  cal_e=1.4224  u_l2=0.0635  cov_n=0.0011
        col=12.1913  smp=0.0037  enc_t=0.2645  bnd_t=0.0420  wm_t=1.1698  crt_t=0.0097  diag_t=0.6253
        charts: 16/16 active  c0=0.10 c1=0.03 c2=0.06 c3=0.01 c4=0.11 c5=0.11 c6=0.04 c7=0.05 c8=0.02 c9=0.08 c10=0.06 c11=0.07 c12=0.10 c13=0.03 c14=0.08 c15=0.06
        symbols: 92/128 active  c0=7/8(H=1.56) c1=6/8(H=1.41) c2=5/8(H=1.24) c3=5/8(H=1.06) c4=6/8(H=1.47) c5=7/8(H=1.38) c6=4/8(H=1.01) c7=6/8(H=1.51) c8=5/8(H=1.15) c9=6/8(H=1.14) c10=6/8(H=1.57) c11=6/8(H=1.38) c12=6/8(H=1.49) c13=5/8(H=0.85) c14=6/8(H=1.37) c15=6/8(H=1.34)
E0220 [5upd]  ep_rew=8.2083  rew_20=14.3119  L_geo=0.5696  L_rew=0.0002  L_chart=1.6640  L_crit=0.2984  L_bnd=0.4002  lr=0.0010  dt=15.25s
        recon=0.1739  vq=0.2638  code_H=1.3113  code_px=3.7158  ch_usage=0.0297  rtr_mrg=0.0026  enc_gn=9.0800
        ctrl=0.0001  tex=0.0294  im_rew=0.0357  im_ret=0.5176  value=0.0214  wm_gn=0.1621
        z_norm=0.6307  z_max=0.8993  jump=0.0000  cons=0.0889  sol=0.9996  e_var=0.0002  ch_ent=2.7003  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5176  dret=0.4998  term=0.0207  bnd=0.0357  chart_acc=0.4454  chart_ent=1.7219  rw_drift=0.0000
        v_err=1.4561  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7917
        bnd_x=0.0402  bell=0.0403  bell_s=0.0278  rtg_e=1.4561  rtg_b=-1.4561  cal_e=1.4561  u_l2=0.0503  cov_n=0.0014
        col=9.9456  smp=0.0025  enc_t=0.2266  bnd_t=0.0343  wm_t=0.7076  crt_t=0.0061  diag_t=0.3714
        charts: 16/16 active  c0=0.10 c1=0.05 c2=0.04 c3=0.05 c4=0.09 c5=0.09 c6=0.03 c7=0.04 c8=0.10 c9=0.08 c10=0.07 c11=0.03 c12=0.05 c13=0.09 c14=0.04 c15=0.05
        symbols: 87/128 active  c0=6/8(H=1.59) c1=5/8(H=1.14) c2=5/8(H=1.31) c3=6/8(H=1.03) c4=7/8(H=1.48) c5=5/8(H=1.23) c6=4/8(H=0.89) c7=6/8(H=1.12) c8=6/8(H=0.83) c9=6/8(H=1.18) c10=5/8(H=1.45) c11=5/8(H=1.38) c12=6/8(H=0.99) c13=5/8(H=1.14) c14=5/8(H=1.01) c15=5/8(H=1.55)
E0221 [5upd]  ep_rew=14.3382  rew_20=14.8157  L_geo=0.6310  L_rew=0.0004  L_chart=1.6544  L_crit=0.2832  L_bnd=0.4252  lr=0.0010  dt=12.98s
        recon=0.1421  vq=0.2925  code_H=1.3192  code_px=3.7513  ch_usage=0.0236  rtr_mrg=0.0050  enc_gn=7.4954
        ctrl=0.0001  tex=0.0306  im_rew=0.0431  im_ret=0.6249  value=0.0239  wm_gn=0.2193
        z_norm=0.6380  z_max=0.9079  jump=0.0000  cons=0.0806  sol=0.9952  e_var=0.0004  ch_ent=2.6765  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6249  dret=0.6038  term=0.0245  bnd=0.0350  chart_acc=0.4729  chart_ent=1.6524  rw_drift=0.0000
        v_err=1.5372  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.7276
        bnd_x=0.0368  bell=0.0423  bell_s=0.0271  rtg_e=1.5372  rtg_b=-1.5372  cal_e=1.5372  u_l2=0.0424  cov_n=0.0014
        col=7.3988  smp=0.0033  enc_t=0.2299  bnd_t=0.0347  wm_t=0.7477  crt_t=0.0064  diag_t=0.4263
        charts: 16/16 active  c0=0.11 c1=0.03 c2=0.07 c3=0.05 c4=0.10 c5=0.03 c6=0.05 c7=0.03 c8=0.08 c9=0.11 c10=0.04 c11=0.06 c12=0.06 c13=0.05 c14=0.09 c15=0.04
        symbols: 96/128 active  c0=5/8(H=1.35) c1=6/8(H=1.20) c2=7/8(H=1.61) c3=7/8(H=1.45) c4=7/8(H=1.44) c5=4/8(H=0.97) c6=6/8(H=1.34) c7=8/8(H=1.52) c8=6/8(H=1.43) c9=5/8(H=1.03) c10=6/8(H=1.52) c11=6/8(H=1.51) c12=7/8(H=1.04) c13=5/8(H=1.15) c14=6/8(H=1.20) c15=5/8(H=1.17)
E0222 [5upd]  ep_rew=6.4878  rew_20=14.0837  L_geo=0.5617  L_rew=0.0004  L_chart=1.6914  L_crit=0.2716  L_bnd=0.4290  lr=0.0010  dt=13.14s
        recon=0.1713  vq=0.2548  code_H=1.2751  code_px=3.5846  ch_usage=0.0357  rtr_mrg=0.0012  enc_gn=7.3175
        ctrl=0.0001  tex=0.0282  im_rew=0.0421  im_ret=0.6065  value=0.0207  wm_gn=0.2401
        z_norm=0.6661  z_max=0.9229  jump=0.0000  cons=0.0905  sol=0.9959  e_var=0.0004  ch_ent=2.5930  ch_act=16.0000  rtr_conf=1.0000
        obj=0.6065  dret=0.5895  term=0.0198  bnd=0.0289  chart_acc=0.4646  chart_ent=1.7270  rw_drift=0.0000
        v_err=1.2813  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.6685
        bnd_x=0.0327  bell=0.0351  bell_s=0.0180  rtg_e=1.2813  rtg_b=-1.2813  cal_e=1.2813  u_l2=0.0342  cov_n=0.0012
        col=7.6406  smp=0.0038  enc_t=0.2280  bnd_t=0.0345  wm_t=0.7365  crt_t=0.0063  diag_t=0.4102
        charts: 16/16 active  c0=0.05 c1=0.05 c2=0.10 c3=0.02 c4=0.08 c5=0.14 c6=0.03 c7=0.03 c8=0.03 c9=0.11 c10=0.04 c11=0.08 c12=0.04 c13=0.03 c14=0.14 c15=0.03
        symbols: 88/128 active  c0=4/8(H=0.48) c1=5/8(H=1.20) c2=6/8(H=1.27) c3=5/8(H=1.07) c4=6/8(H=1.03) c5=6/8(H=0.99) c6=5/8(H=1.07) c7=5/8(H=1.13) c8=5/8(H=1.14) c9=4/8(H=0.43) c10=5/8(H=1.41) c11=6/8(H=1.44) c12=8/8(H=1.43) c13=5/8(H=0.76) c14=7/8(H=1.33) c15=6/8(H=1.49)
E0223 [5upd]  ep_rew=15.7391  rew_20=13.8415  L_geo=0.5525  L_rew=0.0003  L_chart=1.6473  L_crit=0.3169  L_bnd=0.4562  lr=0.0010  dt=13.07s
        recon=0.1784  vq=0.2439  code_H=1.3213  code_px=3.7753  ch_usage=0.0711  rtr_mrg=0.0020  enc_gn=8.0784
        ctrl=0.0001  tex=0.0313  im_rew=0.0416  im_ret=0.5995  value=0.0190  wm_gn=0.1823
        z_norm=0.5923  z_max=0.8862  jump=0.0000  cons=0.1252  sol=0.9845  e_var=0.0003  ch_ent=2.6704  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5995  dret=0.5828  term=0.0194  bnd=0.0286  chart_acc=0.4679  chart_ent=1.7024  rw_drift=0.0000
        v_err=1.4726  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.4762
        bnd_x=0.0315  bell=0.0418  bell_s=0.0294  rtg_e=1.4726  rtg_b=-1.4726  cal_e=1.4726  u_l2=0.0378  cov_n=0.0014
        col=7.5016  smp=0.0040  enc_t=0.2325  bnd_t=0.0346  wm_t=0.7494  crt_t=0.0065  diag_t=0.3935
        charts: 16/16 active  c0=0.04 c1=0.06 c2=0.09 c3=0.07 c4=0.06 c5=0.08 c6=0.09 c7=0.05 c8=0.05 c9=0.06 c10=0.03 c11=0.11 c12=0.07 c13=0.01 c14=0.11 c15=0.03
        symbols: 87/128 active  c0=4/8(H=0.96) c1=6/8(H=1.50) c2=7/8(H=1.17) c3=4/8(H=0.88) c4=7/8(H=1.42) c5=6/8(H=1.11) c6=7/8(H=1.41) c7=6/8(H=1.57) c8=3/8(H=0.96) c9=6/8(H=1.50) c10=4/8(H=0.67) c11=5/8(H=1.02) c12=7/8(H=1.23) c13=4/8(H=1.27) c14=6/8(H=1.33) c15=5/8(H=1.42)
E0224 [5upd]  ep_rew=34.9221  rew_20=14.8487  L_geo=0.6290  L_rew=0.0005  L_chart=1.8495  L_crit=0.2795  L_bnd=0.4585  lr=0.0010  dt=21.12s
        recon=0.1838  vq=0.2216  code_H=1.3997  code_px=4.0554  ch_usage=0.0354  rtr_mrg=0.0019  enc_gn=6.9895
        ctrl=0.0001  tex=0.0302  im_rew=0.0398  im_ret=0.5686  value=0.0145  wm_gn=0.2924
        z_norm=0.6187  z_max=0.9577  jump=0.0000  cons=0.1142  sol=0.9950  e_var=0.0003  ch_ent=2.6848  ch_act=16.0000  rtr_conf=1.0000
        obj=0.5686  dret=0.5567  term=0.0139  bnd=0.0214  chart_acc=0.4021  chart_ent=1.7262  rw_drift=0.0000
        v_err=1.4450  a_sat=0.0000  wm_ctr=0.0000  enc_ctr=0.0000  dec_ctr=0.0000  crt_ctr=0.0000  u_cos=0.5272
        bnd_x=0.0252  bell=0.0414  bell_s=0.0438  rtg_e=1.4450  rtg_b=-1.4450  cal_e=1.4450  u_l2=0.0447  cov_n=0.0016
        col=12.3691  smp=0.0048  enc_t=0.2741  bnd_t=0.0440  wm_t=1.2594  crt_t=0.0104  diag_t=0.7319
        charts: 16/16 active  c0=0.03 c1=0.03 c2=0.05 c3=0.05 c4=0.04 c5=0.08 c6=0.04 c7=0.04 c8=0.05 c9=0.06 c10=0.07 c11=0.10 c12=0.05 c13=0.08 c14=0.09 c15=0.13
        symbols: 87/128 active  c0=4/8(H=1.02) c1=5/8(H=1.18) c2=6/8(H=1.26) c3=4/8(H=0.83) c4=8/8(H=1.35) c5=5/8(H=0.78) c6=4/8(H=1.11) c7=7/8(H=1.69) c8=4/8(H=1.08) c9=5/8(H=0.95) c10=6/8(H=1.64) c11=4/8(H=1.00) c12=8/8(H=1.38) c13=6/8(H=1.39) c14=5/8(H=1.37) c15=6/8(H=1.21)

