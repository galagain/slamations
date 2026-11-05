# gpu-block-solver

Depot local : `third_party/gpu-block-solver`.

Statut : brique numerique, pas un SLAM complet. Ce depot remplace/accelere une partie du solveur g2o et doit etre integre dans ORB-SLAM3, TurboMap ou une variante equivalente avant de produire une trajectoire TUM, un FPS ou des metriques ATE/RPE.

Comparaison recommandee :

- mesurer d'abord le LBA CPU dans ORB-SLAM3 baseline ;
- integrer le solver GPU dans la branche ORB-SLAM3/TurboMap choisie ;
- relancer les memes sequences TUM ;
- comparer le temps `LBA_Stats.txt`, le temps de local mapping et les metriques ATE/RPE.

