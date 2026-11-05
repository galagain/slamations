# iBoW-LCD

Depot local : `third_party/ibow-lcd`.

Statut : brique de reconnaissance de lieu / detection de fermeture de boucle, pas un SLAM complet. Elle ne produit pas directement une trajectoire TUM RGB-D comparable a ORB-SLAM2/3.

Comparaison recommandee :

- benchmarker le temps de requete et la qualite des candidats de boucle ;
- l'integrer ensuite dans le module de loop closing d'un SLAM complet ;
- mesurer sur une sequence avec boucle, par exemple `rgbd_dataset_freiburg2_large_with_loop`.

