# Kimera

Depot local : `third_party/Kimera`.

Statut : meta-depot pour un pipeline VIO/meshing/semantique. Ce n'est pas un binaire TUM RGB-D de type `rgbd_tum`, et la comparaison directe avec ORB-SLAM2/3 demande de choisir un sous-systeme Kimera et un format de dataset compatible.

Comparaison recommandee :

- choisir le sous-pipeline a evaluer, par exemple Kimera-VIO ou Kimera-Semantics ;
- preparer un dataset compatible avec ses entrees attendues ;
- comparer la trajectoire, puis separer les couts VIO, meshing et semantique.

