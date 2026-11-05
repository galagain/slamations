# Isaac ROS Visual SLAM

Depot local : `third_party/isaac_ros_visual_slam`.

Statut : stack ROS 2/cuVSLAM. Elle ne s'execute pas comme les exemples ORB-SLAM sur un dossier TUM RGB-D brut; il faut fournir des topics ROS 2, des timestamps compatibles et une configuration Isaac ROS/NITROS.

Comparaison recommandee :

- convertir ou rejouer la sequence sous forme de rosbag ROS 2 ;
- lancer le node Isaac ROS Visual SLAM dans son conteneur dedie ;
- enregistrer la trajectoire publiee et mesurer ATE/RPE sur les timestamps associes.

