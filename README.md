# **PFE2019#45 - Segmentation de scènes dans des vidéos événementielles**

## **Introduction**

Nous nous intéressons dans ce projet à des données issues de caméras évènementielles de type DVS, qui encodent les variations de luminosité (positives ou négatives) indépendamment pour chaque pixel. Ainsi, chaque variation v positive ou négative d’un pixel (x,y) à l’instant (t) se traduit par un événement (x,y,t,v) transmis de manière asynchrone (au format Address-Event Representation, AER). Par conséquent, une scène immobile ne génèrera aucun événement (autrement dit, on ne “voit” que les mouvements), ce qui élimine une grande partie de redondance dans l’information. Par ailleurs, ces capteurs ont l'avantage de bénéficier d'un High Dynamic Range (HDR), ainsi que d'une haute résolution temporelle de l'ordre de la microseconde, ce qui  les rend intéressants pour des applications de robotique, pour les drones et pour les véhicules autonomes.

Il s'agit dans ce projet de proposer une méthode permettant la segmentation d'une scène acquise avec une caméra événementielle. La segmentation consiste à partitionner la scène en régions correspondant aux différents objets ou plans de l'image. Une analyse des méthodes existantes est aussi nécessaire. En s'inspirant de l'existant, nous devons proposer et implémenter une méthode de segmentation, puis la tester sur le jeu de données DDD17. 
