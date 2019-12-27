# **PFE2019#45 - Segmentation de scènes dans des vidéos événementielles**

## **Introduction**

Nous nous intéressons dans ce projet à des données issues de caméras évènementielles de type DVS, qui encodent les variations de luminosité (positives ou négatives) indépendamment pour chaque pixel. Ainsi, chaque variation v positive ou négative d’un pixel (x,y) à l’instant (t) se traduit par un événement (x,y,t,v) transmis de manière asynchrone (au format Address-Event Representation, AER). Par conséquent, une scène immobile ne génèrera aucun événement (autrement dit, on ne “voit” que les mouvements), ce qui élimine une grande partie de redondance dans l’information. Par ailleurs, ces capteurs ont l'avantage de bénéficier d'un High Dynamic Range (HDR), ainsi que d'une haute résolution temporelle de l'ordre de la microseconde, ce qui  les rend intéressants pour des applications de robotique, pour les drones et pour les véhicules autonomes.

Il s'agit dans ce projet de proposer une méthode permettant la segmentation d'une scène acquise avec une caméra événementielle. La segmentation consiste à partitionner la scène en régions correspondant aux différents objets ou plans de l'image. Une analyse des méthodes existantes est aussi nécessaire. En s'inspirant de l'existant, nous devons proposer et implémenter une méthode de segmentation, puis la tester sur le jeu de données DDD17. 


Comme nous somme en deux, notre travail est fait ensemble hors ligne.


## **User manual**
If you want to directly use our dataset, please start from 6.
1. Get the DDD17 dataset from https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub (dataset information included)
2. Extract event camara data
    - Environment:   
    Python 2.7+
    - Export only event data of a record:    
    
Option: -s 50%% - play file starting at 50%%    
-s 66s - play file starting at 66s    
-r (False by default), Rotate the scene 180 degrees if True, otherwise False    
To change the out file path, modify the value of exported_h5_path in line 38.    
```$ python export_ddd17_hdf.py <recorded_file.hdf5> ```


3. Export selected sequence from the exported event file
    - Environment:    
    Python 3.6+
    - Get the sequence of a record:    
To indicate the sequence, output path and the record file, please modify the corresponding value in the python file.       
```$ python data_select.py```

4. Our implementation of the data representation of Ev-SegNet    
    - Environment:    
    Python 3.6+
    - Get the data representation of the first 50ms of a record:     
```$ python evsegnet_preprocess.py <recorded_file.hdf5> ```  

5. Our data representation
    - Environment:    
    Python 3.6+
    - Get the data representation of a record:     
For 6-channel representation:     
```$ python our_data_process.py```     
For 3-channel representation:      
```$ python our_data_process_3.py```      
To indicate the output path and the record file, please modify the corresponding value in the python file.     
6. Ev-SegNet
    - Environment:      
    Python 2.7+ with TensorFlow ==1.11, Opencv, Keras, Imgaug and Sklearn
    - Replicate results:      
```$ python Ev-SegNet/train_eager.py --epochs 0```    
    - Train from scratch:     
```$ python Ev-SegNet/train_eager.py --epochs 500 --dataset path_to_dataset  --model_path path_to_model  --batch_size 8```   

7. MobileNetV2-UNet    
    - Environment: Python 2.7+ with TensorFlow ==1.14, Opencv, Keras, Imgaug and Sklearn   
    - Replicate results:    
```$ python Ev-SegNet/train_unet.py --epochs 0```    
    - Train from scratch:    
```$ python Ev-SegNet/train_unet.py --epochs 500 --dataset path_to_dataset  --model_path path_to_model  --batch_size 8```    






---------------------



